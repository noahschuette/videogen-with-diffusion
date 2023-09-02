import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms
import clip
from PIL import Image
import os
import torch.nn as nn
from statistics import mean
from itertools import pairwise
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

methods  = ["ControlVideo","Tune-A-Video","SD-Dino_crop"] #methods to test: requires folder structure: method_name/folder_with_video/frame.jpg

open('/media/compute/homes/nschuette/bachelor/metrices/frame_consistency.csv', 'w').close()
open('/media/compute/homes/nschuette/bachelor/metrices/perceptual_consistency.csv', 'w').close()
open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'w').close()

# dict of prompts for each video folder
prompts = {
    "blackcar_whitecar" : "White Car",
    "dog_blackdog" : "Black Dog",
    "hike_snowboarder" : "Snowboarder with yellow jacket is hiking",
    "hike_tennis" : "Tennis player is hiking",
    "polar_bear" : "Polar bear walking",
    "rollerblade_hike" : "Hiker is inline skating",
    "rollerblade_snowboarder" : "Snowboarder with yellow jacket is inline skating",
    "tennis_hike" : "Hiker is playing tennis",
    "tennis_snowboarder" : "Snowboarder with yellow jacket is playing tennis",
    "whitecar_blackcar" : "Black Car"
}

transform = transforms.Compose([ transforms.ToTensor() ])
cutchannel = transforms.Lambda(lambda x: x[:3])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
def to_tensor(img):
    img = Image.open(img)
    tensor = transform(img)
    if tensor.shape[0] == 4:
        #convert the image from RGBA2RGB
        tensor = cutchannel(tensor)
    return tensor

def img_similarity(dir):
    images = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                images.append(  root  + '/'+ file)
    cos = torch.nn.CosineSimilarity(dim=0)
    similarities = []
    for image1, image2 in pairwise(images):
        #print(f"compute {image1}, {image2}")
        with torch.no_grad():
            image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
            image1_features = model.encode_image( image1_preprocess)

            image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
            image2_features = model.encode_image( image2_preprocess)

            similarity = cos(image1_features[0],image2_features[0]).item()
            similarity = (similarity+1)/2
            #if (similarity < min):
            #    min = similarity
            #if (similarity > max):
            #    max = similarity
            similarities.append(similarity * 100)
    return mean(similarities)

# Calculate similarity between prompt & each image from dir 
# CLIP score between text & image adapted from https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html
def clip_score(dir):
    images = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                images.append( root  + '/'+ file)

    text = prompts[dir.split("/")[1]] # load prompt from dict where folder name = dict key
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    result = []
    for img in images:
        tensor = to_tensor(img)
        score = metric(tensor,text) # calculate metric between image I, text C: CLIPScore(I, C) = max(100 * cos(E_I, E_C), 0)
        score.detach()
        score = score.item() 
        result.append(score)
        #print("score",score)
    return mean(result)

# Calculate patch image similarity between for each pair of two images from dir 
# adapted from https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
def patch_similarity(dir):
    images = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                images.append( root  + '/'+ file)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True)
    result = []
    for img1, img2 in pairwise(images):
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        t1 = rearrange(t1, 'c w h -> 1 c w h') #extend to 4-channel tensor
        t2 = rearrange(t2, 'c w h -> 1 c w h')
        score = lpips_metric(t1, t2)
        score.detach()
        result.append(score.item() * 100)
    return mean(result)


def main():
    for method in methods:

        method_clip_similarities = []
        method_perc_similarities = []
        method_prompt_similarities = []

        for dir in os.listdir(method):

            print(f"Computing video {dir} for {method}")

            avrg_clip_sim = img_similarity(f"{method}/{dir}")
            method_clip_similarities.append(avrg_clip_sim)

            avrg_sim = patch_similarity(f"{method}/{dir}")
            method_perc_similarities.append(avrg_sim)

            text_sim = clip_score(f"{method}/{dir}")
            method_prompt_similarities.append(text_sim)

            # save dict to csv
            with open('/media/compute/homes/nschuette/bachelor/metrices/frame_consistency.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, dir, round(avrg_clip_sim,2)))

            with open('/media/compute/homes/nschuette/bachelor/metrices/perceptual_consistency.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, dir, round(avrg_sim,2)))

            with open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, dir, round(text_sim,2)))

            torch.cuda.empty_cache()

        avg = mean(method_clip_similarities)
        with open('/media/compute/homes/nschuette/bachelor/metrices/frame_consistency.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, "AVERAGE", round(avg,2)))

        avg = mean(method_perc_similarities)
        with open('/media/compute/homes/nschuette/bachelor/metrices/perceptual_consistency.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, "AVERAGE", round(avg,2)))

        avg = mean(method_prompt_similarities)
        with open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, "AVERAGE", round(avg,2)))

if __name__ == '__main__':
    main()
