# adapted from https://medium.com/@jeremy-k/unlocking-openai-clip-part-1-intro-to-zero-shot-classification-f81194f4dff7 and https://medium.com/@jeremy-k/unlocking-openai-clip-part-2-image-similarity-bf0224ab5bb0

import torch
import clip
from PIL import Image
import os
import torch.nn as nn
from statistics import mean
from itertools import pairwise
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

methods  = ["ControlVideo","Tune-A-Video","SD-Dino"] #methods to test: requires folder structure: method_name/folder_with_video/frame.jpg

open('/media/compute/homes/nschuette/bachelor/metrices/clip.csv', 'w').close()
open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'w').close()

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

def img_similarity(method, dir):
    print(f"Computing video {dir} for {method}")
    images = []
    for root, _, files in os.walk(f"{method}/{dir}"):
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
            similarities.append(similarity)
    return mean(similarities)

def txt_similarity(method, dir):
    images = []
    for root, _, files in os.walk(f"{method}/{dir}"):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                images.append(  root  + '/'+ file)

    text = clip.tokenize([prompts[dir]]).to(device)
    print(f"Computing video {dir} for {method} with text {prompts[dir]}")
    text_features = model.encode_text(text)
    result = {}
    cos = torch.nn.CosineSimilarity(dim=0)
    #For each image, compute its cosine similarity with the prompt and store the result in a dict
    for img in images:
        with torch.no_grad():
            image_preprocess = preprocess(Image.open(img)).unsqueeze(0).to(device)
            image_features = model.encode_image( image_preprocess)
            sim = cos(image_features[0],text_features[0]).item()
            sim = (sim+1)/2
            result[img]=sim

    return mean(result.values())

def main():
    for method in methods:

        method_similarities = []
        method_prompt_similarities = []

        for dir in os.listdir(method):

            avrg_sim = img_similarity(method,dir)
            method_similarities.append(avrg_sim)

            text_sim = txt_similarity(method,dir)
            method_prompt_similarities.append(text_sim)

            # save dict to csv
            with open('/media/compute/homes/nschuette/bachelor/metrices/clip.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, dir, round(avrg_sim*100,2)))

            with open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, dir, round(text_sim*100,2)))

            torch.cuda.empty_cache()

        avg = mean(method_similarities)
        with open('/media/compute/homes/nschuette/bachelor/metrices/clip.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, "AVERAGE", round(avg*100,2)))
                #f.write("%s,'%s',%s\n" % (method, "MIN", min))
                #f.write("%s,'%s',%s\n" % (method, "MAX", max))

        avg = mean(method_prompt_similarities)
        with open('/media/compute/homes/nschuette/bachelor/metrices/clip_prompt.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (method, "AVERAGE", round(avg*100,2)))

if __name__ == '__main__':
    main()
