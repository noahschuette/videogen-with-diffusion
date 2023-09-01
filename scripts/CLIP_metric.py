import torch
import clip
from PIL import Image
import os
import torch.nn as nn
from statistics import mean
from itertools import pairwise
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-S/32", device=device)

methods  = ["SD-Dino","TuneAVideo","ControlVideo"] #methods to test: requires folder structure: method_name/folder_with_video/frame.jpg

for method in methods:

    for _, dirs, _ in os.walk(method):
        for dir in dirs:
            images = []
            for root, _, files in os.walk(dir):
                for file in files:
                    if file.endswith('jpg') or file.endswith('png'):
                        images.append(  root  + '/'+ file)

            cos = torch.nn.CosineSimilarity(dim=0)
            similarities = []

            for image1, image2 in pairwise(images):
                with torch.no_grad():
                    image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
                    image1_features = model.encode_image( image1_preprocess)

                    image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
                    image2_features = model.encode_image( image2_preprocess)

                    similarity = cos(image1_features[0],image2_features[0]).item()
                    similarity = (similarity+1)/2
                    similarities.append(similarity)
            avrg_sim = mean(similarities)
            print("Frame CLIP similarity: ", avrg_sim)

            dict = {"CLIP": avrg_sim, "folder": dir, "method": method}

            # save dict to csv
            with open('clip.csv', 'a') as f:
                f.write("%s,'%s',%s\n" % (dict["method"], dict["folder"], dict["CLIP"]))

            torch.cuda.empty_cache()
