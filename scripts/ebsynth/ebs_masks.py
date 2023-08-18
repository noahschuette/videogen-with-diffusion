#adapted from https://github.com/s9roll7/ebsynth_utility/blob/main/stage1.py (last visited 18.10.2023)
import cv2
import numpy as np
import os
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import glob
import argparse

#load torch
import torch
device="cuda"
print(torch.cuda.is_available())

#resizing image by given width & height
def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

#create masks with clipseg
def create_mask_clipseg(input_path, output_path, mask_threshold, mask_blur_size, mask_blur_size2, prompts):

    #use clipseg proc & model with cuda
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)

    images = glob.glob( os.path.join(input_path, "*.png") )

    for img_count,img in enumerate(images):

        image = Image.open(img)
        base_name = os.path.basename(img)

        inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
        inputs = inputs.to(device)

        #predict / forward-pass
        with torch.no_grad():
            outputs = model(**inputs)

        if len(prompts) == 1:
            preds = outputs.logits.unsqueeze(0)
        else:
            preds = outputs.logits

        mask_img = None

        for i in range(len(prompts)):
            x = torch.sigmoid(preds[i])
            x = x.to('cpu').detach().numpy()

            x = x > mask_threshold

            if i < len(prompts):
                if mask_img is None:
                    mask_img = x
                else:
                    mask_img = np.maximum(mask_img,x)
            else:
                mask_img[x > 0] = 0

        mask_img = mask_img*255
        mask_img = mask_img.astype(np.uint8)

        if mask_blur_size > 0:
            mask_blur_size = mask_blur_size//2 * 2 + 1
            mask_img = cv2.medianBlur(mask_img, mask_blur_size)

        if mask_blur_size2 > 0:
            mask_blur_size2 = mask_blur_size2//2 * 2 + 1
            mask_img = cv2.GaussianBlur(mask_img, (mask_blur_size2, mask_blur_size2), 0)

        mask_img = resize_img(mask_img, image.width, image.height)

        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        save_path = os.path.join(output_path, base_name)
        cv2.imwrite(save_path, mask_img)

        #print("{0} / {1}".format( img_count+1,len(images) ))

def main(args):
    prompts = ["a human","human","body","man"]
    create_mask_clipseg(input_path=args.input_path, output_path=args.output_path, threshold=0.4, median_blur_size=5, gaussian_blur_size=5, prompts=prompts)
    print("Masks generated")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Masks from input folder")
    parser.add_argument('frames_path', help='Path to the video frames folder')
    parser.add_argument('output_path', help='Output folder for the masks')
    args = parser.parse_args()
    main(args)

