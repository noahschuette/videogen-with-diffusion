# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
import os
import torch
import cv2
import numpy as np
import requests
import io
import base64
from PIL import Image, PngImagePlugin

inpaint = True
imgNumFrom = 5
imgNumTo = 24

prompt = "snowboarder, yellow jacket, black helmet"

folder = "/media/compute/homes/nschuette/bachelor/sd-dino"
maskPath = f"{folder}/masks/"
referencePath = f"{folder}/img2img/detailed.png"
outputPath = f"{folder}/output/"

img_extension = ".png"
mask_extension = ".png"

# Erstelle Ordner für die Abspeicherung
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Initzalisiere den predictor mit GPU falls kompatibel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Verwende', device, 'für das Inpainting.')

for i in range(imgNumFrom, imgNumTo+1):
    # Read an image
    s = str(i)
    img_name = s.zfill(3)
    print(img_name)

    mask_filename = img_name + "_mask" + mask_extension
    inpainted_filename = img_name + ".png"

    img = cv2.imread(maskPath + img_name + img_extension)
    img = cv2.resize(img, (512, 512))

    mask = cv2.imread(maskPath + mask_filename)
    mask = cv2.resize(mask, (512, 512))

    reference = cv2.imread(referencePath)
    reference = cv2.resize(reference, (512, 512))

    url = "http://129.70.133.115:7861"
    #print(url)

    opt = requests.get(url=f'{url}/sdapi/v1/options')
    response = opt.json()
    response['sd_model_checkpoint'] = 'dreamshaper_631Inpainting.safetensors'
    #response['sd_model_checkpoint'] = 'sd-v1-4.ckpt'
    requests.post(url=f'{url}/sdapi/v1/options', json=response)

    _, buffer = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.png', mask)
    encoded_mask = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.png', reference)
    encoded_reference = base64.b64encode(buffer).decode('utf-8')

    height, width = img.shape[:2]

    payload = {
      "init_images": [
        encoded_image
      ],
      "resize_mode": 0,
      "denoising_strength": 0.5,
      "image_cfg_scale": 7,
      "mask": encoded_mask,
      "mask_blur": 4,
      "inpainting_fill": 1, #0: fill (start with mean surrounding color), 1: original (start with reference img), 2: latent noise (start with pure noise), 3: latent nothing (start with "black" latent)
      "inpaint_full_res": False,
      "inpaint_full_res_padding": 4,
      "inpainting_mask_invert": 0,
      "initial_noise_multiplier": 0,
      "prompt": prompt,
      "seed": 240,
      "subseed": -1,
      "subseed_strength": 0,
      "seed_resize_from_h": -1,
      "seed_resize_from_w": -1,
      "sampler_name": "DDIM",
      "batch_size": 1,
      "n_iter": 1,
      "steps": 20,
      "cfg_scale": 7,
      "width": width,
      "height": height,
      "restore_faces": True,
      "tiling": False,
      "do_not_save_samples": True,
      "do_not_save_grid": True,
      "negative_prompt": "",
      "eta": 0,
      "s_churn": 0,
      "s_tmax": 0,
      "s_tmin": 0,
      "s_noise": 1,
      "override_settings": {},
      "override_settings_restore_afterwards": True,
      "sampler_index": "DDIM",
      "include_init_images": True,
      "send_images": True,
      "save_images": False,
      "alwayson_scripts": {
      "controlnet": {
      #"controlnet_units": [
      "args": [
            {
                "input_image": encoded_reference,
                "module": "reference_only"
            }
        ]
      }
      },
      
    }

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    r = response.json()

    #j = 1
    #for i in r['images']:
    i = r['images'][0]

    inpainted_image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    inpainted_image.save(os.path.join(outputPath, inpainted_filename ), pnginfo=pnginfo)
    #j += 1 m