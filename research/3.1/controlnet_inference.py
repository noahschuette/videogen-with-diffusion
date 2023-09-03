from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np

#load torch
import torch
device="cuda"
print(torch.cuda.is_available())

#prepare controlnet pipeline using canny, openpose & stable-diffusion-v1.5
controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16),
]
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe = pipe.to(device)

#enable model offloading & xformers for faster developement
from diffusers import UniPCMultistepScheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

#use canny edge filter on image
def apply_canny_edge(name, img, high_threshold, low_threshold, savename):
    image = np.array(img)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save("canny/"+ savename)
    return canny_image

#use openpose filter on image
def apply_openpose(name, img):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(img)
    openpose_image.save("openpose/" + name)
    return openpose_image

#use img2img with multiple filters
def img2img(name, images, prompt, seed, steps, canny_scale, op_scale):
    generator = torch.Generator("cuda").manual_seed(seed)
    out_image = pipe(
        prompt, images, num_inference_steps=steps, generator=generator, controlnet_conditioning_scale=[canny_scale, op_scale],
    ).images[0]
    out_image.save("img2img/"+ name)

#main workflow
path = 'keyframes'
name = '0.png'
img = cv2.imread(path + "/" + name)
op_image = apply_openpose(name, img)
prompt = "male portrait with cap hat, candid photo journalism, sharp bokeh uhd, gritty realistic, warm dappled lighting, in focus low depth of field, 16mm film quality with grain, pantone analog style, Rim lighting, perfect color tones, (imperfect skin quality), skin blemishes, freckles and skin pores, sharp fine details, Extremely detailed"
canny_weight = 0.44 #weight of canny controlnet
openpose_weight = 0.5 #weight of openpose controlnet

import random
rnd = random.randint(0,99999999)
print(str(rnd))

thresholds = [[100,200],[0,10],[100,300],[300,100],[0,300],[300,0],[300,500]]

for th in thresholds:
    newname = str(th[0]) + "," + str(th[1]) + ".png"
    canny_image = apply_canny_edge(name, img, th[0], th[1], newname)
    images = [canny_image, op_image]
    out = img2img(newname, images, prompt, rnd, 20, canny_weight, openpose_weight)
