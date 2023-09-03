# Inference script to use Stable Diffusion with ControlNet
# Adapted from https://huggingface.co/lllyasviel/sd-controlnet-openpose and https://huggingface.co/lllyasviel/sd-controlnet-canny

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import cv2
from PIL import Image
import numpy as np
import torch
import random
import argparse
from util import *

# Set device and check if GPU is available
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

#Optional:  enable model offloading, enable XFormers for faster developement
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

#use canny edge filter on image
def apply_canny_edge(img):
    image = np.array(img)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    #canny_image.save("canny/"+ savename)
    return canny_image

#use openpose filter on image
def apply_openpose(img):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(img)
    #openpose_image.save("openpose/" + name)
    return openpose_image

#apply image to image with the selected controls
def img2img(images, prompt, seed, inference_steps=20, canny_scale=0.5, openpose_scale=0.5):
    generator = torch.Generator("cuda").manual_seed(seed)
    out_image = pipe(
        prompt, images, num_inference_steps=inference_steps, generator=generator, controlnet_conditioning_scale=[canny_scale, openpose_scale],
    ).images[0]
    return out_image


def main(args):
    seed = random.randint(0,99999999) #set random seed

    video_path = args.video_path # path to input video
    frame_path = "./input" # temporary input path for frames
    out_path = './out' # temporary output path for frames
    prompt = args.prompt # text prompt for img2img

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    import_video(video_path, frame_path) # import video to input directory

    frames = 0
    for filename in os.listdir(frame_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frames += 1
            path = os.path.join(out_path, filename)
            img = cv2.imread(path)
            openpose_image = apply_openpose(img) #add openpose control
            canny_image = apply_canny_edge(img) #add canny edge control
            output = img2img([canny_image, openpose_image], prompt, seed) #run diffusion for frame
            output.save(out_path)

    export_video(out_path, frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform img2img with Control-Net models")
    parser.add_argument('video_path', help='Path to the video frames folder') #'../input_videos/human_1.mp4' 
    parser.add_argument('prompt', help='Text prompt for img2img') #'Tony Stark with beard in Iron Man suit'
    args = parser.parse_args()
    main(args)