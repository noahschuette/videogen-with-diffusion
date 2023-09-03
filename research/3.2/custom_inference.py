from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import argparse
import torch
import random

# Set device and check if GPU is available
device="cuda"
print(torch.cuda.is_available())

#Prepare ControlNet Pipeline with model from path
MODEL_NAME = 'dino' #name of the model (path)
controlnet = [ ControlNetModel.from_pretrained(f"models/{MODEL_NAME}", torch_dtype=torch.float16), ] 
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe = pipe.to(device)

#Enable model offloading & xformers for faster developement
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

#Run inference with conditioning image & prompt, save as .png under name
def img2img(out_path, name, images, prompt, seed, steps=20, conditioning=0.5):
    generator = torch.Generator("cuda")
    generator = generator.manual_seed(seed)
    out_image = pipe(
        prompt, images, num_inference_steps=steps, generator=generator, controlnet_conditioning_scale=[conditioning]
    ).images[0]
    out_image.save(out_path + "/" + name)

def main(args):
#main workflow
    path = args.frames_path
    output_folder = args.output_path
    frames = args.frames
    prompt = args.prompt
    conditioning = 0.5
    rnd = random.randint(0,99999999)

    if args.conditioning_scale is not None:
        conditioning = args.conditioning_scale
    if args.seed is not None:
        rnd = args.seed

    for i in range(frames):
        img = Image.open(f"{path}/{i}.png") #important to use PIL instead of cv2
        img2img(output_folder, str(i).zfill(3) + ".png", [img], prompt, rnd, 20, conditioning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform img2img with custom Control-Net")
    parser.add_argument('frames_path', help='Path to the video frames folder')
    parser.add_argument('output_path', help='Path to the output folder')
    parser.add_argument('frames', type=int, help='Amount of frames to process')
    parser.add_argument('prompt', help='Text prompt for img2img')
    parser.add_argument('--conditioning_scale', type=float, help='Control-Net conditioning scale')
    parser.add_argument('--seed', type=int, help='Generator seed')
    args = parser.parse_args()
    main(args)


 