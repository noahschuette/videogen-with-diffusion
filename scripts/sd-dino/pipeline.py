from util import import_video, add_leading_zeros
#from demo_swap import demo_swap
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
import os
import numpy as np
import cv2

def canny_edge(img):
    image = np.array(img)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

#import the target and source video and covert it
def import_vid(source, target, src_input_path, trg_input_path):
    import_video(source, src_input_path, True)
    import_video(target, trg_input_path, True)
    add_leading_zeros(src_input_path)
    add_leading_zeros(trg_input_path)

def single_optimization_model():
    device = "cuda"
    #controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("Uminosachi/dreamshaper_631Inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16).to(device)
    generator = torch.Generator(device=device).manual_seed(5645657) #730675724 #35622375
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return (pipe, generator, openpose)

# Optimize one of the source images with the mask of sd-dino as new reference image
def single_optimization(source_data_path , opt_index, prompt, model):
    opt_index = str(opt_index).zfill(3)
    ref_image_path = f"{source_data_path}/{opt_index}.png" # do optimization based on the i-th source image
    swapped_image_path = f"masks/{opt_index}.png" # path to the image of sd-dino
    mask_path = f"masks/{opt_index}_mask.png" # path to the mask of sd-dino for the ref_image

    if not os.path.exists("img2img"):
        os.mkdir("img2img")

    # edit one surfer for reference with sd 
    pipe, generator, openpose = model

    init_image = Image.open(swapped_image_path)
    control_image = openpose(Image.open(ref_image_path)) #Apply openpose
    #control_image = canny_edge(Image.open(ref_image_path))
    control_image.save("img2img/openpose.png")
    mask_image = Image.open(mask_path)

    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        strength=0.5,
        generator=generator).images[0]
    image.save(f"img2img/{str(opt_index).zfill(3)}.png")

def move(start, end, reference_target):
    if not os.path.exists("masks"):
        os.mkdir("masks")

    for i in range(start, end):
        i = str(i).zfill(3)
        if os.path.exists(f"results_swap/{i}_{reference_target}/mask2.png"):
            os.rename(f"results_swap/{i}_{reference_target}/mask2.png", f"masks/{i}_mask.png")
        if os.path.exists(f"results_swap/{i}_{reference_target}/swapped_image.png"):
            os.rename(f"results_swap/{i}_{reference_target}/swapped_image.png", f"masks/{i}.png")
        #if os.path.exists(f"results_swap/{i}_{reference_target}"):    
        #    os.rmdir(f"results_swap/{i}_{reference_target}")

def main():

    source_video_path = "input/bus_red.mp4" #path to the source video, which is the base of the output
    target_video_path = "input/bus_yellow.mp4" #path to the target video, which should be painted to the source
    reference_target = 0 #number of the reference target frame (between first and last frame of target video)
    extension = ".png" #extension of the source video
    start = 0 #start frame
    end = 55 #end frame

    #Importing source and data video
    source_data_path = "sd-dino/data/man_surfing"
    target_data_path = "sd-dino/data/man_skiing"
    #import_vid(source_video_path, target_video_path, source_data_path, target_data_path)
    
    #Run SD-DINO swapping
    reference_target = str(reference_target).zfill(3)
    trg_img_path = f"{target_data_path}/{reference_target}.png"
    categories = [['person'], ['person']] # categories of the element which should be swiped (similar to a text prompt)
    #demo_swap(source_data_path, extension, trg_img_path, start, end, categories)

    #Move files to other folder for later video_inpainting
    #move(start, end, reference_target)

    #Run single optimization on one image
    #prompt = "surfer, bare chest"
    prompt = "man with yellow jacket, black pants, ski helmet"
    opt_indexes = [50] #index of the image guided through optimization (between first and last frame of sd-dino output)
    model = single_optimization_model()
    for opt_index in opt_indexes:
        single_optimization(source_data_path, opt_index, prompt, model)

    #After that run video_inpainting.py while the webui api is running

if __name__ == '__main__':
    main()
