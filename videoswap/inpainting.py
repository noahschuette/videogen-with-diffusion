import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
import os

# Single Inpainting
class SingleInpainting:
    def __init__(self, source_path, seed, openpose_strenght=0.5):
        self.openpose_strenght = openpose_strenght
        self.source_path = source_path
        device = "cuda"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "Uminosachi/dreamshaper_631Inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        if not os.path.exists("img2img"):
            os.mkdir("img2img")


    # Optimize one of the source images with the mask of sd-dino as new reference image
    def optimize_frame(self, frame, prompt):
        frame = str(frame).zfill(3)
        ref_image_path = f"{self.source_path}/{frame}.png" # do optimization based on the i-th source image
        swapped_image_path = f"masks/{frame}.png" # path to the image of sd-dino
        mask_path = f"masks/{frame}_mask.png" # path to the mask of sd-dino for the ref_image

        init_image = Image.open(swapped_image_path)
        control_image = self.openpose(Image.open(ref_image_path)) #Apply openpose
        control_image.save("img2img/openpose.png")
        mask_image = Image.open(mask_path)

        image = self.pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            strength=self.openpose_strenght,
            generator=self.generator).images[0]
        image.save(f"img2img/{str(frame).zfill(3)}.png")
