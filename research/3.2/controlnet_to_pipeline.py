# Helper script to convert a raw controlnet model to a diffusers-friendly checkpoint.
# Adapted from https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt
import argparse
import torch
device="cuda"
print(torch.cuda.is_available())

version = 9800
epoch = 41
step = 10499
model_name = "dino_interpolated"

def main(args):

    version = args.version
    epoch = args.epoch
    step = args.step
    checkpoint_path = f"./ControlNet/lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt" #ckpt file to convert to
    original_config_file = "./ControlNet/models/cldm_v15.yaml" #The YAML config file corresponding to the original architecture
    print("Checkpoint path: ", checkpoint_path)
    print("Original model config: ", original_config_file)

    model_name = args.model_name
    dump_path = f"./models/{model_name}" #path to the output model

    controlnet = download_controlnet_from_original_ckpt(
            checkpoint_path=checkpoint_path,
            original_config_file=original_config_file,
            image_size=512,
            num_in_channels=None, #None = Auto-Detect num of input channels
            from_safetensors=False,
            device=device,
        )
    print("Loaded ControlNet Config")

    controlnet.save_pretrained(dump_path, safe_serialization=False)
    print("Saved pretrained to ./models/" + model_name)

if __name__ == '__main__':
    #project_dir, original_movie_path, blend_rate=1, export_type="mp4"
    parser = argparse.ArgumentParser(description="Export raw controlnet model to diffusers-friendly checkpoint.")
    parser.add_argument('model_name', help='Give the model a fancy name')
    parser.add_argument('version', type=int, help='Model version (from lightning_logs folder, required for filename)')
    parser.add_argument('epoch', type=int, help='Epoch of the last lightning_log checkpoint, required for filename)')
    parser.add_argument('step', type=int, help='Step of the last lightning_log checkpoint, required for filename)')
    args = parser.parse_args()
    main(args)

# Use model afterwards like this in inference script:
# controlnet = [ ControlNetModel.from_pretrained(f"models/{MODEL_NAME}", torch_dtype=torch.float16), ]