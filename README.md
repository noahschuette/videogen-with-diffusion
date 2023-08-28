# Video Generation with Diffusion Models
Repository for my Bachelor's Thesis

## Video Sources:

For research we used clips of the following videos:  
balloon.mp4 - https://www.youtube.com/watch?v=mW3mxzCNwCo  
airballoon.mp4 - https://www.youtube.com/watch?v=t-C8WcMqXb4  
bus_yellow.mp4 - https://www.youtube.com/watch?v=R1JuCkqx3d8&t=74s  
bus_red.mp4 - https://www.youtube.com/watch?v=CFlBtMD2Qis&t=143s  
eagle.mp4 - https://www.youtube.com/watch?v=i94QoqvmgrM  
plane.mp4 - https://www.youtube.com/watch?v=xk4LFFm1zAA  
skateboard-man.mp4 - from TAV ?  
snowboarder.mp4 - DAVIS dataset ?  
man_surfing.mp4 - from TAV ?  

## 	Installation Notes:

###	EB-Synth

Install EB-Synth as described in https://ebsynth.com/

The application can now be dragged into the same folder to use the code from `scripts/EB-Synth` 

###	SD-Dino
Install Conda Environment and SD-Dino Repository as described in https://github.com/Junyi42/sd-dino:
```
conda create -n sd-dino python=3.9
conda activate sd-dino
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
git clone git@github.com:Junyi42/sd-dino.git 
cd sd-dino
pip install -e .
pip install diffusers
pip install controlnet_aux
```

If torch does not work properly with CUDA, try installing it differently
```
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
``` 
Next, move all python files from `scripts/sd-dino` to the parent folder of the cloned repository, so that `extractor_dino.py` overwrites the existing file in the cloned subfolder. 

###	Stable-Diffusion Web-UI

To use `video_inpainting.py` it is required to install Control-Net for the Web-UI version of Stable Diffusion. 
Install [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [Control-Net for WebUI](https://github.com/Mikubill/sd-webui-controlnet)

Also ensure that the `url` variable in `video_inpainting.py` matches the url in `webui.py:362`. The url should be a valid ip-adress shown in `hostname -I` . 

To start the environment without the UI, use 
`python launch.py --nowebui --cors-allow-origins=http://192.168.0.1:7861` 
where you also replace the IP-adress. 
The API is now ready to listen to calls from `video_inpainting.py` 

## Running Application:

Run `demo.py` in `videoswap` to demonstrate SD-Dino with EB-Synth.
