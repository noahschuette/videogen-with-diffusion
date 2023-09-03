import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import argparse

#Load DinoV2 from Facebook Research
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

#transform image
transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

#Get DINOv2 features from forward step
#src: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py line 221 ff.
def get_token(img):
    with torch.no_grad():
        features = dinov2_vits14.forward_features(img.cuda())["x_norm_patchtokens"]
    return features

# visualize first PCA components, adapted from https://github.com/facebookresearch/dinov2/issues/6
def visualize_pca(features, frames):
    pca = PCA(n_components=3)
    features = features.cpu()[0]
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min()) 
    pca_features = pca_features * 255 # 0-1 to 255 (RGB)
    print("PCA_shape",pca_features.shape)
    return pca_features.reshape(16 * frames, 16, 3).astype(np.uint8)

def main(args):

    input_dir = args.frames_path #input directory of video frames, default "input"
    output_dir = args.output_path #output for dino frames, default "dino_testing"
    start_frame = args.start_frame #number of the starting frame
    end_frame = args.end_frame #number of the end frame (excluded)
    frames = end_frame - start_frame #number of frames to process

    # load FRAMES images from input folder, forward step for each sample, 
    features = None
    for i in range(start_frame, end_frame):
        img = Image.open(f'{input_dir}/{i}.png') #load image
        img = transform(img)[:3].unsqueeze(0) #transform image    
        feature = get_token(img) #get forward features for image
        if (features == None):
            features = feature
        else:
            features = torch.cat((features, feature), 1) #concat features to one array
    print("final",features.shape)
    arr = visualize_pca(features, frames) #PCA for every frame at once
    arr = np.split(arr, (frames)) #split array to extract each frame
    for i in range(frames):
        im = Image.fromarray(arr[i]) #convert Array to Image
        im = im.resize((512, 512), resample=Image.Resampling.NEAREST) #upscale image to HD for training
        #im = im.resize((512, 512))
        #im.save(f'{OUTPUT_DIR}/{str(i).zfill(3)}.png') #save image
        im.save(f'{output_dir}/{i}.png') #save image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract DinoV2 PCA features for each frame in a folder")
    parser.add_argument('frames_path', help='Folder with frames from input video')
    parser.add_argument('output_path', help='Output folder for the PCA-features')
    parser.add_argument('start_frame', type=int, help='Start frame')
    parser.add_argument('end_frame', type=int, help='End frame')
    args = parser.parse_args()
    main(args)
