from util import import_video, add_leading_zeros
from demo_swap import demo_swap
import os
import argparse

#import the target and source video and covert it
def import_vid(source, target, src_input_path, trg_input_path):
    import_video(source, src_input_path, True)
    import_video(target, trg_input_path, True)
    add_leading_zeros(src_input_path)
    add_leading_zeros(trg_input_path)

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

def main(args):

    source_video_path = "input/bus_red.mp4" #path to the source video, which is the base of the output
    target_video_path = "input/bus_yellow.mp4" #path to the target video, which should be painted to the source
    reference_target = 0 #number of the reference target frame (between first and last frame of target video)
    extension = ".png" #extension of the source video
    start = 0 #start frame
    end = 55 #end frame

    #Step 1: Importing source and data video
    source_data_path = "sd-dino/data/man_surfing"
    target_data_path = "sd-dino/data/man_skiing"
    if not (args.skip_import):
        import_vid(source_video_path, target_video_path, source_data_path, target_data_path)
    
    #Step 2: Run SD-DINO: Swap, each source frame (from start to end index) with the target from reference_target index
    reference_target = str(reference_target).zfill(3)
    trg_img_path = f"{target_data_path}/{reference_target}.png"
    categories = [['person'], ['person']] # categories of the element which should be swiped (similar to a text prompt)
    demo_swap(source_data_path, extension, trg_img_path, start, end, categories)

    #Step 3: Move files to other folder for later video_inpainting
    move(start, end, reference_target)

    #After that run video_inpainting.py while the webui api is running

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SD-Dino Pipeline")
    parser.add_argument('--skip_import', action='store_true', help='Skip video importing')
    args = parser.parse_args()
    main(args)
