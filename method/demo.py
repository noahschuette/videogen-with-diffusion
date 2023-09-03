# Demonstration of SD-Dino on videos

from video_swapper import VideoSwapper
from inpainting import SingleInpainting
from to_ebs import main as export_to_ebsynth

def main():

    videoswapper = VideoSwapper()

    # Import the video which serves as the base
    videoswapper.import_source_video(video_path="../videos/input_videos/skateboard-man.mp4", source_path="data/man_skateboarding")

    # Import the reference video and set the reference target
    videoswapper.import_target_video(video_path="../videos/input_videos/man_skiing.mp4", target_path="data/man_skiing")
    videoswapper.set_reference_target(frame=0)

    # Execute Swap between frame 0 and 20, categories define the elements which should be swapped
    videoswapper.swap(start_frame=0, end_frame=20, categories=[['person'], ['person']])

    # Inpaint one of the frames as our keyframe
    inpainting = SingleInpainting(source_path="data/man_skateboarding", seed=5645657, openpose_strenght=0.5)
    inpainting.optimize_frame(frame=0, prompt="man with yellow jacket, black pants, ski helmet")

    # Optional: Export to EB-Synth to generate .ebs project file, or use video_inpainting.py
    export_to_ebsynth()
    print("Now run EBSynth.exe with the exported .ebs file")
    print('After finishing synthesis, run: " python export.py "out" " to generate video from results')
