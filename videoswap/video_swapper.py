import os
from util import import_video, add_leading_zeros, remove_leading_zeros
import demo_swap

# VideoSwapper: Swap Source and Target Video with SD-Dino
class VideoSwapper:
    def __init__(self):
        self.source_path = None
        self.target_path = None
        self.reference_target = None
        self.mkdirs()
    
    def mkdirs(self):
        if not os.path.exists("masks"):
            os.mkdir("masks")
        if not os.path.exists("input"):
            os.mkdir("input")

    # Set path to source video manually (alternatively to import from video)
    def set_source_path(self, source_path):
        self.source_path = source_path

    # Define reference Target
    def set_reference_target(self, frame, target_path=None, extension=".png"):
        if target_path is not None:
            self.target_path = target_path
        frame = str(frame).zfill(3)
        if not os.path.exists(f"{target_path}/{frame}.{extension}"):
            raise Exception("Frame not avaiable in target path")
        self.reference_target = f"{target_path}/{frame}.{extension}"

    # Import source and target video
    def import_source_video(self, video_path, source_path):
        self.source_path = source_path
        if not os.path.exists(source_path):
            os.mkdir(source_path)
        import_video(video_path, True)
        add_leading_zeros(source_path)

    def import_target_video(self, video_path, target_path):
        self.target_path = target_path
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        import_video(target_path, video_path, True)
        add_leading_zeros(target_path)

    # Swap source and target video with SD-Dino
    def swap(self, start_frame, end_frame, categories, extension=".png"):
        if self.source_path is None or self.reference_target is None:
            raise Exception("Source or Reference Target not set")
        if not os.path.exists(f"{self.source_path}/{str(start_frame).zfill(3)}{extension}"):
            raise Exception("Start frame not in source path")
        if not os.path.exists(f"{self.source_path}/{str(end_frame).zfill(3)}{extension}"):
            raise Exception("End frame not in source path")
        
        print("--- Starting SD-Dino Swap ---")
        demo_swap(self.source_path, extension, self.reference_target, start_frame, end_frame, categories)

        self.move(start_frame, end_frame)

    # Move files in directories
    def move(self, start, end):
        for i in range(start, end):
            i = str(i).zfill(3)
            if os.path.exists(f"results_swap/{i}_{self.reference_target}/mask2.png"):
                os.rename(f"results_swap/{i}_{self.reference_target}/mask2.png", f"masks/{i}.png")
            if os.path.exists(f"results_swap/{i}_{self.reference_target}/swapped_image.png"):
                os.rename(f"results_swap/{i}_{self.reference_target}/swapped_image.png", f"input/{i}.png")