import os
from util import import_video, add_leading_zeros, remove_leading_zeros
from demo_swap import demo_swap

ZFILL_NUM = 5

# VideoSwapper: Swap Source and Target Video with SD-Dino
class VideoSwapper:
    def __init__(self, dir):
        self.source_path = None
        self.target_path = None
        self.reference_target = None
        self.dir = dir
        self.mkdirs(dir)
    
    def mkdirs(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(f"{dir}/masks"):
            os.mkdir(f"{dir}/masks")
        if not os.path.exists(f"{dir}/video"):
            os.mkdir(f"{dir}/video")

    # Set path to source video manually (alternatively to import from video)
    def set_source_path(self, source_path):
        self.source_path = source_path
        print("Source Path set")

    # Define reference Target
    def set_reference_target(self, frame, target_path=None, extension="png"):
        if target_path is not None:
            self.target_path = target_path
        frame = str(frame).zfill(ZFILL_NUM)
        if not os.path.exists(f"{target_path}/{frame}.{extension}"):
            raise Exception(f"Frame {target_path}/{frame}.{extension} not avaiable in target path")
        self.reference_target = f"{target_path}/{frame}.{extension}"
        print("Reference Target set")

    def set_reference_from_image(self, reference_path):
        if not os.path.exists(f"{reference_path}"):
            raise Exception("Image not avaiable in target path")
        self.reference_target = reference_path
        print("Reference Target set")

    # Import source and target video
    def import_source_video(self, video_path, source_path):
        self.source_path = source_path
        if not os.path.exists(source_path):
            os.mkdir(source_path)
        import_video(video_path, output_dir=source_path, clear_folder=True)
        add_leading_zeros(source_path)
        print("Source Video imported")

    def import_target_video(self, video_path, target_path):
        self.target_path = target_path
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        import_video(video_path, output_dir=target_path, clear_folder=True)
        add_leading_zeros(target_path)

    # Swap source and target video with SD-Dino
    def swap(self, start_frame, end_frame, categories, extension=".png"):
        if self.source_path is None or self.reference_target is None:
            raise Exception("Source or Reference Target not set")
        if not os.path.exists(f"{self.source_path}/{str(start_frame).zfill(ZFILL_NUM)}{extension}"):
             raise Exception("Start frame not in source path")
        if not os.path.exists(f"{self.source_path}/{str(end_frame).zfill(ZFILL_NUM)}{extension}"):
            raise Exception("End frame not in source path")
        
        print("--- Starting SD-Dino Swap ---")
        demo_swap(self.source_path, extension, self.reference_target, start_frame, end_frame, categories)

        self.move(start_frame, end_frame)

    # Move files in directories
    def move(self, start, end):
        ref_path = os.path.basename(os.path.splitext(self.reference_target)[0]).split('/')[-1]
        print(ref_path)
        for i in range(start, end):
            i = str(i).zfill(ZFILL_NUM)
            if os.path.exists(f"results_swap/{i}_{ref_path}/mask2.png"):
                print("Move" , i)
                os.rename(f"results_swap/{i}_{ref_path}/mask2.png", f"{self.dir}/masks/{i}.png")
            if os.path.exists(f"results_swap/{i}_{ref_path}/swapped_image.png"):
                print("Move" , i)
                os.rename(f"results_swap/{i}_{ref_path}/swapped_image.png", f"{self.dir}/video/{i}.png")