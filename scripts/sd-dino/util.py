import os

def rename():
    for i in range(0, 50):
        j = 200 + i
        os.rename(f"sd-dino/data/eagle/{str(j).zfill(3)}.png", f"sd-dino/data/eagle/{str(i).zfill(3)}.png")

def crop(start, end, output_dir):
    from PIL import Image
    for i in range(start,end):
        i = str(i).zfill(3)
        if os.path.exists(f"{output_dir}/{i}.png"):
            img = Image.open(f"{output_dir}/{i}.png")
            img = img.resize((718,718))
            img.save(f"{output_dir}/{i}.png")

#import video from video to directory
def import_video(video_path, output_dir = "input", clear_folder = True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory " + output_dir)
    elif clear_folder:
        import glob
        for fl in glob.glob(f"/{output_dir}/*.png"):
            os.remove(fl)
        print("Cleaned folder")

    import cv2
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_dir,"%d.png" % count), image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    print("Imported video to " + output_dir + " directory")

def remove_leading_zeros(dir):
    import glob
    import os
    import re

    for filename in glob.glob(dir + '/*.png'):
        if filename == dir + "/0.png":
            continue
        new_filename = re.sub('(\d+)', lambda x: x.group(1).lstrip("0"), filename)
        os.rename(filename, new_filename)
    print("Removed leading zeros from " + dir + " directory")

#add leading zeros to each file from directory
def add_leading_zeros(dir):
    import glob
    import os
    import re

    for filename in glob.glob(dir + '/*.png'):
        new_filename = re.sub('(\d+)', lambda x: x.group(1).zfill(3), filename)
        os.rename(filename, new_filename)
    print("Added leading zeros to " + dir + " directory")

#create mp4 movie from frames to directory dir for frames
def export_video(dir, frames, start_frame=0, video_name = "video", fps = 30, downscale = False):
    import subprocess

    path = os.path.join(dir , '%03d.png')
    out = os.path.join(dir, video_name + ".mp4")

    # ffmpeg -r 10 -start_number n -i snapshot_%03d.png -vframes 50 example.gif
    subprocess.call("ffmpeg -framerate " + str(fps) + " -r " + str(fps) +
                        " -start_number " + str(start_frame) +
                        " -i " + path +
                        " -vframes " + str(frames) +
                        " -vcodec openh264 -pix_fmt yuv420p " +
                        out, shell=True)
    print("Created movie " + video_name + ".mp4" + " in " + dir + " directory")
    if (downscale):
        downscale_video(os.path.join(dir, video_name))

def downscale_video(video_name):
    import subprocess
    subprocess.call("ffmpeg -i " + video_name + ".mp4 -vcodec libx265 -crf 28 " + video_name + "_downscaled.mp4", shell=True)
    print("Downscaled video " + video_name + ".mp4" + " to " + video_name + "_downscaled.mp4")