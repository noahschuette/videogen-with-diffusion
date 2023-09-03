# Utility Class
import os

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
        #print('Read a new frame: ', success)
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
def add_leading_zeros(dir, leading_zeros=5):
    import glob
    import os
    import re

    for filename in glob.glob(dir + '/*.png'):
        new_filename = re.sub('(\d+)', lambda x: x.group(1).zfill(leading_zeros), filename)
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

#resize masks
def resize_masks(dir):
    import cv2
    print("Resizing masks")

    files = os.listdir(dir)
    c = 0
    for file in files:
        img = cv2.imread(dir + "/" + file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        c += 1
        cv2.imwrite(dir + "/" + file, img)

#crop images
def crop_images(dir):
    import cv2
    for i in range(16):
        path = f"{dir}/{str(i).zfill(5)}.png"
        img = cv2.imread(path)
        #img = img[112:400, 0:512]
        img = cv2.resize(img, (512,512))
        cv2.imwrite(path, img)