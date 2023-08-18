from PIL import Image
import os

#Utility functions for several workflows

#create collage from images in directory
def create_collage(dir, name, count, size):
    #DIR = "output"
    #NAME = "collage"
    #COUNT = 1000 #number of images
    #SIZE = 512 #size of images

    # open images to array from directory
    def open_images_to_array():
        images = []
        for i in range(count):
            img = Image.open(dir + '/' + str(i) + '.png')
            images.append(img)
        return images

    images = open_images_to_array()

    # create new image
    new_im = Image.new('RGB', (size * 10, size * 10))

    # paste images to new image
    for i in range(len(images)):
        new_im.paste(images[i], (size * (i % 10), size * (i // 10)))

    # save image
    new_im.save(name + ".png")
    print("Created collage " + name + ".png")

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


#crop dino images to 512x512, dir = image directory, count = number of images
def crop_target(dir, count):
    #DIR = "target"
    #COUNT = 1000

    # open images to array from directory
    def open_images_to_array():
        images = []
        for i in range(count):
            img = Image.open(dir + '/' + str(i) + '.png')
            images.append(img)
        return images

    images = open_images_to_array()
    for i in range(len(images)):
        width, height = images[i].size

        # crop image to width in center:
        top = (height - width) / 2
        bot = (height + width) / 2
        image = images[i].crop((0, top, 0, bot))

        image = image.resize((512, 512))
        image.save(dir + '/' + str(i) + '.png')
    print("Cropped images in " + dir + " directory")

#OUTPUT_DIR = "dinotarget"
#VIDEO_NAME = "demo_video"
#FPS = 30
#FRAMES = 1000

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
                        " -vcodec libx264 -pix_fmt yuv420p " +
                        out, shell=True)
    print("Created movie " + video_name + ".mp4" + " in " + dir + " directory")
    if (downscale):
        downscale_video(os.path.join(dir, video_name))

def downscale_video(video_name):
    import subprocess
    subprocess.call("ffmpeg -i " + video_name + ".mp4 -vcodec libx265 -crf 28 " + video_name + "_downscaled.mp4", shell=True)
    print("Downscaled video " + video_name + ".mp4" + " to " + video_name + "_downscaled.mp4")

#dir1 = "dino"
#dir2 = "target"
#concat each image from two directories dir1, dir2 to compare them:
def concat_images(dir1, dir2 ,output_dir, frames):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory " + output_dir)

    for i in range(frames):
        images = []
        images.append(Image.open(dir1+'/'+str(i)+'.png'))
        images.append(Image.open(dir2+'/'+str(i)+'.png'))
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save(f'{output_dir}/{str(i).zfill(3)}.png') #save image
    print("Concatenated images to " + output_dir + " directory")

#prompt = "male face with white background"
def gen_dataset_json(frames, prompt = "", file_name = "prompt.json"):
    list = []
    for i in range(frames):
        list.append({"source": "source/" + str(i) + ".png", "target": "target/" + str(i) + ".png", "prompt": prompt})

    with open(file_name, 'w') as filehandle:
        for listitem in list:
            filehandle.write('%s\n' % listitem)
    print("Generated dataset-json file " + file_name)

#remove all png files from folder
def clear_folder(dir):
    import glob
    for fl in glob.glob(f"/{dir}/*.png"):
        os.remove(fl)
    print("Cleaned folder " + dir)