import os
import subprocess

def export_video(dir, frames, start_frame=0, video_name = "video", fps = 30, downscale = False):
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
    subprocess.call("ffmpeg -i " + video_name + ".mp4 -vcodec libx265 -crf 28 " + video_name + "_downscaled.mp4", shell=True)
    print("Downscaled video " + video_name + ".mp4" + " to " + video_name + "_downscaled.mp4")