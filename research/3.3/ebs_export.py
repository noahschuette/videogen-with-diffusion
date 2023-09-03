#adapted from https://github.com/s9roll7/ebsynth_utility/blob/main/stage7.py (last visited 18.10.2023)
import os
import re
import glob
import shutil
import time
import cv2
import subprocess
import numpy as np
import argparse


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]

def create_movie_from_frames( dir, start, end, number_of_digits, fps, output_path, export_type):
    def get_export_str(export_type):
        if export_type == "mp4":
            return " -vcodec libx264 -pix_fmt yuv420p "
        elif export_type == "webm":
#            return " -vcodec vp9 -crf 10 -b:v 0 "
            return " -crf 40 -b:v 0 -threads 4 "
        elif export_type == "gif":
            return " "
        elif export_type == "rawvideo":
            return " -vcodec rawvideo -pix_fmt bgr24 "

    vframes = end - start + 1
    path = os.path.join(dir , '%0' + str(number_of_digits) + 'd.png')

    # ffmpeg -r 10 -start_number n -i snapshot_%03d.png -vframes 50 example.gif
    subprocess.call("ffmpeg -framerate " + str(fps) + " -r " + str(fps) +
                        " -start_number " + str(start) +
                        " -i " + path +
                        " -vframes " + str( vframes ) +
                        get_export_str(export_type) +
                        output_path, shell=True)

def search_out_dirs(proj_dir, blend_rate):
    ### create out_dirs
    p = re.compile(r'.*[\\\/]out\-([0-9]+)[\\\/]')

    number_of_digits = -1

    out_dirs=[]
    for d in glob.glob( os.path.join(proj_dir ,"out-*/"), recursive=False):
        m = p.fullmatch(d)
        if m:
            if number_of_digits == -1:
                number_of_digits = len(m.group(1))
            out_dirs.append({ 'keyframe':int(m.group(1)), 'path':d })

    out_dirs = sorted(out_dirs, key=lambda x: x['keyframe'], reverse=True)

    print(number_of_digits)

    prev_key = -1
    for out_d in out_dirs:
        out_d['next_keyframe'] = prev_key
        prev_key = out_d['keyframe']

    out_dirs = sorted(out_dirs, key=lambda x: x['keyframe'])


    ### search start/end frame
    prev_key = 0
    for out_d in out_dirs:
        imgs = sorted(glob.glob(  os.path.join( out_d['path'], '[0-9]'*number_of_digits + '.png') ))

        first_img = imgs[0]
        print(first_img)
        basename_without_ext = os.path.splitext(os.path.basename(first_img))[0]
        blend_timing = (prev_key - out_d['keyframe'])*blend_rate + out_d['keyframe']
        blend_timing = round(blend_timing)
        start_frame = max( blend_timing, int(basename_without_ext) )
        out_d['startframe'] = start_frame

        last_img = imgs[-1]
        print(last_img)
        basename_without_ext = os.path.splitext(os.path.basename(last_img))[0]
        end_frame = min( out_d['next_keyframe'], int(basename_without_ext) )
        if end_frame == -1:
            end_frame = int(basename_without_ext)
        out_d['endframe'] = end_frame
        prev_key = out_d['keyframe']

    return number_of_digits, out_dirs

def get_ext(export_type):
    if export_type in ("mp4","webm","gif"):
        return "." + export_type
    else:
        return ".avi"

def main(args):

    # Pass Arguments

    project_dir = args.project_dir
    fps = 30
    blend_rate = 1
    export_type = "mp4"
    if args.fps is not None:
        fps = args.fps
    if args.blend_rate is not None:
        blend_rate = args.blend_rate
    if args.ext is not None:
        export_type = args.ext

    # Blend results together

    blend_rate = clamp(blend_rate, 0.0, 1.0)

    tmp_dir = os.path.join(project_dir,"crossfade_tmp")

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    number_of_digits, out_dirs = search_out_dirs(project_dir,blend_rate)
    print(project_dir)

    start = out_dirs[0]['startframe']
    end = out_dirs[-1]['endframe']

    cur_clip = 0
    next_clip = cur_clip+1 if len(out_dirs) > cur_clip+1 else -1

    current_frame = 0

    print(str(start) + " -> " + str(end+1))

    black_img = np.zeros_like( cv2.imread( os.path.join(out_dirs[cur_clip]['path'], str(start).zfill(number_of_digits) + ".png") ) )

    for i in range(start, end+1):
        print(str(i) + " / " + str(end))
        if next_clip == -1:
            break
        if i in range( out_dirs[cur_clip]['startframe'], out_dirs[cur_clip]['endframe'] +1):
            pass
        elif i in range( out_dirs[next_clip]['startframe'], out_dirs[next_clip]['endframe'] +1):
            cur_clip = next_clip
            next_clip = cur_clip+1 if len(out_dirs) > cur_clip+1 else -1
            if next_clip == -1:
                break
        else:
            ### black
            # front ... none
            # back ... none
            cv2.imwrite( os.path.join(tmp_dir, filename) , black_img)
            current_frame = i
            continue
        filename = str(i).zfill(number_of_digits) + ".png"

        # front ... cur_clip
        # back ... next_clip or none

        if i in range( out_dirs[next_clip]['startframe'], out_dirs[next_clip]['endframe'] +1):
            # front ... cur_clip
            # back ... next_clip
            img_f = cv2.imread( os.path.join(out_dirs[cur_clip]['path'] , filename) )
            img_b = cv2.imread( os.path.join(out_dirs[next_clip]['path'] , filename) )

            back_rate = (i - out_dirs[next_clip]['startframe'])/ max( 1 , (out_dirs[cur_clip]['endframe'] - out_dirs[next_clip]['startframe']) )

            img = cv2.addWeighted(img_f, 1.0 - back_rate, img_b, back_rate, 0)

            cv2.imwrite( os.path.join(tmp_dir , filename) , img)
        else:
            # front ... cur_clip
            # back ... none
            filename = str(i).zfill(number_of_digits) + ".png"
            shutil.copy( os.path.join(out_dirs[cur_clip]['path'] , filename) , os.path.join(tmp_dir , filename) )

        current_frame = i


    start2 = current_frame+1

    print(str(start2) + " -> " + str(end+1))

    for i in range(start2, end+1):
        filename = str(i).zfill(number_of_digits) + ".png"
        shutil.copy( os.path.join(out_dirs[cur_clip]['path'] , filename) , os.path.join(tmp_dir , filename) )

    ### Create Movie
    movie_base_name = time.strftime("%Y%m%d-%H%M%S")

    nosnd_path = os.path.join(project_dir , movie_base_name + get_ext(export_type))

    start = out_dirs[0]['startframe']
    end = out_dirs[-1]['endframe']

    create_movie_from_frames( tmp_dir, start, end, number_of_digits, fps, nosnd_path, export_type)

    print("Exported : " + nosnd_path)
    print("Stage 7: Video render done")

if __name__ == '__main__':
    #project_dir, original_movie_path, blend_rate=1, export_type="mp4"
    parser = argparse.ArgumentParser(description="Blend results together and export to video")
    parser.add_argument('project_dir', help='Path to the project directory')
    parser.add_argument('--fps', type=int, help='Frames per second for the result')
    parser.add_argument('--blend_rate', type=int, help='Blend rate')
    parser.add_argument('--ext', action='store_true', help='Format export type defaults .mp4')
    args = parser.parse_args()
    main(args)
