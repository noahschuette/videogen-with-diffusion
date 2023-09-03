#adapted from https://github.com/s9roll7/ebsynth_utility/blob/main/stage5.py

import os
import re
import os
import glob
import time
from sys import byteorder
import binascii
import numpy as np
import argparse

SYNTHS_PER_PROJECT = 15

def to_float_bytes(f):
    if byteorder == 'little':
        return np.array([ float(f) ], dtype='<f4').tobytes()
    else:
        return np.array([ float(f) ], dtype='>f4').tobytes()

def path2framenum(path):
    return int( os.path.splitext(os.path.basename( path ))[0] )

def search_key_dir(key_dir):
    frames = glob.glob( os.path.join(key_dir ,"[0-9]*.png"), recursive=False)

    frames = sorted(frames)

    basename = os.path.splitext(os.path.basename( frames[0] ))[0]

    key_list = [ path2framenum(key) for key in frames ]

    print("digits = " + str(len(basename)))
    print("keys = " + str(key_list))

    return len(basename), key_list

def search_video_dir(video_dir):
    frames = glob.glob( os.path.join(video_dir, "[0-9]*.png"), recursive=False)

    frames = sorted(frames)

    first = path2framenum( frames[0] )
    last = path2framenum( frames[-1] )

    return first, last

def rename_keys(key_dir):
    imgs = glob.glob(os.path.join(key_dir, "*.png"), recursive=False)

    if not imgs:
        print('no files in %s' % key_dir)
        return

    p = re.compile(r'([0-9]+).*\.png')

    for img in imgs:

        filename = os.path.basename(img)

        m = p.fullmatch(filename)

        if m:
            f = m.group(1) + ".png"
            dirname = os.path.dirname(img)
            os.rename(img, os.path.join(dirname, f))

def export_project( project, proj_filename ):

    proj_path = os.path.join( project["proj_dir"] , proj_filename + ".ebs")

    with open(proj_path, 'wb') as f:
        # header
        f.write( binascii.unhexlify('45') )
        f.write( binascii.unhexlify('42') )
        f.write( binascii.unhexlify('53') )
        f.write( binascii.unhexlify('30') )
        f.write( binascii.unhexlify('35') )
        f.write( binascii.unhexlify('00') )

        # video
        f.write( len( project["video_dir"] + project["file_name"]).to_bytes(4, byteorder) )
        f.write( (project["video_dir"] + project["file_name"]).encode() )

        # mask
        if project["mask_dir"]:
            f.write( len( project["mask_dir"] + project["file_name"]).to_bytes(4, byteorder) )
            f.write( (project["mask_dir"] + project["file_name"]).encode() )
        else:
            f.write( int(0).to_bytes(4, byteorder) )

        # key
        f.write( len( project["key_dir"] + project["file_name"]).to_bytes(4, byteorder) )
        f.write( (project["key_dir"] + project["file_name"]).encode() )

        # mask on flag
        if project["mask_dir"]:
            f.write( int(1).to_bytes(1, byteorder) )
        else:
            f.write( int(0).to_bytes(1, byteorder) )


        # keyframe weight
        f.write( to_float_bytes( project["key_weight"] ) )

        # video weight
        f.write( to_float_bytes( project["video_weight"] ) )

        # mask weight
        f.write( to_float_bytes( project["mask_weight"] ) )

        # mapping
        f.write( to_float_bytes( project["adv_mapping"] ) )

        # de-flicker
        f.write( to_float_bytes( project["adv_de-flicker"] ) )

        # diversity
        f.write( to_float_bytes( project["adv_diversity"] ) )


        # num of synths
        f.write( len( project["synth_list"] ).to_bytes(4, byteorder) )

        # synth
        for synth in project["synth_list"]:
            # key frame
            f.write( int( synth["key"] ).to_bytes(4, byteorder) )
            # is start frame exist
            f.write( int(1).to_bytes(1, byteorder) )
            # is end frame exist
            f.write( int(1).to_bytes(1, byteorder) )
            # start frame
            f.write( int( synth["prev_key"] ).to_bytes(4, byteorder) )
            # end frame
            f.write( int( synth["next_key"] ).to_bytes(4, byteorder) )

            # out path
            path =  "out-" + str(synth["key"]).zfill( project["number_of_digits"] ) + project["file_name"]
            f.write( len(path).to_bytes(4, byteorder) )
            f.write( path.encode() )

        # ?
        f.write( binascii.unhexlify('56') )
        f.write( binascii.unhexlify('30') )
        f.write( binascii.unhexlify('32') )
        f.write( binascii.unhexlify('00') )

        # synthesis detail
        f.write( int( project["adv_detail"] ).to_bytes(1, byteorder) )

        # padding
        f.write( binascii.unhexlify('00') )
        f.write( binascii.unhexlify('00') )
        f.write( binascii.unhexlify('00') )

        # use gpu
        f.write( int( project["adv_gpu"] ).to_bytes(1, byteorder) )

        # ?
        f.write( binascii.unhexlify('00') )
        f.write( binascii.unhexlify('00') )
        f.write( binascii.unhexlify('F0') )
        f.write( binascii.unhexlify('41') )
        f.write( binascii.unhexlify('C0') )
        f.write( binascii.unhexlify('02') )
        f.write( binascii.unhexlify('00') )
        f.write( binascii.unhexlify('00') )

def main(args):

    frame_path = args.frames_path
    frame_mask_path = args.mask_path
    img2img_upscale_key_path = args.keyframe_path

    if not (args.skip_renaming):
        rename_paths = [frame_path, frame_mask_path, img2img_upscale_key_path]
        for path in rename_paths:
            for filename in os.listdir(path):
                num = filename[:-4]
                num = str(int(num))
                num = num.zfill(5)
                new_filename = num + ".png"
                os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

    no_upscale = False

    rename_keys(img2img_upscale_key_path)

    number_of_digits, keys = search_key_dir( img2img_upscale_key_path )

    if number_of_digits == -1:
        print("No Key frame")
        return

    first_frame, last_frame = search_video_dir( frame_path )

    ### add next key
    synth_list = []
    next_key = last_frame

    for key in keys[::-1]:
        synth_list.append( { "next_key": next_key })
        next_key = key

    synth_list = synth_list[::-1]
    prev_key = first_frame

    ### add key / prev key
    for i, key in enumerate(keys):
        synth_list[i]["key"] = key
        synth_list[i]["prev_key"] = prev_key
        prev_key = key

    project = {
        "proj_dir" : "",
        "file_name" : "/[" + "#" *  number_of_digits + "].png",
        "number_of_digits" : number_of_digits,

        "key_dir" : img2img_upscale_key_path,
        "video_dir" : frame_path,
        "mask_dir" : frame_mask_path,
        "key_weight" : 1.0,
        "video_weight" : 4.0,
        "mask_weight" : 1.0,
        "adv_mapping" : 10.0,
        "adv_de-flicker" : 1.0,
        "adv_diversity" : 3500.0,
        "adv_detail" : 1,   # high
        "adv_gpu" : 1,      # use gpu
    }

    proj_base_name = time.strftime("%Y%m%d-%H%M%S")

    tmp=[]
    proj_index = 0
    for i, synth in enumerate(synth_list):
        tmp.append(synth)
        if (i % SYNTHS_PER_PROJECT == SYNTHS_PER_PROJECT-1):
            project["synth_list"] = tmp
            proj_file_name = proj_base_name + "_" + str(proj_index).zfill(5)
            export_project(project,proj_file_name)
            proj_index += 1
            tmp = []
            print("exported : " + proj_file_name + ".ebs" )
    if tmp:
        project["synth_list"] = tmp
        proj_file_name = proj_base_name + "_" + str(proj_index).zfill(5)
        export_project( project, proj_file_name )
        proj_index += 1
        print("exported : " + proj_file_name + ".ebs" )

    print("Exported EB-Synth project file")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert frames, masks and keyframes to EB-Synth project file")
    parser.add_argument('frames_path', help='Path to the video frames folder')
    parser.add_argument('mask_path', help='Path to the video masks folder')
    parser.add_argument('keyframe_path', help='Path to the keyframe folder')
    parser.add_argument('--skip_renaming', action='store_true', help='Skip adding trailing zeros to frames')
    args = parser.parse_args()
    main(args)
