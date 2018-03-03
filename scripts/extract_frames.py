"""
Extract frames by passing a csv file 
and extracting the frames from the subjects listed in that csv

"""

import sys
import os

import basic_utils.basics as base
import basic_utils.video_core as vc


def usage(with_help=True):
    print("[Usage]: %s <csv_file> <data_dir> <output_directory>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to extract video files from a specified data_dir ")
    print("\t| by reading a .csv file and convert the videos to frame data\n")
    usage(with_help=False)


if __name__ == "__main__":

    try:
        if sys.argv[1] in ["HELP", "help", "h"]:
            help_msg()

        csv_name = sys.argv[1]
        data_dir = sys.argv[2]
        output_dir = sys.argv[3]
        
        base.check_exists_create_if_not(output_dir)

        specs, header = base.csv2data(csv_name)
        
        fmt_dir = "S%04d"
        fmt_file = "Trial%d.MOV"

        imgs_captured = []
        for row in specs: 
            
            subject, trial = row
            
            subject = int(subject)
            
            if subject == 23:
                continue
            
            trial = int(trial)
            
            target_dir = os.path.join(data_dir, fmt_dir % subject)
            target_file = fmt_file % trial

            target = os.path.join(target_dir, target_file)
            
            imgs = vc.video_file_to_frames(target, output_dir=output_dir, suppress=False)
            print("-"*78)
            imgs_captured.extend(imgs)
        
        print("[*] Extracted %d images from %d different video files" % (len(imgs_captured), len(specs)))

    except IndexError:
        usage()
