import sys
import os
import we_panic_utils.basic_utils.video_core as vc
import we_panic_utils.basic_utils.basics as base

def usage(with_help=True): 
    print("[Usage]: %s <partitions_dir> <resized_out>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()

def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to take a directory of")
    print("\t| frame partitions and resize them")
    usage(with_help=False)

def extract_subject_name(subj_dir):
    subj_dir = subj_dir[1:]
    return str(int(subj_dir))

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args == 2 and sys.argv[1] in ["HELP", "help", "h"]:
        help_msg()
    elif num_args < 3:
        usage()
     
    partitions_dir   = sys.argv[1]
    resize_path = sys.argv[2]
    
    if not os.path.exists(partitions_dir):
        raise FileNotFoundError("Error: path {} does not exists".format(partitions_dir))
    if not os.path.isdir(partitions_dir):
        raise IOError("Error: path {} is not a directory".format(partitions_dir))
    
    base.check_exists_create_if_not(resize_path)

    for subject in os.listdir(partitions_dir):
        trial_path = os.path.join(partitions_dir, subject)
        if not os.path.isdir(trial_path):
            continue

        for trial in os.listdir(trial_path):
            partition_path = os.path.join(trial_path, trial)
            if not os.path.isdir(partition_path):
                continue
            
            for partition in os.listdir(partition_path):
                frame_path = os.path.join(partition_path, partition)
                if not os.path.isdir(frame_path):
                    continue
                output_path = os.path.join(resize_path, subject, trial, partition)
                vc.resize_frame_dir(frame_path, output_path, width=100, height=100)
    print("Done.")

