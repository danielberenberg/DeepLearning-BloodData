import os
import sys
import basic_utils.basics as base
import basic_utils.video_core as vc

INPUT_CSV = "DeepLearningClassData.csv"

def usage(with_help=True): 
    print("[Usage]: %s <frame_dir> <patition_out> <output_csv>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to take a directory of video frames and")
    print("\t| partition them into several 2 second long collections")
    usage(with_help=False)

def extract_subject_name(subj_dir):
    subj_dir = subj_dir[1:]
    return str(int(subj_dir))

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args == 2 and sys.argv[1] in ["HELP", "help", "h"]:
        help_msg()
    elif num_args < 4:
        usage()
     
    frames_path   = sys.argv[1]
    partition_dir = sys.argv[2]
    output_csv    = sys.argv[3]
    
    if not os.path.exists(frames_path):
        raise IOError("Error: frames path not found | " + frames_path)
    
    if not output_csv.endswith('.csv'):
        output_csv += '.csv'

    csv_in = open(INPUT_CSV, 'r')
    csv_out = open(output_csv, 'w')
    helper = base.CSV_Helper(csv_in, csv_out)
    
    base.check_exists_create_if_not(partition_dir)
    
    for frame_dir in os.listdir(frames_path):
        trial_path = os.path.join(frames_path,frame_dir)
        if not os.path.isdir(trial_path):
           continue
        subj_name = extract_subject_name(frame_dir)
        #For some reason .DS_STORE is included when we traverse the directories, 
        #so we need to account for it
        
        for trial_dir in os.listdir(trial_path):
            frame_path = os.path.join(trial_path, trial_dir)
            output_path = os.path.join(partition_dir, frame_dir, trial_dir)
            if not os.path.isdir(frame_path):
                continue

            print('*'*85)
            num_partitions = vc.partition_frame_dir(frame_path, output_dir=output_path)
        data = helper.look_up(subj_name)
        for p in range(num_partitions):
            helper.write_to(subj_name, p, data)
    helper.release()

