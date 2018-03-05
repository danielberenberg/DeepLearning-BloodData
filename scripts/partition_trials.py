import csv
import os
import sys
import basic_utils.basics as base
import basic_utils.video_core as vc
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
    
    base.check_exists_create_if_not(partition_dir)
    
    #print([x for x in os.listdir(frames_path) if x != ".DS_Store"])
    for frame_dir in os.listdir(frames_path):
        trial_path = os.path.join(frames_path,frame_dir)

        #For some reason .DS_STORE is included when we traverse the directories, 
        #so we need to account for it
        if not os.path.isdir(trial_path):
            continue
        
        for trial_dir in os.listdir(trial_path):
            frame_path = os.path.join(trial_path, trial_dir)
            output_path = os.path.join(partition_dir, frame_dir, trial_dir)
            if os.path.isdir(frame_path):
                vc.partition_frame_dir(frame_path, output_dir=output_path)


#ignore this
def test_csv():
    with open('test.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Subject', 'Trial', 'Partition', 'Heart Rate', 'Respiratory Rate'])
        csv_writer.writerow(['2', '1', '1', '80', '30'])
        csv_writer.writerow(['Spam'] * 5 + ['Baked Beans'])

