import csv
import subprocess
import os
import sys
import we_panic_utils.basic_utils.basics as base
import we_panic_utils.basic_utils.video_core as vc
data_dir = 'data'
selected = 'NextStartingPoint.csv'
output_dir = 'augmented'


if __name__ == "__main__":
    try:
        data_dir = sys.argv[1]
        selected = sys.argv[2]
        output_dir = sys.argv[3]
    except:
        print("\tUsage: <data_dir> <selected_subjects.csv> <output_dir>")
    else:        
        base.check_exists_create_if_not(output_dir)

        with open(selected, 'r') as video_csv:
            csvreader = csv.reader(video_csv)
            next(csvreader) #skip the header
            for line in csvreader:
                subj_path = vc.fetch_path(line[0], data_dir)
                trial = line[1]
                trial_path = ""
                for video in os.listdir(subj_path):
                    if trial in video:
                        if trial == "1":
                            speed = 0.5
                            #ext = "FAST"
                            trial_path = video
                        else:
                            speed = 2
                            #ext = "SLOW"
                            trial_path = video
                        break
                out = os.path.join(output_dir, subj_path.split('/')[1])
                base.check_exists_create_if_not(out)
                new_name = os.path.join(out, trial_path)
                vc.change_speed(os.path.join(subj_path, video), new_name, speed)
