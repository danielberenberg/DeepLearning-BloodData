import csv
import subprocess
import os
import sys
import pandas as pd

import we_panic_utils.basic_utils.basics as base
import we_panic_utils.basic_utils.video_core as vc
data_dir = 'data'
selected = 'NextStartingPoint.csv'
output_dir = 'augmented'


UPPER_THRESHOLD = 230
LOWER_THRESHOLD = 35

if __name__ == "__main__":
    try:
        data_dir = sys.argv[1]
        master_csv = sys.argv[2]
        selected = sys.argv[3]
        output_dir = sys.argv[4]
    except:
        print("\tUsage: <data_dir> <master_csv> <selected_subjects.csv> <output_dir>")
    else:        
        base.check_exists_create_if_not(output_dir)
        
        selected_df = pd.read_csv(selected)
        data = base.csv2data(master_csv)[0]

        filtered = zip(list(selected_df["Subject"]), list(selected_df["Trial"]))
        
        i = 0
        for subj, trial in filtered:
            subj_row = data[subj-1]
            heart_rate, resp_rate = 0, 0
            if trial == 1:
                heart_rate, resp_rate = float(subj_row[1]), float(subj_row[2])
            elif trial == 2:
                heart_rate, resp_rate = float(subj_row[3]), float(subj_row[4])
           
            subj_origin = os.path.join(data_dir, "S%04d" % subj, "Trial%d.MOV" % trial)
            assert os.path.exists(subj_origin)
            
            base.check_exists_create_if_not(os.path.join(output_dir, "S%d%03d" % (trial, subj)))

            subj_target_fast = os.path.join(output_dir, "S%d%03d" % (trial, subj), "Trial1.MOV")
            subj_target_slow = os.path.join(output_dir, "S%d%03d" % (trial, subj), "Trial2.MOV")
            
            if heart_rate * 2 <= UPPER_THRESHOLD:
                vc.change_speed(subj_origin, subj_target_fast, 0.5)
            else:
                print("Not doubling the speed of subject {} trial {}: resulting heart rate would\
                        have been {}".format(subj, trial, heart_rate * 2))
            if heart_rate / 2 >= LOWER_THRESHOLD:
                vc.change_speed(subj_origin, subj_target_slow, 2)
            else:
                print("Not halving the speed of subject {} trial {}: resulting heart rate would\
                        have been {}".format(subj, trial, heart_rate / 2))
            i += 1
            if i == 4:
                break
        #data = [s for s in data if int(s[0]) in list(selected_df["Subject"])]
        #for subject, tri1hr, tri1rr, tri2hr, tri2rr in data:
         #   print(subject)


        #master_df = pd.read_csv(master_csv)
        
        
        #print(master_df)


        #with open(selected, 'r') as video_csv:
        #    csvreader = csv.reader(video_csv)
        #    next(csvreader) #skip the header
        #    for line in csvreader:
        #        subj_path = vc.fetch_path(line[0], data_dir)
        #        trial = line[1]
        #        trial_path = ""
        #        for video in os.listdir(subj_path):
        #            if trial in video:
        #                if trial == "1":
        #                    speed = 0.5
        #                    #ext = "FAST"
        #                    trial_path = video
        #                else:
        #                    speed = 2
        #                    #ext = "SLOW"
        #                    trial_path = video
        #                break
        #        out = os.path.join(output_dir, subj_path.split('/')[1])
        #        base.check_exists_create_if_not(out)
        #        new_name = os.path.join(out, trial_path)
        #        vc.change_speed(os.path.join(subj_path, video), new_name, speed)
