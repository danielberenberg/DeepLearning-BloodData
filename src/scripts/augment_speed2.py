import csv
import subprocess
import os
import sys
import pandas as pd
import string

import we_panic_utils.basic_utils.basics as base
import we_panic_utils.basic_utils.video_core as vc

UPPER_THRESHOLD = 230
LOWER_THRESHOLD = 35

if __name__ == "__main__":
    try:
        data_dir = sys.argv[1]
        master_csv = sys.argv[2]
        selected = sys.argv[3]
    except:
        print("\tUsage: <data_dir> <master_csv> <selected_subjects.csv>")
    else:       
        #gives a list of values between 0.5 and 2 without 1
        speed_changes = [round(x*0.1, 1) for x in range(5, 21) if x != 10]
        mapping = {key : letter for key, letter in zip(speed_changes, string.ascii_lowercase)}
        
        selected_df = pd.read_csv(selected)
        data = base.csv2data(master_csv)[0]

        filtered = sorted(zip(list(selected_df["Subject"]), list(selected_df["Trial"])), key=lambda x: x[1])
         
        with open("aug_log.txt", "w") as aug_log, open(master_csv, 'a') as master:     
            master_writer = csv.writer(master)
            master_writer.writerow([])
            for subj, trial in filtered:
                subj_row = data[subj-1]
                heart_rate, resp_rate = 0, 0
                hr_index, rr_index = -1, -1
                if trial == 1:
                    heart_rate, resp_rate = float(subj_row[1]), float(subj_row[2])
                    hr_index = 1
                    rr_index = 2
                elif trial == 2:
                    heart_rate, resp_rate = float(subj_row[3]), float(subj_row[4])
                    hr_index = 3
                    rr_index = 4

                subj_origin = os.path.join(data_dir, "S%04d" % subj, "Trial%d.MOV" % trial)
                assert os.path.exists(subj_origin)
                
                for change in speed_changes:
                    if heart_rate * change > UPPER_THRESHOLD or heart_rate * change < LOWER_THRESHOLD:
                        log_str = "Not changing the speed of subject {} trial {} by factor of {}: resulting" \
                                " heart rate would have been {}".format(subj, trial, change, heart_rate * change)
                        print(log_str)
                        aug_log.write(log_str + '\n')
                    else:
                        """     s   hr1  rr1  hr2  rr2   """
                        """     |    |    |    |    |    """ 
                        row = ["",  "",  "",  "",  ""]
                        
                        new_subj = "%s%d%02d" % (mapping[change], trial,  subj)
                        new_subj_path = "S%s" % new_subj
                        subj_target = os.path.join(data_dir, new_subj_path, "Trial%d.MOV" % trial)
                        base.check_exists_create_if_not(os.path.join(data_dir, new_subj_path))
                        row[0] = new_subj
                        
                        row[hr_index] = round(heart_rate * change, 1)
                        row[rr_index] = (heart_rate * change) / 4
                        selected_df.loc[len(selected_df)] = [new_subj, trial]

                        vc.change_speed(subj_origin, subj_target, change)
                        master_writer.writerow(row) 
            
            selected_df.to_csv(selected, encoding='utf-8', index=False)
