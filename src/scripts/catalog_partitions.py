"""
Catalog partitions in a .csv of the form:

SUBJECT     TRIAL       PARTITION       HEART RATE      RESPIRATORY RATE
"""

import os
import sys
import basic_utils.basics as base

def usage():
    print("[usage]: python %s <partition-dir> <master_csv>" % sys.argv[0])
    sys.exit();

def parse_input():
    try:
        dir_ = sys.argv[1]
        csv_ = sys.argv[2]

        if os.path.isdir(dir_) and os.path.exists(csv_):
            return dir_, csv_

        else:
            raise OSError("%s not found" % dir_)

    except IndexError:
        usage()

if __name__ == "__main__":
    dir_, csv_ = parse_input()
    hrhigh = 0
    hrlow = 0
    mat, header = base.csv2data(csv_)
    
    csv_ = open(csv_, "r")
    #print(header)
    output_csv = open("partitions.csv","w")
    
    exclude = [".DS_Store","._.DS_Store"]
    
    csvh = base.CSV_Helper(csv_, output_csv)
    header = ["SUBJECT","TRIAL","PARTITION","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"]
    
    csvh.generate_header(header)
    #print(mat)
    # gives subjects
    for child in sorted(os.listdir(dir_)):
        if os.path.isdir(os.path.join(dir_, child)) and child not in exclude:
            pth = os.path.join(dir_, child)
            
            # gives TrialN_frames
            for grandchild in sorted(os.listdir(pth)):
                if grandchild not in exclude:
                    
                    fullpth = os.path.join(pth, grandchild)
                    
                    # gives partition namees
                    for greatgc in sorted(os.listdir(fullpth)):
                        if greatgc not in exclude:
                            
                            # write to output_csv the follwoing
                            # SUBJECT
                            # TRIAL
                            # PARTITION
                            # HEART RATE 
                            # RESPIRATORY RATE

                            SUBJECT = str(int(child[1:]))
                            TRIAL   = grandchild.split("_")[0][-1]
                            PARTITION = str(int(greatgc))
                            row = mat[int(SUBJECT)-1]
                            
                            HEART_RATE, RESPIRATORY_RATE = "", ""
                            HEART_RATE_CLASS = "LOW"
                            if int(TRIAL) == 1:
                                HEART_RATE, RESPIRATORY_RATE = row[1:3]
                            else:
                                HEART_RATE, RESPIRATORY_RATE = row[3:]

                            if int(HEART_RATE) >= 100:
                                HEART_RATE_CLASS = "HIGH"
                                hrhigh+=1

                            else:
                                hrlow+=1

                            csvh.csv_writer.writerow([SUBJECT, TRIAL, PARTITION, HEART_RATE, RESPIRATORY_RATE,HEART_RATE_CLASS])
                            #print([SUBJECT, TRIAL, PARTITION])
    csvh.release()

    print("done! %d samples with high heart rate, %d samples with low heart rate" % (hrhigh,hrlow))
