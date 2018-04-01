"""
Convert the input csv to one that mirrors the dir/subdir
format in consolidated data directory
"""

import sys, os
import basic_utils.basics as base

def usage():
    print("[usage]: python %s <partition-csv>" % sys.argv[0])
    sys.exit()
 
def parse_input():
    try:
        dir_ = sys.argv[1]

        if os.path.exists(dir_):
            return dir_

        else:
            raise OSError("%s not found" % dir_)

    except IndexError:
        usage()

if __name__ == "__main__":
    csv_ = parse_input()
    

    mat, header = base.csv2data(csv_)
    csv_ = open(csv_, "r")
    output_csv = open("partitions_out.csv","w")
    
    header = ["SAMPLE","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"] 
    csvh = base.CSV_Helper(csv_, output_csv, header=header)
    #header = ["SUBJECT","TRIAL","PARTITION","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"]
    for row in mat:
        subj, tr, part = [int(r) for r in row[:3]]
        fmt_slug = "S%04d_t%d_p%02d" % (subj, tr, part)
        
        newr = [fmt_slug]
        newr.extend(row[3:])

        csvh.csv_writer.writerow(newr)
    
    csvh.release()

    print("done!")
