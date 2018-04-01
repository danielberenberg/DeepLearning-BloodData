import csv
import random
import os

data_path = "consolidated"
partition_csv = "partitions_out.csv"
filtered_csv = "NextStartingPoint.csv"

percent = 0.2
def main():
    subjects, trial1, trial2 = all_subjects()

    
    training_set = split_subjects(trial1, trial2, 0.20)
    validation_set = split_subjects(trial1, trial2, 0.10)

    trial1.extend(trial2)
    testing_set = trial1 
    
    all_paths = fetch_paths_with_labels(partition_csv)
    
    filter_path_with_set(testing_set, all_paths)
    #print(all_paths)
    return

def filter_path_with_set(filter_set, all_paths):
    filtered_set_paths = {}
    for subject, trial in filter_set:
        subj_tri = "S{}_t{}".format("0" * (4 - len(subject)) + subject, trial)
       
        #¯\_﹙ツ﹚_/¯
        #filter out only the keys of the dictionary the belong to this subject and trial
        filtered_keys = [key for key in all_paths.keys() 
                if key.split('/')[1].split('_')[0] == subj_tri.split('_')[0] 
                and key.split('/')[1].split('_')[1] == subj_tri.split('_')[1]]
        
        print(filtered_keys)
        #if not filtered_keys:
            #print(subj_tri)
        
        #for path_key in all_paths:
        #    split_key = path_key.split('_')
        #    subj_tri_key = split_key[0] + '_' + split_key[1]
        #    if subj_tri == subj_tri_key:
        #        prepare_removal = []

def fetch_paths_with_labels(consolidated_csv): 
    with open(consolidated_csv, 'r') as p_csv:
        reader = csv.reader(p_csv)
        next(reader) #skip the header
        paths = {}
        for path in reader:
            full_path = os.path.join(data_path, path[0])
            if not os.path.exists(full_path):
                raise FileNotFoundError("{} does not exist!".format(full_path))
            paths[full_path] = (path[1], path[2])
    return paths
        
def split_subjects(trial1, trial2, split_p=0.2):
    smallest = min(len(trial1), len(trial2))
    
    num_split = int((smallest * split_p))
    num_split = 1 if num_split == 0 else num_split
    split_set = []
    while len(split_set) < num_split * 2:
        trial1_index = random.randint(0, len(trial1)-1)
        trial2_index = random.randint(0, len(trial2)-1)
        split_set.append(trial1[trial1_index])
        split_set.append(trial2[trial2_index])
        trial1.pop(trial1_index)
        trial2.pop(trial2_index)  
    
    return split_set

def all_subjects():
    with open(filtered_csv, 'r') as starting_point:
        reader = csv.reader(starting_point)
        next(reader)
        
        all_subj = []
        trial1 = []
        trial2 = []
        for subj in reader:
            if subj[0] not in all_subj:
                all_subj.append(subj[0])
            if subj[1] == "1":
                trial1.append(subj)
            else:
                trial2.append(subj)
        return all_subj, trial1, trial2


main()
