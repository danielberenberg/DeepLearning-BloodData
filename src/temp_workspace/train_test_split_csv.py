import csv
import random
import os

def main():
    
    regular_data_path = "reg_consolidated"
    augmented_data_path = "aug_consolidated"
    filtered_csv = "NextStartingPoint.csv" 
    consolidated_csv = "reg_part_out.csv"
    
    train, test, val = train_test_split_with_csv(regular_data_path, augmented_data_path, filtered_csv, consolidated_csv) 
    print(test)

def train_test_split_with_csv(regular_data_path, augmented_data_path, filtered_csv, consolidated_csv,
        train_csv_out="train.csv", test_csv_out="test.csv", val_csv_out="val.csv"): 

    trial1, trial2 = all_subjects(filtered_csv)
    testing_set = split_subjects(trial1, trial2, 0.20)
    validation_set = split_subjects(trial1, trial2, 0.10)

    trial1.extend(trial2)
    training_set = trial1 
    all_paths = fetch_paths_with_labels(consolidated_csv, regular_data_path)

    filtered_training_paths = filter_path_with_set(training_set, all_paths, augmented_data_path)
    filtered_testing_paths = filter_path_with_set(testing_set, all_paths, augmented_data_path)
    filtered_validation_paths = filter_path_with_set(validation_set, all_paths, augmented_data_path)
    
    return filtered_training_paths, filtered_testing_paths, filtered_validation_paths

def filter_path_with_set(filter_set, all_paths, augment_path=None):
    filtered_set_paths = {}
    for subject, trial in filter_set:
        subj_tri = "S{}_t{}".format("0" * (4 - len(subject)) + subject, trial)
       
        #¯\_﹙ツ﹚_/¯
        #filter out only the keys of the dictionary the belong to this subject and trial
        filtered_keys = [key for key in all_paths.keys() 
                if key.split('/')[1].split('_')[0] == subj_tri.split('_')[0] 
                and key.split('/')[1].split('_')[1] == subj_tri.split('_')[1]]
        
        
        for key in filtered_keys:
            filtered_set_paths[key] = all_paths[key]
            if augment_path != None:
                try:
                    augmented = fetch_augmented(key, augment_path)
                except FileNotFoundError as e:
                    print(e)
                else:
                    heart_rate, resp_rate = filtered_set_paths[key]
                    heart_rate = int(heart_rate)
                    resp_rate = int(resp_rate)
                    if augmented.split('_')[2] == "t1":
                        heart_rate *= 2
                    else:
                        heart_rate //= 2
                    resp_rate = int(round(heart_rate/4))
                    filtered_set_paths[augmented] = (heart_rate, resp_rate)
            del(all_paths[key])

    return filtered_set_paths

def fetch_augmented(path, augmented_path):
    without_dir = path.split('/')[1]
    with_augmented = os.path.join(augmented_path, without_dir)
    if not os.path.exists(with_augmented):
        raise FileNotFoundError("{} does not exist!".format(with_augmented))
    return with_augmented

def fetch_paths_with_labels(consolidated_csv, data_path): 
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

def all_subjects(filtered_csv):
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
        return trial1, trial2

main()
