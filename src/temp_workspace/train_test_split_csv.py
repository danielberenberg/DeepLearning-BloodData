import split_utils as util
import we_panic_utils.basic_utils.basics as base

import csv
import os

"""
Public API
"""

def main():
    
    regular_data_path = "reg_consolidated"
    augmented_data_path = "aug_consolidated"
    filtered_csv = "NextStartingPoint.csv" 
    consolidated_csv = "reg_part_out.csv"
    dir_out = "test/"

    train, test, val = train_test_split_with_csv(regular_data_path, augmented_data_path, filtered_csv, consolidated_csv, dir_out) 
    
    #data = data_set_from_csv('test/train.csv')
    #for key in data:
        #print(key, data[key])

def data_set_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("{} was not found".format(csv_path))
    data = {}
    with open(csv_path, 'r') as csv_in:
        reader = csv.reader(csv_in)
        next(reader)
        for path, hr, rr in reader:
            data[path] = (hr, rr)
    return data

def data_set_to_csv(data_set, csv_path):
    if not csv_path.endswith('.csv'):
        csv_path += '.csv'
    with open(csv_path, 'w') as csv_out:
        writer = csv.writer(csv_out)
        header = ['PATH', 'HEART RATE', 'RESPIRATORY RATE']
        writer.writerow(header)
        for key in data_set:
            data = data_set[key]
            writer.writerow([key, data[0], data[1]])


def train_test_split_with_csv(regular_data_path, augmented_data_path, filtered_csv, consolidated_csv, dir_out, 
        train_csv_out="train.csv", test_csv_out="test.csv", val_csv_out="val.csv", verbose=True): 
    
    base.check_exists_create_if_not(dir_out)

    trial1, trial2 = util.all_subjects(filtered_csv)
    testing_set = util.split_subjects(trial1, trial2, 0.20)
    validation_set = util.split_subjects(trial1, trial2, 0.10)
    
    trial1.extend(trial2)
    training_set = trial1 
    
    if verbose:
        print(util.set_to_str("----Training Set----", training_set))        
        print(util.set_to_str("----Validation Set----", validation_set))         
        print(util.set_to_str("----Testing Set----", testing_set))         
    
    all_paths = util.fetch_paths_with_labels(consolidated_csv, regular_data_path)

    filtered_training_paths = util.filter_path_with_set(training_set, all_paths, augmented_data_path, verbose)
    filtered_testing_paths = util.filter_path_with_set(testing_set, all_paths, augmented_data_path, verbose)
    filtered_validation_paths = util.filter_path_with_set(validation_set, all_paths, augmented_data_path, verbose)
   
    data_set_to_csv(filtered_training_paths, os.path.join(dir_out, train_csv_out)) 
    data_set_to_csv(filtered_testing_paths, os.path.join(dir_out, test_csv_out))
    data_set_to_csv(filtered_validation_paths, os.path.join(dir_out, val_csv_out))
    return filtered_training_paths, filtered_testing_paths, filtered_validation_paths

main()
