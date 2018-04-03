from . import split_utils as util
import we_panic_utils.basic_utils.basics as base

import csv
import os
import sys
"""
Public API
"""

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

def data_set_to_csv(data_set, csv_path, verbose=True):
    if not csv_path.endswith('.csv'):
        csv_path += '.csv'
    if verbose:
        print('[data_set_to_csv]: creating csv at -> {}'.format(csv_path))
    with open(csv_path, 'w') as csv_out:
        writer = csv.writer(csv_out)
        header = ['PATH', 'HEART RATE', 'RESPIRATORY RATE']
        writer.writerow(header)
        for key in data_set:
            data = data_set[key]
            writer.writerow([key, data[0], data[1]])

def train_test_split_with_csv_support(regular_data_path, filtered_csv, consolidated_csv, dir_out, 
        augmented_data_path=None, test_split=0.2, val_split=0.1, train_csv_out="train.csv", 
        test_csv_out="test.csv", val_csv_out="val.csv", verbose=True): 
    """
    Split all available data into train, test, and validation sets.
    This splitting method does not allow the same trial of the same subject to appear in more than one set,
    e.g., if s1 t1 appears in the training set, then any partition of s1 t1 cannot appear in
    testing or validation. If the path for speed augmented data is provided, then whenever any subject
    and trial is added to a set, its associated augmented partition will also be added.

    Three csv files will be created upon calling this method. These csv files record the data
    delegated to each of the training, testing, and validation sets, so that they may be loaded in the future.
    
    Example usage:
    train, test, split = train_test_split_with_csv_support('reg_consolidated', 'NextStartingPoint.csv', 
    'reg_part_out.csv', 'test3', augmented_data_path='aug_consolidated')
    
    args:
        regular_data_path : path of the directory containing all of the data
        filtered_csv : path to the csv containing the subjects and trials that will be used in the data split
        consolidated_csv : path to the csv containing the path, heart rate, and respiratory rate of each data point
        dir_out : directory that the csv files will be written to
        (optional) augmented_data_path : path to the directory containin the augmented data
        (optional) test_split : percentage of data to be assigned to the test set
        (optional) val_split : percentage of data to be assigned to the validation set
        (optional) verbose : boolean which controls whether or not the program will log each action taken 
    returns:
        -> a map containing full path keys of data to heart rate, respiratory rate pairs
    """

    if not os.path.exists(regular_data_path):
        raise FileNotFoundError("Data path {} does not exist".format(regular_data_path))
    if not os.path.isdir(regular_data_path):
        raise ValueError("Data path {} is not a directory".format(regular_data_path))
    if augmented_data_path != None and not os.path.exists(augmented_data_path):
        raise FileNotFoundError("Augmented data path {} does not exist".format(augmented_data_path))
    if augmented_data_path != None and not os.path.isdir(augmented_data_path):
        raise ValueError("Augmented data path {} is not a directory".format(augmented_data_path))    
    if not os.path.exists(filtered_csv):  
        raise FileNotFoundError("Chosen subjects csv path {} does not exist".format(filtered_csv))        
    if not os.path.exists(consolidated_csv):  
        raise FileNotFoundError("Consolidated csv path {} does not exist".format(consolidated_csv)) 
    
    if verbose:
        if augmented_data_path == None:
            print('[train_test_split_with_csv_support]: splitting data without augmented data')
        else:
            print('[train_test_split_with_csv_support]: splitting data with augmented data directory -> {}'.format(augmented_data_path))

    trial1, trial2 = util.all_subjects(filtered_csv)
    testing_set = util.split_subjects(trial1, trial2, test_split)
    validation_set = util.split_subjects(trial1, trial2, val_split)
    
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
   
    base.check_exists_create_if_not(dir_out, not verbose)

    data_set_to_csv(filtered_training_paths, os.path.join(dir_out, train_csv_out), verbose=verbose) 
    data_set_to_csv(filtered_testing_paths, os.path.join(dir_out, test_csv_out), verbose=verbose)
    data_set_to_csv(filtered_validation_paths, os.path.join(dir_out, val_csv_out), verbose=verbose)
    
    return filtered_training_paths, filtered_testing_paths, filtered_validation_paths
