from . import split_utils as util
import we_panic_utils.basic_utils.basics as base

import csv
import os
import sys
import pandas as pd
import random

"""
Public API
"""


def data_set_from_csv(csv_path, augmented_dir=None):
    ignore = False

    if augmented_dir is not None:
        ignore = True

    if not os.path.exists(csv_path):
        raise FileNotFoundError("{} was not found".format(csv_path))
    data = {}
    with open(csv_path, 'r') as csv_in:
        reader = csv.reader(csv_in)
        next(reader)
        for path, hr, rr in reader:
            if not ignore or augmented_dir not in path:
                # data[path] = (hr, rr)

                data[path] = (int(hr), int(rr))
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

            # writer.writerow([key, data[0], data[1]])
            writer.writerow([key, str(data[0]), str(data[1])])


def generate_paths2labels(df, data_path):
    """
    create the sought after and beloved `paths2labels`
    also know as `filtered_#$@%ing_paths` from a dataframe
    
    format assuming "S%4d/Trial%d_frames" is the name of every
    directory inside of data_path

    args:
        df : DataFrame
        data_path : path to data
    """
    
    filtered_paths = dict()
   
    subj_fmt = "S%04d"
    trial_fmt = "Trial%d_frames"

    for row in df.iterrows():
        subject, trial, h_rate, resp_rate = row['Subject'], row['Trial'], row['Heart Rate'], row['Respiratory Rate']
    
        subject = subj_fmt % subject
        trial = trial_fmt % trial
    
        P = os.path.join(data_path, subject, trial)
        filtered_paths[P] = (h_rate, resp_rate)

    return filtered_paths


def ttswcsv2(data_path, metadata, output_dir,
             test_split=0.2, val_split=0.1,
             csvnames={'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}, verbose=True):
    """
    Train Test Split With Csv Support Version 2
    perform a split on metadata into separate sets, ensure that no 1 subject
    spans across 2 sets
    
    the idea for this function is to use it on any frame directory and corresponding
    metadata csv and so the usage for the combination of directories would be to 
    take the union of the output here
    
    the way the `ignore_augmented` functionality is implemented in this function
    is to simply provide 'set_name' in csvnames to be None, or simply exclude 'set_name' from
    csvnames entirely

    eg
    --
    suppose the data_path passed in is the augmented set of frames and we do not want to test on this set.
    the function signature is then:

        ttswcsv2(<augmented_path>, 
                 <metadata>, 
                 <output_dir>, 
                 test_split=<$>, 
                 val_split=<$>,
                 csvnames = {'train':<train_name>, 'val':<val_name>'})

    args:
        data_path : str - path to frame directory
        metadata : the subject, trial, hrate, resprate csv
        output_dir : the output directory
        test_split : percentage of full data set approximately the test set should be
        val_split : percentage of the train set approximately to allocate for validation
        csvnames : the names of the output csvs, relative to the output_dir
    """
    
    if type(csvnames) is not dict:
        raise ValueError("got a bum csvname parameter of type %s, should be a dict" % str(type(csvnames))) 
   
    if 'train' not in csvnames:
        csvnames['train'] = None

    if 'val' not in csvnames:
        csvnames['val'] = None

    if 'test' not in csvnames:
        csvnames['test'] = None

    metadf = pd.read_csv(metadata)
    instances = metadf['Subject'] 
    subjects = list(set(instances))
    
    random.shuffle(subjects)
    
    filtered_training_paths, filtered_testing_paths, filtered_validation_paths = None, None, None
    
    N = len(subjects)
    
    if csvnames['test'] is not None:
        test = int(test_split * N)
        test_samples = subjects[:test]
        subjects = subjects[test:]
        
        test_df = metadf[metadf['Subject'].isin(test_samples)]
        test_df.to_csv(os.path.join(output_dir, csvnames['test']))
        
        filtered_testing_paths = generate_paths2labels(test_df, data_path)

    random.shuffle(subjects)
    
    N = len(subjects)
    
    if csvnames['val'] is not None:
        val = int(val_split * N)
        val_samples = subjects[:val] 
        subjects = subjects[val:]
        
        val_df = metadf[metadf['Subject'].isin(val_samples)]
        val_df.to_csv(os.path.join(output_dir, csvnames['val']))
        
        filtered_validation_paths = generate_paths2labels(val_df, data_path)

    if csvnames['train'] is not None:
        train_samples = subjects
        train_df = metadf[metadf['Subject'].isin(train_samples)]
        train_df.to_csv(os.path.join(output_dir, csvnames['train']))

        filtered_training_paths = generate_paths2labels(train_df, data_path)
    
    # return train_df, test_df, val_df
    return filtered_training_paths, filtered_testing_paths, filtered_validation_paths 


def train_test_split_with_csv_support(regular_data_path, filtered_csv, consolidated_csv, dir_out, 
        augmented_data_path=None, ignore_augmented=[], test_split=0.2, val_split=0.1, train_csv_out="train.csv", 
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
    filtered_testing_paths = util.filter_path_with_gtset(testing_set, all_paths, augmented_data_path, verbose)
    filtered_validation_paths = util.filter_path_with_set(validation_set, all_paths, augmented_data_path, verbose)
   
    base.check_exists_create_if_not(dir_out, not verbose)

    data_set_to_csv(filtered_training_paths, os.path.join(dir_out, train_csv_out), verbose=verbose) 
    data_set_to_csv(filtered_testing_paths, os.path.join(dir_out, test_csv_out), verbose=verbose)
    data_set_to_csv(filtered_validation_paths, os.path.join(dir_out, val_csv_out), verbose=verbose)
    
    if "train" in ignore_augmented:
        filtered_training_paths = {path : filtered_training_paths[path] for path in filtered_training_paths if augmented_data_path not in path}
    if "test" in ignore_augmented:
        filtered_testing_paths = {path : filtered_testing_paths[path] for path in filtered_testing_paths if augmented_data_path not in path}
    if "validation" in ignore_augmented: 
        filtered_validation_paths = {path : filtered_validation_paths[path] for path in filtered_validation_paths if augmented_data_path not in path}

    return filtered_training_paths, filtered_testing_paths, filtered_validation_paths
