import os
import cv2
import numpy as np
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

SLOW_TRIALS = "Trial1_frames"
FAST_TRIALS = "Trial2_frames"

SLOW_LABEL = "slow"
FAST_LABEL = "fast"

def create_image_array(frames_list):
    """
    Turn a list of paths of image frames into a list of images.
    
    args:
        frames_list: list of image frames
    return:
        -> an np array of images
    """
    imgs = []
    for frame in frames_list:
        img = cv2.imread(frame)
        imgs.append(img)
    return np.array(imgs)

def package_data(frames_dir, csv_in):
    """
    Given a directory containing partitioned videos for training and an csv file
    containing data details, package the data up such that each partitioned directory
    is returned as a list of image frame paths, with an accompanying list containing the
    appropriate label for each video partition.

    A list of frame paths are returned instead of a list of images as to reduce the memory
    required at one time by the neural network. This way, instead of several hundred video clips
    being loaded at once, only small batches of videos need to be loaded, leveraging the
    paths returned by this method to load the clips.

    args:
        frames_dir: the name of the directory containing all partitioned videos
        csv_in: the csv file containing the data relevant to each photo
    return:
        -> the training data, represented as lists of image file paths, and an accompanying
            list of labels.
    """
    SUBJ = 0
    PART = 1
    TRI1 = 2
    TRI2 = 4
    with open(csv_in, "r", newline='') as csv_file:
        info = csv_file.readlines()
        info = info[1:]

        X = []
        Y = []
        for line in info:
            line = line.split(",")
            #If the trial exists
            if line[TRI1] != "":
                x, y = create_labeled_data_from_partition(frames_dir, line[SUBJ], 
                        SLOW_TRIALS, line[PART], int(line[TRI1]))
                X.append(x)
                Y.append(y)
            #If the trial exist 
            if line[TRI2] != "":
                x, y = create_labeled_data_from_partition(frames_dir, line[SUBJ], 
                        FAST_TRIALS, line[PART], int(line[TRI2]))
                X.append(x)
                Y.append(y)
        return X, Y

def create_labeled_data_from_partition(frames_dir, subject, trial_name, partition, rate):
    """
    Create a list of image file paths, corresponding to one video partition, and a label.
    The path is constructed, and a label chosen, using data pulled from the csv file.
    The image file paths are sorted chronologically.
    Calling this method produces one "value" to be used for training or testing, with
    an accompanying label.

    args:
        frames_dir: the name of the directory containing all partitioned videos
        subject: the subject number that training data is being created for
        trial_name: the trial (1 or 2) that is being used
        partition: the partition number that is being used
        rate: the heart rate of the subject
    return:
        -> a single training data point with an accompanying label
    """
    zeros = 4 - len(subject)
    subject = "S" + (zeros * "0") + subject
    full_part_path = os.path.join(frames_dir, subject, trial_name, partition)
    data = []
    for frame in sorted(os.listdir(full_part_path), key=numericalSort):
        frame_path = os.path.join(full_part_path, frame)
        data.append(frame_path)
    label = FAST_LABEL if rate > 100 else SLOW_LABEL
    return data, label
