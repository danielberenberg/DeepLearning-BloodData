# Analyzing Video Data to Determine Heart Rate/Respiratory Rate

## Overview

There is promising evidence to suggest that pulse is detectable by smartphone cameras: one can lightly press their finger up to a smartphone camera with flash on, and produce a video that clearly shows their pulse. This project aims to leverage deep learning techniques to predict a patient’s heart rate and respiratory rate, given short videos of this nature. 

To run this project, you will need to collect samples yourself, or be granted permission to see the existing samples. Contact repo owners for more information. This project expects each clip to be around 30 seconds long, and for the movies to be organized in the following way:

-> movie_data <br />
->     | -> subject_1 <br />
->     | ->     | -> trial_1.mov <br />
->     | ->     | -> trial_2.mov <br />
... <br />
->     | -> subject_n <br />
->     | ->     | -> trial_1.mov <br />
->     | ->     | -> trial_2.mov <br />

Where each subject directory contains two video trials: one at resting heart rate, and one after 60 seconds following an intense workout. This project also assumes the existence of two csv files. The first one contains trial data for each subject in the following format:

| Subject  | Trial1_Heart_Rate| Trial2_Heart_Rate | Trial1_Respiratory_Rate| Trial2_Respiratory_Rate |
|:--------:|:----------------:|:-----------------:|:----------------------:|:-----------------------:|
| 1        | 60               | 120               | 30                     | 45                      |
| ..       | ..               | ..                | ..                     | ..                      |

This second csv file contains trial data of only those subjects that you want to use for training. It should be in the following format:

| Subject  | Trial|
|:--------:|:----:|
| 1        | 1    |
| 1        | 2    |
| ..       | ..   |

## Install Requirements

This project assumes anaconda is being used.

First, install pip if you do not already have it.
```{r, engine='bash'}
conda install pip
```

Then, install the required packages for data and video utilities. Run the following command:
```{r, engine='bash'}
pip install src/we_panic_utils/
```

## Preprocess Data

Once you have the above steps completed, you can begin preprocessing data. Run the following command:
```{r, engine='bash'}
./preprocess.sh <movie_directory/> <subjects_for_training.csv>
```
This does the following:
* convert each .mov file into a directory of frames and convert each .mov file into 30 fps
* partition each frame directory into 2 second partitions
* resize all frames to 225x225x3

This may take upwards of 30 minutes to finish.

## Tasks (As of 3/31/18):

- [x] resize frames
- [x] partition videos
- [x] train classifier
- [ ] introduce data augmentation
- [ ] train regression model
