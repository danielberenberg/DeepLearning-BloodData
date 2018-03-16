from keras.layers import Conv3D, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

from data_utils import create_image_array, package_data 

import numpy as np
import os
import sys
import cv2
from sklearn.utils import shuffle

BATCH_SIZE = 4
CLASSES = 2
EPOCH = 20

def usage(with_help=True): 
    print("[Usage]: %s <frames_dir> <csv_file>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()

def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0])) 
    print("\t| Running this script will train a 3D-CNN.")
    print("\t| Using keras front-end with tensorflow backend")
    usage(with_help=False)

def train_network(X, Y):
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y)
    Y = np_utils.to_categorical(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    model = Sequential()
    
    model.add(Conv3D(32, kernel_size=(3,3,3), 
           input_shape=(60, 224, 224, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
    model.add(Dropout(0.5)) 
    model.add(Conv3D(64, kernel_size=(3,3,3), 
           input_shape=(60, 224, 224, 3), activation='relu')) 
    model.add(MaxPooling3D(pool_size=2, strides=2))
    model.add(Dropout(0.5))
    model.add(Conv3D(128, kernel_size=(3,3,3), 
           input_shape=(60, 224, 224, 3), activation='relu')) 
    model.add(Conv3D(128, kernel_size=(3,3,3), 
           input_shape=(60, 224, 224, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=2, strides=2))
    
    model.add(Flatten())
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dense(2))
    
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("[{}]: Model compiled. Training on {} samples...".format(sys.argv[0],
        len(X_train)))
    try:
        hist = {}
        for e in range(1, EPOCH+1):
            print('Epoch {}/{}:'.format(e, EPOCH))
            shuffled_x, shuffled_y = shuffle(X_train, Y_train)
            for i in range(0, len(X_train), BATCH_SIZE):
                vid_batch = shuffled_x[i:i+BATCH_SIZE]
                batch_x = []
                for vid in vid_batch:
                    batch_x.append(create_image_array(vid))
                batch_y = shuffled_y[i:i+BATCH_SIZE]
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                model.fit(batch_x, batch_y)

        loss = 0 
        acc = 0
        print("[{}]: Model finished training. Testing on {} samples".format(sys.argv[0], 
            len(X_test)))
        for i in range(0, len(X_test)):
            video = create_image_array(X_test[i]) 
            video = np.expand_dims(video, 0)
            label = np.expand_dims(Y_test[i], 0)
            result = model.evaluate(video, label, verbose=0)
            loss+= result[0]
            acc+= result[1]
        print("[{}]: average loss on testing {} samples: {}".format(sys.argv[0], len(X_test), loss/len(X_test)))
        print("[{}]: average accuracy on testing {} samples: {}".format(sys.argv[0], len(X_test), acc/len(X_test)))
        model.save_weights('3D_CNN_Weights.h5')
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Saving weights...")
        model.save_weights('3D_CNN_Weights.h5')
    except Exception as e:
        print("Something else went wrong. Saving weights...") 
        model.save_weights('3D_CNN_Weights.h5')
        raise e
if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args == 2 and sys.argv[1] in ["HELP", "help", "h"]:
        help_msg()
    elif num_args < 3:
        usage()
    frames_dir = sys.argv[1]
    csv_file = sys.argv[2]
    if not os.path.exists(frames_dir):
        raise IOError("Error: frames path {} not found".format(frames_dir))
    if not os.path.exists(csv_file):
        raise IOError("Error: csv file {} not found".format(csv_file))
    X, Y = package_data(frames_dir, csv_file)
    print("[{}]: labeled data created. Initializing network.".format(sys.argv[0]))
    train_network(X, Y)
