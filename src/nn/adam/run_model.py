from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
from data_utils import create_image_array, package_data, split_data, split_data_with_heart_rate
from generator import frame_generator
from classifier import Classifier
import os
import sys

BATCH_SIZE = 4
EPOCH = 40

def usage(with_help=True): 
    print("[Usage]: %s <frames_dir> <csv_file>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()

def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0])) 
    print("\t| Running this script will train a video classification model.")
    print("\t| Using keras front-end with tensorflow backend")
    usage(with_help=False)

def train_regression_model(X, Y):
    X_train, X_test, Y_train, Y_test = split_data_with_heart_rate(X, Y, 0.2)
    model = Classifier('lrcn').lrcn()
    model.load_weights('3D_CNN_Generator_Weights.h5')
    model.layers.pop()
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
   

    csv_logger = CSVLogger('training.log') 

    generator = frame_generator(BATCH_SIZE, X_train, Y_train)
    val_gen = frame_generator(BATCH_SIZE, X_test, Y_test)
    model.fit_generator(generator=generator,
                       steps_per_epoch=len(X_train)//BATCH_SIZE,
                       epochs=EPOCH,
                       verbose=1,
                       callbacks=[csv_logger],
                       validation_data=val_gen,
                       validation_steps=len(X_test)//BATCH_SIZE, workers=4)
  
    model.save_weights('regression_weights.h5')

def test_regression_model(X, Y):
    X_train, X_test, Y_train, Y_test = split_data_with_heart_rate(X, Y, 0.2)
    model = Classifier('lrcn').lrcn()
    model.layers.pop()
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.load_weights('regression_weights.h5')
    
    #for i in range(len(X)):
    #    s = create_image_array(X[i])
    #    s = np.expand_dims(s, 0)
    #    pred = model.predict(s)
    #    print("{}: actual {}, pred {}".format(i, Y[i], pred))
    #    
    batch = []
    
    sample_x, sample_y1 = X[98], Y[98]
    
    seq = create_image_array(sample_x)
    batch.append(seq)
    #print(sample_x)

    batch = np.array(batch)
    print(batch.shape)
    pred = model.predict(batch)
    #print(sample_x)
    print("actual:", sample_y1)
    print("predicted:", pred)
def train_model(X, Y):
    model = Classifier('lrcn').model
    X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2)
    
    generator = frame_generator(BATCH_SIZE, X_train, Y_train)
    val_gen = frame_generator(BATCH_SIZE, X_test, Y_test)
    model.fit_generator(generator=generator,
                       steps_per_epoch=len(X_train)//BATCH_SIZE,
                       epochs=EPOCH,
                       verbose=1,
                       validation_data=val_gen,
                       validation_steps=len(X_test)//BATCH_SIZE, workers=4)
  
    model.save_weights('3D_CNN_Generator_Weights.h5')

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
    test_regression_model(X, Y)
