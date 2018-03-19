"""
Train the video classifier
"""


import sys, os
import numpy as np
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

import classifier
import dataproc
import basic_utils.basics as base


def usage():
    print("[Usage] python %s <data_dir> <labels-csv>" % sys.argv[0])
    print("------- train a video classifier on our pulse data")
    sys.exit()

def parse_input():
    try:
        dir_ = sys.argv[1]
        csv_ = sys.argv[2]

        if os.path.isdir(dir_) and os.path.exists(csv_):
            return dir_, csv_

        else:
            if not os.path.isdir(dir_):
                raise OSError("%s not found" % dir_)
            
            if not os.path.exists(csv_):
                raise OSError("%s not found" % csv_)

    except IndexError:
        usage()

def csv_dict(csv_, idx):
    """
    Convert csv file to dictionary where leading element of each row
    is the key and the value is row[idx]
    """

    mat, header = base.csv2data(csv_)

    d = dict()
    for row in mat:
        #print(row)
        k = row[0]
        v = row[idx]
        d[k] = v
    
    return d


def train(data_dir, data_dict, model_dir, model='lrcn', batch_size=4, epochs=100 ):

    base.check_exists_create_if_not(os.path.join(model_dir,"checkpoints"))

    checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir,"checkpoints",model+'-'+\
                                                        '{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1,
                                                        save_best_only=True)


    base.check_exists_create_if_not('logs')
    tb = TensorBoard(log_dir='logs')

    early_stopper = EarlyStopping(patience=20)
    timestamp = time.time()

    csv_logger = CSVLogger(os.path.join('logs',model + '-' + 'training-' +\
                           str(timestamp) + '.log'))

    proc = dataproc.DataProcessor()
    steps_per_epoch = (len(os.listdir(data_dir))*0.8)//batch_size
    
    generator = proc.frame_generator(batch_size,'train',data_dir,data_dict)
    val_gen = proc.frame_generator(batch_size,'test',data_dir,data_dict)
    
    print("[*] --- initializing/compiling model ---")

    ### model initialization
    clf = classifier.Classifier()

    #### #### #### #### #### #### #### #### ####

    clf.model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[tb, early_stopper, csv_logger, checkpointer],
                            validation_data=val_gen,
                            validation_steps=30,workers=4)

if __name__ == "__main__":
    directory,csv_file  = parse_input()
    dir2data = csv_dict(csv_file, -1) 
    model_dir = os.path.join(os.getcwd(), "models/")
    base.check_exists_create_if_not(model_dir)

    print("---===[*] model directory ---> %s" % model_dir)
    #### #### #### #### #### #### #### #### ####

    #print("[*] --- data generators --- [*]")
    #datagen = ImageDataGenerator(shear_range=0.2,
    #                             zoom_range=0.2,
    #                             width_shift_range=0.2,
    #                             height_shift_range=0.2,
    #                             horizontal_flip=True,
    #                             vertical_flip=True)

    #val_datagen = ImageDataGenerator()


    # train, test = train_test_split(directory)

    #print(len(train))
    #print(len(test))

    train(directory,dir2data,model_dir)
