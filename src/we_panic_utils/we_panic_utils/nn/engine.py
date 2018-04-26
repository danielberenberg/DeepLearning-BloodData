from .data_load import train_test_split_with_csv_support, ttswcsv2, ttswcvs3, data_set_to_csv, data_set_from_csv
from .models import C3D, CNN_LSTM, CNN_3D, CNN_3D_small, dumb
from .processing import FrameProcessor
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras import backend as K
import os
import pandas as pd

class Engine():
    """
    The engine for training/testing a model

    args:
        regular_data - data not generated
        augmented_data - data generated using halve/doubling speed augmentation
        filtered_csv - the csv containing all of the stats for every sample
        partition_csv - the csv that maps individual partititions to their respective labels
        batch_size - the batch size
        epochs - number of epochs to train
        train - boolean stating whether or not to train
        test - boolean stating whether or not to test
        frameproc - FrameProcessor object for augmentation
        ignore_augmented - list containing phases of running the model in which to ignore augmented data
        input_shape - shape of the sequence passed, 60 separate 100x100x3 frames
        output_shape - the number of outputs
    """
    def __init__(self, 
                 data,
                 #regular_data, 
                 #augmented_data, 
                 model_type, 
                 filtered_csv, 
                 batch_size, 
                 epochs, 
                 train, 
                 load,
                 test, 
                 inputs, 
                 outputs,
                 frameproc,
                 ignore_augmented=[""], 
                 input_shape=(60, 100, 100, 3), 
                 output_shape=1):

        #self.regular_data = regular_data
        #self.augmented_data = augmented_data
        self.data = data
        self.model_type = model_type
        self.metadata = filtered_csv
        # self.partition_csv = partition_csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.train = train
        self.load = load
        self.test = test
        self.inputs = inputs
        self.outputs = outputs
        self.ignore_augmented = ignore_augmented 
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.processor = frameproc
    
    def run2(self):

        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        model = self.__choose_model().instantiate()   
        train_set = test_set = val_set = None

        if self.train and not self.load:
            print("Training the model.")
            train_set, test_set, val_set = ttswcvs3(self.data, self.metadata, self.outputs)
            
            train_generator = self.processor.train_generator_v3(train_set)
            val_generator = self.processor.testing_generator_v3(val_set)

            csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                           verbose=1, 
                                           save_best_only=True)
            test_results_file = os.path.join(self.outputs, "test_results.log")
            test_callback = TestResultsCallback(self.processor.testing_generator_v3(test_set), test_set, test_results_file, self.batch_size)
    
            model.fit_generator(generator=train_generator,
                                steps_per_epoch=300,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=[csv_logger, checkpointer, test_callback],
                                validation_data=val_generator,
                                validation_steps=len(val_set), workers=4)
        
        if self.train and self.load:
            print("Resuming training of existing model.")
            model.load_weights(os.path.join(self.inputs, "models", self.model_type + ".h5"))
            
            print(K.eval(model.optimizer.lr))            

            train_set = pd.read_csv(os.path.join(self.inputs, "train.csv"))
            test_set = pd.read_csv(os.path.join(self.inputs, "test.csv"))
            val_set = pd.read_csv(os.path.join(self.inputs, "val.csv"))

            train_generator = self.processor.train_generator_v3(train_set)
            val_generator = self.processor.testing_generator_v3(val_set)
            
            csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                           verbose=1, 
                                           save_best_only=True)
            
            model.fit_generator(generator=train_generator,
                                steps_per_epoch=300,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=[csv_logger, checkpointer],
                                validation_data=val_generator,
                                validation_steps=len(val_set), workers=4)

        if self.test:

            # if the test set doesn't exist yet, it means we are testing without training
            if test_set is None:
                print("Testing model without training.")
                model_dir = os.path.join(self.inputs, "models")
                model_path = "" 
                for path in os.listdir(model_dir):
                    if self.model_type in path and path.endswith(".h5"):
                        model_path = os.path.join(model_dir, path)
                        break
                if model_path == "":
                    raise FileNotFoundError("Could not locate model file in {}-- have you trained the model yet?".format(model_dir))
                
                #print("Loading model from file: {}".format(model_path))
                #model.load_weights(model_path)
                
                model = models.load_model(model_path)
                
                test_dir = os.path.join(self.inputs, "test.csv")
                
                test_set = pd.read_csv(test_dir)

                test_generator = self.processor.testing_generator_v3(test_set)
                loss = model.evaluate_generator(test_generator, len(test_set))
                
                pred = model.predict_generator(test_generator, len(test_set))

                with open(os.path.join(self.outputs, "test.log"), 'w') as log:
                    log.write(str(loss[0]) + "," + str(loss[1])) 

                print(loss)
                print(pred) 

            # otherwise, we can use the existing test set that was generated during the training phase
            else:
                print("Testing model after training.")
                test_generator = self.processor.testing_generator_v3(test_set)
                loss = model.evaluate_generator(test_generator, len(test_set))
                pred = model.predict_generator(test_generator, len(test_set))
                
                with open(os.path.join(self.outputs, "test.log"), 'w') as log:
                    log.write(str(loss[0]) + "," + str(loss[1])) 

                print(loss)
                print(pred) 

    
    def __choose_model(self):
        """
        choose a model based on preferences
        """
        if self.model_type == "C3D": 
            return C3D(self.input_shape, self.output_shape)
        
        if self.model_type == "CNN+LSTM":
            return CNN_LSTM(self.input_shape, self.output_shape)

        if self.model_type == "3D-CNN":
            return CNN_3D(self.input_shape, self.output_shape)
        
        if self.model_type == "CNN_3D_small":
            return CNN_3D_small(self.input_shape, self.output_shape)

        if self.model_type == "dumb":
            return dumb(self.input_shape, self.output_shape)
        
        raise ValueError("Model type does not exist: {}".format(self.model_type))

class TestResultsCallback(Callback):
    def __init__(self, test_gen, test_set, log_file, batch_size):
        self.test_gen = test_gen
        self.test_set = test_set
        self.log_file = log_file
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % 5 == 0:
            print('Logging tests at epoch', epoch)
            with open(self.log_file, 'a') as log:
                pred = self.model.predict_generator(self.test_gen, len(self.test_set))
                subjects = list(self.test_set['Subject'])
                trial = list(self.test_set['Trial'])
                hr = list(self.test_set['Heart Rate'])
                i = 0
                s = 0
                log.write("Epoch: " + str(epoch+1) + '\n')
                for p in pred:
                    subj = subjects[s]
                    tri = trial[s]
                    h = hr[s]
                    val = p[0]
                    log.write(str(subj) + ', ' + str(tri) + '| prediction=' + str(p) + ', actual=' + str(h) + '\n')
                    i+=1
                    if i % self.batch_size == 0:
                        s += 1
                    if s == len(subjects):
                        s = 0

#    def run(self):
#        """
#        a general method that computes the 'procedure' to follow based on the
#        preferences passed in the constructor and runs that procedure
#        """
#        model = self.__choose_model().instantiate()   
#        train_set = test_set = val_set = None
#
#        if self.train:
#            print("Training the model.")
#            # train_set, test_set, val_set = train_test_split_with_csv_support(self.regular_data,
#            #                                                                 self.filtered_csv,
#
#            #                                                                 self.partition_csv, 
#            #                                                                 self.outputs, 
#            #                                                                 augmented_data_path=self.augmented_data,
#            #                                                                 ignore_augmented=self.ignore_augmented)
#            
#            ignore_augmented = {'train': 'train_aug.csv', 'val': 'val_aug.csv', 'test': 'test_aug.csv'}
#            
#            for ign in self.ignore_augmented:
#                if ign in ignore_augmented:
#                    ignore_augmented.pop(ign)
#
#            train_reg, test_reg, val_reg = ttswcsv2(self.regular_data, self.metadata, self.outputs) 
#            train_aug, test_aug, val_aug = ttswcsv2(self.augmented_data, self.metadata, self.outputs)
#
#            regs = [train_reg, test_reg, val_reg]
#            augs = [train_aug, test_aug, val_aug]
#            sets = []
#
#            for reg, aug in zip(regs, augs):
#                if aug is not None:
#                    s = reg.copy()
#                    s.update(aug)
#
#                    sets.append(s)
#        
#            train_set, test_set, val_set = sets
#
#            train_generator = self.processor.train_generator(train_set)
#            val_generator = self.processor.testing_generator(val_set, "validation")
#
#            csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
#            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
#                                           verbose=1, 
#                                           save_best_only=True, 
#                                           save_weights_only=True)
#
#            model.fit_generator(generator=train_generator,
#                                steps_per_epoch=len(train_set) // self.processor.batch_size,
#                                epochs=self.epochs,
#                                verbose=1,
#                                callbacks=[csv_logger, checkpointer],
#                                validation_data=val_generator,
#                                validation_steps=len(val_set), workers=4)
#        
#        if self.test:
#
#            # if the test set doesn't exist yet, it means we are testing without training
#            if not test_set:
#                print("Testing model without training.")
#                model_dir = os.path.join(self.inputs, "models")
#                model_path = "" 
#                for path in os.listdir(model_dir):
#                    if self.model_type in path and path.endswith(".h5"):
#                        model_path = os.path.join(model_dir, path)
#                        break
#                if model_path == "":
#                    raise FileNotFoundError("Could not locate model file in {}-- have you trained the model yet?".format(model_dir))
#                
#                print("Loading model from file: {}".format(model_path))
#                model.load_weights(model_path)
#                
#                test_dir = os.path.join(self.inputs, "test.csv")
#                ignore = None
#                if "test" in self.ignore_augmented:
#                    ignore = self.augmented_data
#                test_set = data_set_from_csv(test_dir, ignore)
#
#                test_generator = self.processor.testing_generator_v2(test_set)
#                loss = model.evaluate_generator(test_generator, len(test_set))
#                
#                pred = model.predict_generator(test_generator, len(test_set))
#                print(loss)
#                print(pred) 
#                 # print(loss)
#
#            # otherwise, we can use the existing test set that was generated during the training phase
#            else:
#                print("Testing model after training.")
#                test_generator = self.processor.testing_generato_v2r(test_set)
#                loss = model.evaluate_generator(test_generator, len(test_set))
#                pred = model.predict_generator(test_generator, len(test_set))
#                
#                with open(os.path.join(self.outputs, "test.log"), 'w') as log:
#                    log.write(str(loss[0]) + "," + str(loss[1])) 
#
#                print(loss)
#                print(pred) 
#


