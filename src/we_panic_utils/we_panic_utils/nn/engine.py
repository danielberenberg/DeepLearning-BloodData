from .data_load import train_test_split_with_csv_support, data_set_to_csv, data_set_from_csv
from .models import CNN_LSTM, CNN_3D
from .processing import *


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
        input_shape - shape of the sequence passed, 60 separate 224x224x3 frames
        output_shape - the number of outputs
    """
    def __init__(self, 
                 regular_data, 
                 augmented_data, 
                 model_type, 
                 filtered_csv, 
                 partition_csv, 
                 batch_size, 
                 epochs, 
                 train, 
                 test, 
                 inputs, 
                 outputs, 
                 ignore_augmented, 
                 input_shape=(60, 224, 224, 3), 
                 output_shape=2):

        self.regular_data = regular_data
        self.augmented_data = augmented_data
        self.model_type = model_type
        self.filtered_csv = filtered_csv
        self.partition_csv = partition_csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.train = train
        self.test = test
        self.inputs = inputs
        self.outputs = outputs
        self.ignore_augmented = ignore_augmented 
        self.input_shape = input_shape
        self.output_shape = output_shape


    def run(self):
        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        model = self.__choose_model().instantiate()   
        
        train_set = test_set = val_set = None

        if self.train:
            train_set, test_set, val_set = train_test_split_with_csv_support(self.regular_data,
                                                                             self.filtered_csv, 
                                                                             self.partition_csv, 
                                                                             self.outputs, 
                                                                             augmented_data_path=self.augmented_data,
                                                                             ignore_augmented=self.ignore_augmented)

            print("This is where the model would be trained")
        
        if self.test:

            if not test_set:
                print("This is where data would be loaded")
            else:
                print("This is where we train with the existing data set")

    def __choose_model(self):
        """
        choose a model based on preferences
        """
        if self.model_type == "CNN+LSTM":
            return CNN_LSTM(self.input_shape, self.output_shape)

        if self.model_type == "CNN-3D":
            return CNN_3D(self.input_shape, self.output_shape)

        raise ValueError("Model type does not exist: {}".format(self.model_type))


