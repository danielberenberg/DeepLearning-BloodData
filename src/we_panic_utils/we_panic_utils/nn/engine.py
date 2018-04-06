from .data_load import train_test_split_with_csv_support, data_set_to_csv, data_set_from_csv
from .models import CNN_LSTM, CNN_3D
from .processing import FrameProcessor
from keras.callbacks import CSVLogger, ModelCheckpoint
import os


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
                 frameproc,
                 ignore_augmented=[""], 
                 input_shape=(60, 100, 100, 3), 
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
        self.processor = frameproc

    def run(self):
        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        model = self.__choose_model().instantiate()   
        train_set = test_set = val_set = None

        if self.train:
            print("Training the model.")
            train_set, test_set, val_set = train_test_split_with_csv_support(self.regular_data,
                                                                             self.filtered_csv, 
                                                                             self.partition_csv, 
                                                                             self.outputs, 
                                                                             augmented_data_path=self.augmented_data,
                                                                             ignore_augmented=self.ignore_augmented)

            train_generator = self.processor.frame_generator(train_set, "train")
            val_generator = self.processor.testing_generator(val_set, "validation")

            csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                           verbose=1, 
                                           save_best_only=True, 
                                           save_weights_only=True)

            model.fit_generator(generator=train_generator,
                                steps_per_epoch=len(train_set) // self.processor.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=[csv_logger, checkpointer],
                                validation_data=val_generator,
                                validation_steps=len(val_set), workers=4)
        
        if self.test:

            # if the test set doesn't exist yet, it means we are testing without training
            if not test_set:
                print("Testing model without training.")
                model_dir = os.path.join(self.inputs, "models")
                model_path = "" 
                for path in os.listdir(model_dir):
                    if self.model_type in path and path.endswith(".h5"):
                        model_path = path
                        break
                if model_path == "":
                    raise FileNotFoundError("Could not locate model file in {}-- have you trained the model yet?".format(model_dir))
                
                print("Loading model from file: {}".format(model_path))
                model.load_weights(model_path)
                
                test_dir = os.path.join(self.inputs, "test.csv")
                ignore = None
                if "test" in self.ignore_augmented:
                    ignore = self.augmented_data
                test_set = data_set_from_csv(test_dir, ignore)

                test_generator = self.processor.testing_generator(test_set, "test")
                loss = model.evaluate_generator(test_generator, len(test_set))
                # print(loss)

            # otherwise, we can use the existing test set that was generated during the training phase
            else:
                print("Testing model after training.")
                test_generator = self.processor.testing_generator(test_set, "test")
                loss = model.evaluate_generator(test_generator, len(test_set))
                # print(loss) 

    def __choose_model(self):
        """
        choose a model based on preferences
        """
        if self.model_type == "CNN+LSTM":
            return CNN_LSTM(self.input_shape, self.output_shape)

        if self.model_type == "CNN-3D":
            return CNN_3D(self.input_shape, self.output_shape)

        raise ValueError("Model type does not exist: {}".format(self.model_type))


