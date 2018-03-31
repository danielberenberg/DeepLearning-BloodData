from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from collections import deque
import sys

INPUT_H = 224    # height
INPUT_W = 224    # width
INPUT_C = 3      # RGB
NCLASS = 2       # number of classes
SEQ_LEN = 60     # length of each sample
FEAT_LEN = 2048  # number of features to pass

CLASSES = ["LOW","HIGH"]

class Classifier():

    def __init__(self, model_choice):
        self.seq_length    = SEQ_LEN
        self.NCLASSES      = NCLASS
        self.feature_queue = deque()

        metrics = ['accuracy']
        self.input_shape = (SEQ_LEN, INPUT_H, INPUT_W, INPUT_C)
        
        self.model_options = ["lrcn", "cnn_3D"]
        self.model = self.__choose_model(model_choice)
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())
    
    def __choose_model(self, choice):
        if choice not in self.model_options:
            raise ValueError("Error: {} not a valid options".format(choice))
        if choice == self.model_options[0]:
            return self.lrcn()
        elif choice == self.model_options[1]:
            return self.cnn_3D()

    def cnn_3D(self):
        
        model = Sequential()
        
        model.add(Conv3D(32, kernel_size=(3,3,3), 
               input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Dropout(0.5)) 
        model.add(Conv3D(64, kernel_size=(3,3,3), 
                activation='relu')) 
        model.add(MaxPooling3D(pool_size=2, strides=2))
        model.add(Dropout(0.5))
        model.add(Conv3D(128, kernel_size=(3,3,3), 
                activation='relu')) 
        model.add(Conv3D(128, kernel_size=(3,3,3), 
                activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2))
        
        model.add(Flatten())
        model.add(Dense(64, init='normal', activation='relu'))
        model.add(Dense(self.NCLASSES, activation = 'softmax'))
        
        return model
    def lrcn(self):
        """
        Extract features with a CNN, pass them to an RNN

        adapted from:
        article - [https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5]
        source  - [https://github.com/harvitronix/five-video-classification-methods/blob/master/models.py]
        """

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32,(7,7),strides=(2,2),activation='relu',
                  padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32,(3,3),kernel_initializer="he_normal",activation="relu")))
        model.add(TimeDistributed(MaxPooling2D((2,2),strides=(2,2))))

        model.add(TimeDistributed(Conv2D(64,(3,3),padding="same",activation="relu")))
        model.add(TimeDistributed(Conv2D(64,(3,3),padding="same",activation="relu")))
        model.add(TimeDistributed(MaxPooling2D((2,2),strides=(2,2))))

        model.add(TimeDistributed(Conv2D(256,(3,3),padding="same",activation="relu")))
        model.add(TimeDistributed(Conv2D(256,(3,3),padding="same",activation="relu")))
        model.add(TimeDistributed(MaxPooling2D((2,2),strides=(2,2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.NCLASSES, activation='softmax'))

        return model
