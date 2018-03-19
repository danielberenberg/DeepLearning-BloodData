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
    
    def __init__(self):
        self.seq_length    = SEQ_LEN
        self.NCLASSES      = NCLASS
        self.feature_queue = deque()

        metrics = ['accuracy']
        self.input_shape = (SEQ_LEN, INPUT_H, INPUT_W, INPUT_C)

        self.model = self.lrcn()
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())

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


