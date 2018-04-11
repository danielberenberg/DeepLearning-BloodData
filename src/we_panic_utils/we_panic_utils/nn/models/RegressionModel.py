from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from collections import deque
import sys


class RegressionModel():
    
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def instantiate(self):
        model = self.get_model()
        optimizer = Adam(lr=1e-5, decay=1e-6)
        metrics = ['mse']
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=metrics)
        print(model.summary())
        return model

    def get_model(self):
        raise NotImplementedError 

class CNN_LSTM(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(CNN_LSTM, self).instantiate()

    def get_model(self):
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
        
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5)) 
        model.add(Dropout(0.5)) 
        model.add(Dense(512, activation='relu')) 
        model.add(Dropout(0.5)) 
        model.add(Dense(self.output_shape, activation='linear'))

        return model

class CNN_3D(RegressionModel):
   
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(CNN_3D, self).instantiate()
    
    def get_model(self):
        model = Sequential()
        
        model.add(Conv3D(64, kernel_size=(3,3,3), 
               input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Dropout(0.5)) 
        model.add(Conv3D(128, kernel_size=(3,3,3), 
                activation='relu')) 
        model.add(MaxPooling3D(pool_size=2, strides=2))
        model.add(Dropout(0.5))
        model.add(Conv3D(128, kernel_size=(3,3,3), 
                activation='relu')) 
        model.add(Conv3D(256, kernel_size=(3,3,3), 
                activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2)) 
        model.add(Conv3D(256, kernel_size=(3,3,3), 
                activation='relu')) 
        model.add(MaxPooling3D(pool_size=2, strides=2))
        
        model.add(Flatten()) 
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))
        return model
