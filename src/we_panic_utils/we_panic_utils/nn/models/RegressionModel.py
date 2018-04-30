from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.layers.merge import add
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from collections import deque
import sys
from keras.models import Model
from .residual import residual_block

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

class dumb(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(dumb, self).instantiate()
    
    def get_model(self):
        model = Sequential()
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        model.add(Dense(self.output_shape, activation='linear'))
        return model

class C3D(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(C3D, self).instantiate()

    def get_model(self):
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))

        return model


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
        model.add(LSTM(512, return_sequences=False, dropout=0.5)) 
        model.add(Dense(512, activation='relu'))
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
        
        model.add(Conv3D(64, kernel_size=(3, 3, 3), 
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Dropout(0.5)) 
        model.add(Conv3D(128, kernel_size=(3, 3, 3), 
                  activation='relu')) 
        model.add(MaxPooling3D(pool_size=2, strides=2))
        model.add(Dropout(0.5))
        model.add(Conv3D(128, kernel_size=(3, 3, 3), 
                  activation='relu')) 
        model.add(Conv3D(256, kernel_size=(3, 3, 3), 
                  activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2)) 
        model.add(Conv3D(256, kernel_size=(3, 3, 3), 
                 activation='relu')) 
        model.add(MaxPooling3D(pool_size=2, strides=2))
       
        model.add(Flatten()) 
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))
        return model


class CNN_3D_small(RegressionModel):
   
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(CNN_3D_small, self).instantiate()
    
    def get_model(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(Conv3D(32, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Conv3D(64, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Conv3D(128, kernel_size=(3, 3, 3),
                  activation='relu'))  
        model.add(MaxPooling3D(pool_size=2, strides=2)) 
        model.add(Conv3D(128, kernel_size=(3, 3, 3),
                  activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2)) 
        # batch norm??

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))
        return model



class ResidualLSTM(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(ResidualLSTM, self).instantiate()

    def get_model(self):
        
        # model = Sequential()
        inputs = Input(shape=self.input_shape)
        conv1 = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'))(inputs)

        conv2 = TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation="relu"))(conv1)
        max_pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv2)
        fltn1 = TimeDistributed(Flatten())(max_pool1)
        lstm1 = LSTM(32, dropout=0.5, return_sequences=True)(fltn1)
        resblock1 = TimeDistributed(residual_block(64, 4))(lstm1)
        
        fltn2 = TimeDistributed(Flatten())(resblock1)
        lstm2 = LSTM(64, dropout=0.5, return_sequences=True)(fltn2)
        lstm2 = add([lstm1, lstm2])
        resblock2 = TimeDistributed(residual_block(128, 6))(lstm2)
        
        fltn3 = TimeDistributed(Flatten())(resblock2)
        lstm3 = LSTM(128, dropout=0.5, return_sequences=True)(fltn3)
        lstm3 = add([lstm2, lstm3])
        resblock3 = TimeDistributed(residual_block(256, 8))(lstm3)
        
        conv3 = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(resblock3)
        conv4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(conv3)

        max_pool2 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv4)
        fltn4 = TimeDistributed(Flatten())(max_pool2)
        lstm4 = LSTM(256, dropout=0.5, return_sequences=False)(fltn4)
        
        dense = Dense(self.output_shape, activation='linear')

        model = Model(inputs=Input(shape=self.input_shape), outputs=dense)

        return model
