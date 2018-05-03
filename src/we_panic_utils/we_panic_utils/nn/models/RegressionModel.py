from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input, Reshape, ConvLSTM2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.merge import add
from keras.models import Sequential  # load_model
from keras.optimizers import Adam  # RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Model
from .residual import residualLSTMblock
import keras_resnet.models
import keras

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



class CNN_Stacked_GRU(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(CNN_Stacked_GRU, self).instantiate()

    def get_model(self):
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32,(7,7),strides=(2,2),activation='relu',padding='same'), input_shape=self.input_shape))
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
        model.add(GRU(400, return_sequences=True, dropout=0.5)) 
        model.add(GRU(400, return_sequences=False, dropout=0.5)) 
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
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
        model.add(TimeDistributed(Conv2D(32,(7,7),strides=(2,2),activation='relu',padding='same'), input_shape=self.input_shape))
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



class ResidualLSTM_v00(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)

    def instantiate(self):
        return super(ResidualLSTM_v00, self).instantiate()

    def make_block(self, x, features, blocks, blk_fun=keras_resnet.blocks.time_distributed_bottleneck_2d):
        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x  = blk_fun(features, stage_id, block_id)(x)
            
            features *= 2
         
        #x = time_distributed_bottleneck_1d(32)(x) 
        return x
 
    def get_model(self):
        
        block = keras_resnet.blocks.time_distributed_bottleneck_2d
        
        conv1 = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'))(Input(shape=self.input_shape))  
        conv2 = TimeDistributed(Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu'))(conv1)         

        blocks1 = [2]
        
        resblock1 = self.make_block(conv2, 64, blocks1)
        
        #pool = TimeDistributed(MaxPooling2D((2,2),strides=(2,2)))(resblock1)
        print(resblock1)
        x_rnn = LSTM(64, dropout=0.5, return_sequences=False)(resblock1)
        
        model = Model(outputs=x_rnn)
        

class ResidualLSTM_v01(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)
        self.rnn_depth = 6
        self.rnn_width = 128
        self.rnn_kernel = (3, 3)

    def instantiate(self):
        return super(ResidualLSTM_v01, self).instantiate()

    def get_model(self):
        
        inputs = Input(self.input_shape)

        conv1 = TimeDistributed(Conv2D(256, (7, 7),
                                strides=(1, 1),
                                activation='tanh',
                                padding='same',
                                kernel_initializer='he_normal'))(inputs)
    
        conv2 = TimeDistributed(Conv2D(128, (3, 3),
                                #padding='same',
                                kernel_initializer='he_normal',
                                activation="relu"))(conv1)
        
        conv3 = TimeDistributed(Conv2D(64, (3, 3),
                                #padding='same',
                                kernel_initializer='he_normal',
                                activation='relu'))(conv2)

        pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv3)

        x_rnn = ConvLSTM2D(32,
                           (3,3),
                           recurrent_dropout=0.5,
                           dropout=0.5,
                           padding='same',
                           return_sequences=True)(pool1)
        
        x_rnn1 = ConvLSTM2D(32,
                            (3,3),
                            recurrent_dropout=0.5,
                            dropout=0.5,
                            padding='same',
                            return_sequences=True)(x_rnn)

        x_rnn = add([x_rnn, x_rnn1])
                            
        x_rnn1 = ConvLSTM2D(32,
                            (3,3),
                            recurrent_dropout=0.5,
                            dropout=0.5,
                            padding='same',
                            return_sequences=True)(x_rnn)

        x_rnn = add([x_rnn, x_rnn1])
        
        final_lstm = ConvLSTM2D(32,
                                (3,3), 
                                recurrent_dropout=0.5,
                                dropout=0.5,
                                padding='same',
                                return_sequences=False)(x_rnn)
        
        
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(final_lstm)
 
        fltn = TimeDistributed(Flatten())(pool1)
        dense1 = Dense(256, activation='relu')(fltn)
        fltn2 = Flatten()(dense1)
        #dense2 = Dense(256, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(fltn2)
        #outputs = Reshape(list(outputs.shape) + [1])(outputs)
        model = Model(inputs=inputs, outputs=outputs)

        return model

class ResidualLSTM_v02(RegressionModel):
    def __init__(self, input_shape, output_shape):
        RegressionModel.__init__(self, input_shape, output_shape)
        self.rnn_depth = 6
        self.rnn_width = 128
        self.rnn_kernel = (3, 3)

    def instantiate(self):
        return super(ResidualLSTM_v02, self).instantiate()
    
        inputs = Input(self.input_shape) 
        
        x_rnn = ConvLSTM2D(32,
                           (7,7),
                           recurrent_dropout=0.5,
                           dropout=0.5,
                           padding='same',
                           return_sequences=True)(pool1)
        
        x_rnn1 = ConvLSTM2D(64,
                            (5,5),
                            recurrent_dropout=0.5,
                            dropout=0.5,
                            padding='same',
                            return_sequences=True)(x_rnn)

        x_rnn = add([x_rnn, x_rnn1])
                            
        x_rnn1 = ConvLSTM2D(128,
                            (3,3),
                            recurrent_dropout=0.5,
                            dropout=0.5,
                            padding='same',
                            return_sequences=True)(x_rnn)

        x_rnn = add([x_rnn, x_rnn1])
        
        fltn = TimeDistributed(Flatten())(x_rnn)
        #fltn = Flatten()(x_rnn)
        drp = Dropout(0.5)(fltn)
        final_lstm = LSTM(256, recurrent_dropout=0.5, dropout=0.5, padding='same', return_sequences=False)(drp)
        dense = Dense(512, activation='relu')(final_lstm)
        drp2 = Dropout(0.5)(dense)

        outputs = Dense(self.output_shape, activation='linear')(drp2)

        model = Model(inputs=inputs, outputs=outputs)
        return model
