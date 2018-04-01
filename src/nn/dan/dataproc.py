import classifier
import threading
import os
import random
import random
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

def train_test_split(data_dir):
    contents = sorted(os.listdir(data_dir))
    
    train, test = [], []

    for i,c in enumerate(contents):
        if i%5 == 0:
            test.append(os.path.join(data_dir,c))
        else:
            train.append(os.path.join(data_dir,c))

    return train, test


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """ decorator """

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


class DataProcessor():
    
    def __init__(self):
        pass

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_dir, data_dict):
        train, test = train_test_split(data_dir) 
        data = train if train_test == "train" else test
    
        print ("[*] creating a %s generator with %d samples" % (train_test, len(data)))
    
        while 1:
            X, y = [], []
    
            for _ in range(batch_size):
                # reset to be safe 
                sequence = None
                
                samp = random.choice(data)
                sample = samp.split("/")[1]

                #print(data_dict)
                #print(data_dict[sample])
                label = data_dict[sample] 
                
                frames = self.get_sample_frames(samp)
                frames = self.rescale(frames, classifier.SEQ_LEN)
                
                sequence = self.build_image_sequence(frames)
    
    
                X.append(sequence)
                
                y.append(self.one_hot(label))
    
            yield np.array(X), np.array(y)
    
    #@staticmethod
    def get_sample_frames(self, sample):
        """
        return the sorted list of absolute image paths
        for this sample
        """
    
        contents = os.listdir(sample)
        max_frame = len(contents)
        
        filenames = [os.path.join(sample,"frame%d.png" % i) for i in range(max_frame)]
    
        return filenames
    
    def rescale(self, frames, sequence_length):
        """
        return the rescaled list of images in case theres an extra img or so
        """
    
        skip = len(frames) // sequence_length
        output = [frames[i] for i in range(0,len(frames),skip)]
    
        return output[:sequence_length]
    
    def build_image_sequence(self, frames,input_shape=(224,224,3)):
        """
        return a list of images from filenames
        """
        return [self.process_img(frame,input_shape) for frame in frames]
    
    def process_img(self,frame,input_shape):
        h_, w_, _ = input_shape
        image = load_img(frame,target_size=(h_,w_))
        img_arr = img_to_array(image)

        x = (img_arr/255.).astype(np.float32)

        return x

    def one_hot(self, label):
        label_encoded = classifier.CLASSES.index(label)
        label_hot = to_categorical(label_encoded, len(classifier.CLASSES))
    
        return label_hot
    
