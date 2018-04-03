"""
A module for processing frames as a sequence
"""

import threading 
import os
import random
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import random_rotation, random_shift, random_shear 
from keras.preprocessing.image import random_zoom, flip_axis

class threadsafe_iterator:
    """
    A class for threadsafe iteratio
    """
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


class FrameProcessor:
    """
    the one stop shop object for data frame sequence augmentation 
    """

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 batch_size=4):

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        
        self.batch_size = batch_size

    @threadsafe_generator
    def frame_generator(self, paths2labels, generator_type="frame"):
        """
        a generator for serving up batches of frame sequences 
        """
        print("[*] creating a %s generator with %d samples" % (generator_type, len(paths2labels))
    
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
    
    """
    #@staticmethod
    def get_sample_frames(self, sample):
        """
        #return the sorted list of absolute image paths
        #for this sample
        """
    
        contents = os.listdir(sample)
        max_frame = len(contents)
        
        filenames = [os.path.join(sample,"frame%d.png" % i) for i in range(max_frame)]
    
        return filenames
    
    def rescale(self, frames, sequence_length):
        """
        #return the rescaled list of images in case theres an extra img or so
        """
    
        skip = len(frames) // sequence_length
        output = [frames[i] for i in range(0,len(frames),skip)]
    
        return output[:sequence_length]
    
    def build_image_sequence(self, frames,input_shape=(224,224,3)):
        """
        #return a list of images from filenames
        """
        return [self.process_img(frame,input_shape) for frame in frames]
    
    def process_img(self,frame,input_shape):
        h_, w_, _ = input_shape
        image = load_img(frame,target_size=(h_,w_))
        img_arr = img_to_array(image)

        x = (img_arr/255.).astype(np.float32)

        return x
    """
