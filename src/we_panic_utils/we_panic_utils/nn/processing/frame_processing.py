"""
A module for processing frames as a sequence
"""

import threading 
import os
import random
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center

from PIL import ImageEnhance
from PIL import Image as pil_image


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
    the one stop shop object for data frame sequence augmentation and generation 
    
    usage example:
        train_proc = FrameProcessor(rotation_range=0.5, zoom_range=0.2)
        train_gen  = train_proc.frame_generator(filtered_training_paths, 'train')

    args:
        rotation_range : int - degree range for random rotations
        width_shift_Range : float - fraction of total width for horizontal shifts
        height_shift_range : float - fraction of total height for vertical shifts
        zoom_range : float - range for random zoom, lower_bd = 1 - zoom_range, upper_bd = 1 + zoom_range 
        vertical_flip : bool - whether or not to flip vertically with prob 0.5
        horizontal_flip : bool - whether or not to flip horizontall with prob 0.5
        batch_size : int - the batch size
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
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
        
        assert type(self.rotation_range) == int, "rotation_range should be integer valued"

        assert type(self.width_shift_range) == float, "width_shift_range should be a float"
        assert self.width_shift_range <= 1. and self.width_shift_range >= 0., "width_shift_range should be in [0, 1], got %f" % self.width_shift_range 
        
        assert type(self.height_shift_range) == float, "height_shift_range should be a float"
        assert self.height_shift_range <= 1. and self.height_shift_range >= 0., "width_shift_range should be in [0, 1], got %f" % self.height_shift_range

        assert type(self.zoom_range) == float, "zoom_range should be a float"
        
        assert type(horizontal_flip) == bool, "horizontal_flip should be a boolean"
        assert type(vertical_flip) == bool, "vertical_flip should be a boolean"

    @threadsafe_generator
    def frame_generator(self, paths2labels, generator_type):
        """
        a generator for serving up batches of frame sequences 
        
        args:
            paths2labels : dictionary - map from image sequence path names to tuples (heart rate, resp rate)
            generator_type : just a little title string to insert into the below format statement
        """

        print("[*] creating a %s generator with %d samples" % (generator_type, len(paths2labels)))
        
        sequence_paths = [pth for pth in paths2labels]

        while 1:
            X, y = [], []
            selected_paths = random.sample(sequence_paths, self.batch_size)
            
            y = [paths2labels[pth] for pth in selected_paths]
            
            for pth in selected_paths:
               
                frames = self.get_sample_frames(pth)
                sequence = self.build_image_sequence(frames)
                
                # now we want to apply the augmentation
                if self.rotation_range > 0.0:
                    sequence = self.random_sequence_rotation(sequence)

                if self.width_shift_range > 0.0 or self.height_shift_range > 0.0:
                    sequence = self.random_sequence_shift(sequence)
                
                if self.shear_range > 0.0:
                    sequence = self.random_sequence_shear(sequence)

                if self.zoom_range > 0.0:
                    sequence = self.random_sequence_zoom(sequence)
                
                if self.vertical_flip:
                    # with probability 0.5, flip vertical axis
                    coin_flip = np.random.random_sample() > 0.5
                    if coin_flip:
                        sequence = self.sequence_flip_axis(sequence, 0)   # flip on the row axis
                
                if self.horizontal_flip:
                    # with probability 0.5, flip horizontal axis (cols)
                    coin_flip - np.random.random_sample() > 0.5

                    if coin_flip:
                        sequence = self.sequence_flip_axis(sequence, 1)   # flip on the column axis
                
                X.append(sequence)

            yield np.array(X), np.array(y)
   
    def random_sequence_rotation(self, seq, row_axis=0, col_axis=1, channel_axis=2,
                                 fill_mode='nearest', cval=0):
        """
        apply a rotation to an entire sequence of frames
        
        args:
            seq : list (or array-like) - list of 3D input tensors
            row_axis : int - index of row axis on each tensor
            col_axis : int - index of cols axis on each tensor
            channel_axis : int - index of channel ax on each tensor
            fill_mode : string - points outside the boundaries of input are filled
                                 according to the given input mode, one of 
                                 {'nearest', 'constant', 'reflect', 'wrap'}

            cval : float - constant value used for fill_mode constant
        
        returns:
            rotated : the rotated sequence of tensors
        """

        theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]])

        h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
        transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w) 
        
        return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq] 
            
    def random_sequence_shift(self, seq, row_axis=0, col_axis=1, channel_axis=2, 
                              fill_mode="nearest", cval=0):
        """
        apply a height/width shift to an entire sequence of frames
        
        args:
            seq : list - the list of 3D input tensors
            row_axis : int - the index of row axis on each tensor
            col_axis : int - the index of col axis on each tensor
            channel_axis : int - the index of channel ax on each tensor
            fill_mode : string - points outside the boundaries of input are filled
                                 according to the given input mode, one of 
                                 {'nearest', 'constant', 'reflect', 'wrap'}

            cval : float - the constant value used for fill_mode constant 
        """

        h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
        tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * h
        ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * w

        translation_matrix = np.array([[1, 0, tx],
                                      [0, 1, ty],
                                      [0, 0, 1]])

        transform_matrix = translation_matrix  # no offset necessary
        
        return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]

    def random_sequence_shear(self, seq, row_axis=0, col_axis=1, channel_axis=2,
                              fill_mode='nearest', cval=0):
        """
        apply a random shear to an entire sequence of frames

        args:
            seq : list - the list of 3D input tensors
            row_axis : int - the index of row axis on each tensor
            col_axis : int - the index of col axis on each tensor
            channel_axis : int - the index of channel ax on each tensor
            fill_mode : string - points outside the boundaries of input are filled
                                 according to the given input mode, one of 
                                 {'nearest', 'constant', 'reflect', 'wrap'}

            cval : float - the constant value used for fill_mode constant 
        
        returns:
            the sequence of sheared frames
        """

        shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
        transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
        
        return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]

    def random_sequence_zoom(self, seq, row_axis=0, col_axis=1, channel_axis=2,
                             fill_mode='nearest', cval=0): 
        """
        apply a random zoom on an entire sequence of frames

        args:
            seq : list - the list of 3D input tensors
            row_axis : int - the index of row axis on each tensor
            col_axis : int - the index of col axis on each tensor
            channel_axis : int - the index of channel ax on each tensor
            fill_mode : string - points outside the boundaries of input are filled
                                 according to the given input mode, one of 
                                 {'nearest', 'constant', 'reflect', 'wrap'}

            cval : float - the constant value used for fill_mode constant 
        
        returns:
            the sequence of zoomed frames
        """

        zlower, zupper = 1 - self.zoom_range, 1 + self.zoom_range
        
        if zlower == 1 and zupper == 1:
            zx, zy = 1, 1
        
        else:
            zx, zy = np.random.uniform(zlower, zupper, 2)

        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]

        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)

        return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]

    def sequence_flip_axis(self, seq, axis):
        
        """
        flip a sequence of images to a different axis (vertical, horizontal) 
        
        args:
            seq : list - the list of 3D image tensors
            axis : axis on which to rotate

        returns
            rotated image sequence
        """

        seq = [np.asarray(x).swapaxes(axis, 0) for x in seq]
        seq = [x[::-1, ...] for x in seq]
        seq = [x.swapaxes(0, axis) for x in seq]

        return seq

    def get_sample_frames(self, sample):
        """
        return the sorted list of absolute image paths for this sample
        """

        contents = os.listdir(sample)
        max_frame = len(contents)
        
        filenames = [os.path.join(sample, "frame%d.png" % i) for i in range(max_frame)]
    
        return filenames
      
    def build_image_sequence(self, frames, input_shape=(224, 224, 3)):
        """
        return a list of images from filenames
        """
        return [self.process_img(frame, input_shape) for frame in frames]
    
    def process_img(self, frame, input_shape):
        """
        load up an image as a numpy array

        args:
            frame : str - image path
            input_shape : tuple (h, w, nchannels)

        returns
            x : the loaded image
        """
        h_, w_, _ = input_shape
        image = load_img(frame, target_size=(h_, w_))
        img_arr = img_to_array(image)

        x = (img_arr / 255.).astype(np.float32)

        return x
