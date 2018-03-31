import numpy as np
from sklearn.utils import shuffle
import threading

from data_utils import package_data, create_image_array

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

@threadsafe_generator
def frame_generator(batch_size, data, labels):
    print("Greating a frame generator")
    while 1:
        shuffled_x, shuffled_y = shuffle(data, labels)
        batch_x, batch_y = [], []
        vid_batch = shuffled_x[:batch_size]
        for vid in vid_batch:
            batch_x.append(create_image_array(vid))
        batch_y = shuffled_y[:batch_size]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield batch_x, batch_y
