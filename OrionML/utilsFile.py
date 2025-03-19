import numpy as np
import math
import copy

class utils():
    def __init__(self):
        return

    def train_test_split(arr, train=1, shuffle=True):
        '''
    
        Parameters
        ----------
        arr : ndarray
            Array which should be split.
        train : float, optional
            Share of arr that should be in the training set. The default is 1.
        shuffle : bool, optional
            If the train and test set should be extracted from shuffeled arr or not. 
            The default is True.
    
        Returns
        -------
        train_arr : ndarray
            Training array.
        test_arr : ndarray
            Test array.
        '''
        arr = copy.copy(arr)
        
        if shuffle==True:
            np.random.shuffle(arr)
                        
        arr_len = len(arr)
        split_pos = math.ceil(arr_len*train)
        train_arr, test_arr = np.split(arr, [split_pos])
        
        return train_arr, test_arr