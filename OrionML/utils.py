import numpy as np
import math
import copy


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

def im2col_indices(x, field_height, field_width, stride=1):
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    out_height = (H - field_height) // stride + 1
    out_width = (W - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col(x, field_height, field_width, stride=1):
    k, i, j = im2col_indices(x, field_height, field_width, stride)
    cols = x[:, k, i, j]  # shape: (N, C * field_height * field_width, out_height*out_width)
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * x.shape[1], -1)
    return cols
    
def col2im(cols, x_shape, field_height, field_width, stride=1):
    N, C, H, W = x_shape
    out_height = (H - field_height) // stride + 1
    out_width = (W - field_width) // stride + 1
    cols_reshaped = cols.reshape(C * field_height * field_width, out_height * out_width, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    x = np.zeros(x_shape)
    k, i, j = im2col_indices(x, field_height, field_width, stride)
    # Use np.add.at to accumulate the gradients for overlapping regions.
    np.add.at(x, (slice(None), k, i, j), cols_reshaped)
    return x
    
# %%

if __name__ == "__main__":
    a = np.array([[[[1, 1, 2, 4],
                    [5, 6, 7, 8],
                    [3, 2, 1, 0],
                    [1, 2, 3, 4]], 
                  
                   [[3, 2, 7, 4],
                    [8, 1, 4, 2],
                    [3, 1, 1, 2],
                    [5, 6, 2, 3]]],
                  
                  [[[1, 1, 2, 4],
                    [5, 2, 7, 1],
                    [3, 2, 1, 0],
                    [1, 2, 5, 4]], 
                 
                   [[3, 2, 1, 4],
                    [1, 1, 4, 2],
                    [4, 1, 1, 2],
                    [5, 3, 2, 3]]]])
    
    al = im2col(a, 2, 2, 1)
    
    am = col2im(al, a.shape, 2, 2, 1)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    