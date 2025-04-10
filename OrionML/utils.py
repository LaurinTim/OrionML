import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def train_test_split(arr, train=1, shuffle=True):
    '''

    Parameters
    ----------
    arr : ndarray
        Array containing the data. The target should also be included in this array.
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

class StandardScaler:
    def __init__(self):
        return
    
    def fit(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        Sets self.mean and self.std to the mean and standard deviation of the columns in arr.

        '''
        self.mean = arr.mean(axis=0, keepdims=True)
        self.std = arr.std(axis=0, keepdims=True)
        
    def transform(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        arr : ndarray
            Scaled array.

        '''
        arr = arr - self.mean
        arr = arr/(self.std + 1e-8)
        return arr
    
    def fit_transform(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        ndarray
            Scaled array.

        '''
        self.fit(arr)
        return self.transform(arr)
    
class MinMaxScaler:
    def __init__(self):
        return
    
    def fit(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        Sets self.min and self.max to the minimum and maximum of the columns of arr.

        '''
        self.min = arr.min(axis=0, keepdims=True)
        self.std = arr.max(axis=0, keepdims=True)
        
    def transform(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        arr : ndarray
            Scaled array.

        '''
        arr = (arr-self.min)/(self.max-self.min + 1e-8)
        return arr
    
    def fit_transform(self, arr):
        '''

        Parameters
        ----------
        arr : ndarray
            Data to scale.

        Returns
        -------
        ndarray
            Scaled array.

        '''
        self.fit(arr)
        return self.transform(arr)
    

def plot_confusion_matrix(cmx, labels, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_norm = np.around(cmx_norm, 2)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(12, 6)
    [a.set_xticks(range(len(labels)), labels=labels, size=12) for a in ax]
    [a.set_yticks(range(len(labels)), labels=labels, size=12) for a in ax]
    [a.set_xlabel(xlabel="Predicted Label", size=13) for a in ax]
    [a.set_ylabel(ylabel="True Label", size=13) for a in ax]
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cmx_norm[i, j]>=np.max(cmx_norm)*0.6:
                ax[0].text(j, i, cmx_norm[i, j], ha="center", va="center", c="black", size=10)
            else:
                ax[0].text(j, i, cmx_norm[i, j], ha="center", va="center", c="white", size=10)
                
            if cmx_zero_diag[i, j]>=np.max(cmx_zero_diag)*0.5:
                ax[1].text(j, i, cmx_zero_diag[i, j], ha="center", va="center", c="black", size=10)
            else:
                ax[1].text(j, i, cmx_zero_diag[i, j], ha="center", va="center", c="white", size=10)
         
    im1 = ax[0].imshow(cmx_norm, vmax=vmax2)
    ax[0].set_title('%', size=15)
    im2 = ax[1].imshow(cmx_zero_diag, vmax=vmax3)
    ax[1].set_title('% and 0 diagonal', size=15)
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.tight_layout()

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
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    