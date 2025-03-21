import numpy as np


def mse(y, y_pred):
    '''

    Parameters
    ----------
    y : ndarray
        Correct labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    float
        Mean Squared Error of the correct and predicted labels.

    '''
    return np.sum( ( y - y_pred ) ** 2 ) / y.shape[0]

def dmse(y, y_pred):
    return 2/y.shape[0] * (y_pred-y)

def mae(y, y_pred):
    '''

    Parameters
    ----------
    y : ndarray
        Correct labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    float
        Mean Absolute Error of the correct and predicted labels.

    '''
    return np.sum( np.abs( y - y_pred ) ) / y.shape[0]

def dmae(y, y_pred):
    return 1/y.shape[0] * np.sign(y-y_pred)

def mbe( y, y_pred ) :
    '''

    Parameters
    ----------
    y : ndarray
        Correct labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    float
        Mean Bias Error of the correct and predicted labels.

    '''
    return np.sum( y - y_pred ) / y.shape[0]

def dmbe(y, y_pred):
    return -1/y.shape[0]

def cross_entropy(y, y_pred):
    '''

    Parameters
    ----------
    y : ndarray
        Correct labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    float
        Cross Entropy Loss of the correct and predicted labels.

    '''
    return - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / y.shape[0]

def dcross_entropy(y, y_pred):
    return (y_pred-y) / (y.shape[0]*y_pred*(1-y_pred))

def hinge(y, y_pred):
    '''

    Parameters
    ----------
    y : ndarray
        Correct labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    float
        Hinge Loss of the correct and predicted labels.

    '''    
    size = y.shape[0]
    
    l = np.sum(np.clip(1 - np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf))

    return l / size

def dhinge(y, y_pred):        
    l = np.array(np.clip(1-np.clip(np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf), a_min=0, a_max=np.inf), dtype=bool).reshape(-1,1)
    return -1*l*y
















































































