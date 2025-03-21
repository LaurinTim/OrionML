import numpy as np
from time import time

def linear(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Linear activation.

    Returns
    -------
    z : ndarray
        Values after the Linear activation.

    '''
    return z

def dlinear(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Linear activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Linear activation function.

    '''
    return np.ones(z.shape)

def relu(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the ReLU activation function.

    Returns
    -------
    ndarray
        Values after the ReLU activation.

    '''
    return np.clip(z, a_min=0, a_max=np.inf)

def drelu(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a ReLU activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a ReLU activation function.

    '''
    return np.array(z>0, dtype=float)

def elu(z, alpha=1):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the eLU activation function.
    alpha : float
        Alpha value used in the eLU function. The default is 1.

    Returns
    -------
    ndarray
        Values after the eLU activation.

    '''
    return np.clip(z, a_min=0, a_max=np.inf) + np.array(z<0, dtype=float)*alpha*(np.exp(z)-1)

def delu(z, alpha=1):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a eLU activation function.
    alpha : float
        Alpha value used in the eLU function. The default is 1.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a eLU activation function.

    '''
    return np.array(z>0, dtype=float) + (np.array(z<0, dtype=float))*alpha*np.exp(z)

def leakyrelu(z, alpha=0.1):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Leaky ReLU activation function.
    alpha : float
        Slope of the leaky ReLU when z<0. The default is 0.1.

    Returns
    -------
    ndarray
        Values after the Leaky ReLU activation.

    '''
    return np.clip(z, a_min=0, a_max=np.inf) + alpha*np.clip(z, a_min=-np.inf, a_max=0)

def dleakyrelu(z, alpha=0.1):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Leaky ReLU activation function.
    alpha : float
        Slope of the leaky ReLU when z<0. The default is 0.1.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Leaky ReLU activation function.

    '''
    return np.array(z>0, dtype=float) + alpha*np.array(z<0, dtype=float)

def softplus(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Softplus activation function.

    Returns
    -------
    ndarray
        Values after the Softplus activation.

    '''
    return np.log(1+np.exp(z))

def dsoftplus(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Softplus activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Softplus activation function.

    '''
    return 1/(1+np.exp(z))

def sigmoid(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Sigmoid activation function.

    Returns
    -------
    ndarray
        Values after the Sigmoid activation.

    '''
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Sigmoid activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Sigmoid activation function.

    '''
    sig = sigmoid(z)
    return sig*(1-sig)

def tanh(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Tanh activation function.

    Returns
    -------
    ndarray
        Values after the Tanh activation.

    '''
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def dtanh(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Tanh activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Tanh activation function.

    '''
    res = tanh(z)
    return 1 - res**2

def softmax(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values to put through the Softmax activation function.

    Returns
    -------
    ndarray
        Values after the Softmax activation.

    '''
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def dsoftmax(z):
    '''

    Parameters
    ----------
    z : ndarray
        Values for which to get the derivate with respect to a Softmax activation function.

    Returns
    -------
    ndarray
        Derivative at the values with respect to a Softmax activation function.

    '''
    sz = softmax(z)
    res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
    res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]))
    return res
 