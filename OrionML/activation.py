import numpy as np
from time import time

class linear():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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

class relu():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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

class elu():
    def __init__(self, alpha=0.1):
        '''
        
        Parameters
        ----------
        alpha : float
            Alpha value used in the eLU function. The default is 0.1.

        '''
        self.alpha = alpha
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray
            Values to put through the eLU activation function.
    
        Returns
        -------
        ndarray
            Values after the eLU activation.
    
        '''
        return np.clip(z, a_min=0, a_max=np.inf) + np.array(z<0, dtype=float)*self.alpha*(np.exp(z)-1)
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray
            Values for which to get the derivate with respect to a eLU activation function.
    
        Returns
        -------
        ndarray
            Derivative at the values with respect to a eLU activation function.
    
        '''
        return np.array(z>0, dtype=float) + (np.array(z<0, dtype=float))*self.alpha*np.exp(z)

class leakyrelu():
    def __init__(self, alpha=0.1):
        '''
        
        Parameters
        ----------
        alpha : float
            Slope of the leaky ReLU when z<0. The default is 0.1.

        '''
        self.alpha = alpha
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray
            Values to put through the Leaky ReLU activation function.
    
        Returns
        -------
        ndarray
            Values after the Leaky ReLU activation.
    
        '''
        return np.clip(z, a_min=0, a_max=np.inf) + self.alpha*np.clip(z, a_min=-np.inf, a_max=0)
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray
            Values for which to get the derivate with respect to a Leaky ReLU activation function.
    
        Returns
        -------
        ndarray
            Derivative at the values with respect to a Leaky ReLU activation function.
    
        '''
        return np.array(z>0, dtype=float) + self.alpha*np.array(z<0, dtype=float)

class softplus():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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

class sigmoid():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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

class tanh():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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
        res = self.value(z)
        return 1 - res**2

class softmax():
    def __init__(self):
        return
    
    def value(self, z):
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
    
    def derivative(self, z):
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
        sz = self.value(z)
        res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
        res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]))
        return res
 