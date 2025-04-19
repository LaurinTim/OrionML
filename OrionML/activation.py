import numpy as np
from time import time

class linear():
    def __init__(self):
        return
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the Linear activation.
    
        Returns
        -------
        z : ndarray, shape: (input size, output size)
            Values after the Linear activation.
    
        '''
        return z
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Linear activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Linear activation function.
    
        '''
        return np.ones(z.shape)
        #return np.tile(np.eye(z.shape[1])[None,:,:], (z.shape[0],1,1))

class relu():
    def __init__(self):
        self.buffers = {}
    
    def init_buffers(self, batch_size, sample_shape):
        return
    
    def value_buffered(self, z, out_buffer):
        return np.clip(z, a_min=0, a_max=np.inf, out=out_buffer)
    
    def derivative_buffered(self, z, out_buffer):
        return np.greater_equal(z, 0, out=out_buffer)
        
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the ReLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the ReLU activation.
    
        '''
        return np.clip(z, a_min=0, a_max=np.inf)
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a ReLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Derivative at the values with respect to a ReLU activation function.
    
        '''
        return z>=0
        #return np.einsum("ij,jk -> ijk", np.array(z>0, dtype=float), np.eye(z.shape[1]))

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
        z : ndarray, shape: (input size, output size)
            Values to put through the eLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the eLU activation.
    
        '''
        return np.clip(z, a_min=0, a_max=np.inf) + np.array(z<0, dtype=float)*self.alpha*(np.exp(z)-1)
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a eLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a eLU activation function.
    
        '''
        return np.array(z>=0, dtype=float) + (np.array(z<0, dtype=float))*self.alpha*np.exp(z)
        #return np.einsum("ij,jk -> ijk", np.array(z>0, dtype=float) + (np.array(z<0, dtype=float))*self.alpha*np.exp(z), np.eye(z.shape[1]))

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
        z : ndarray, shape: (input size, output size)
            Values to put through the Leaky ReLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the Leaky ReLU activation.
    
        '''
        return np.clip(z, a_min=0, a_max=np.inf) + self.alpha*np.clip(z, a_min=-np.inf, a_max=0)
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Leaky ReLU activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Leaky ReLU activation function.
    
        '''
        return np.array(z>=0, dtype=float) + self.alpha*np.array(z<0, dtype=float)
        #return np.einsum("ij,jk -> ijk", np.array(z>0, dtype=float) + self.alpha*np.array(z<0, dtype=float), np.eye(z.shape[1]))

class softplus():
    def __init__(self):
        return
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the Softplus activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the Softplus activation.
    
        '''
        return np.log(1+np.exp(z))
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Softplus activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Softplus activation function.
    
        '''
        return 1/(1+np.exp(z))
        #return np.einsum("ij,jk -> ijk", 1/(1+np.exp(z)), np.eye(z.shape[1]))

class sigmoid():
    def __init__(self):
        return
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the Sigmoid activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the Sigmoid activation.
    
        '''
        return 1/(1+np.exp(-z))
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Sigmoid activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Sigmoid activation function.
    
        '''
        sig = self.value(z)
        return sig*(1-sig)
        #return np.einsum("ij,jk -> ijk", sig*(1-sig), np.eye(z.shape[1]))

class tanh():
    def __init__(self):
        return
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the Tanh activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the Tanh activation.
    
        '''
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Tanh activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Tanh activation function.
    
        '''
        res = self.value(z)
        return 1 - res**2
        #return np.einsum("ij,jk -> ijk", 1 - res**2, np.eye(z.shape[1]))

class softmax():
    def __init__(self):
        self.buffers = {}
        
    def init_buffers(self, batch_size, sample_shape):
        self.buffers["shifted_z"] = np.empty((batch_size, *sample_shape))
        self.buffers["exp_z"] = np.empty((batch_size, *sample_shape))
        
        self.buffers["value"] = np.empty((batch_size, *sample_shape))
        self.buffers["mult"] = np.empty((batch_size, *sample_shape, *sample_shape))
        self.buffers["diagonal"] = np.empty((batch_size, *sample_shape, *sample_shape))
    
    def value_buffered(self, z, out_buffer):
        shifted_z_buffer = self.buffers["shifted_z"]
        exp_z_buffer = self.buffers["exp_z"]
        
        np.subtract(z, np.max(z, axis=1, keepdims=True), out=shifted_z_buffer)
        np.exp(shifted_z_buffer, out=exp_z_buffer)
        np.divide(exp_z_buffer, (np.sum(exp_z_buffer, axis=1, keepdims=True)), out=out_buffer)
        
        return
    
    def derivative_buffered(self, z, out_buffer):
        value_buffer = self.buffers["value"]
        mult_buffer = self.buffers["mult"]
        diagonal_buffer = self.buffers["diagonal"]
        
        self.value_buffered(z, out_buffer=value_buffer)
        np.multiply(-value_buffer.reshape(value_buffer.shape[0],-1,1), value_buffer.reshape(value_buffer.shape[0],1,-1), out=mult_buffer)
        np.einsum("ij,jk->ijk", value_buffer, np.eye(value_buffer.shape[1]), optimize="optimal", out=diagonal_buffer)
        np.add(mult_buffer, diagonal_buffer, out=out_buffer)
        return
    
    def value(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values to put through the Softmax activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size)
            Values after the Softmax activation.
    
        '''
        shifted_z = z-np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z/(np.sum(exp_z, axis=1, keepdims=True))
    
    def derivative(self, z):
        '''
    
        Parameters
        ----------
        z : ndarray, shape: (input size, output size)
            Values for which to get the derivate with respect to a Softmax activation function.
    
        Returns
        -------
        ndarray, shape: (input size, output size, output size)
            Derivative at the values with respect to a Softmax activation function.
    
        '''
        sz = self.value(z)
        res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
        res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]), optimize="optimal")
        return res
    
# %%

if __name__ == "__main__":
    a = np.array([[0,1,0], [1,0,0]])
    b = np.array([[0.2,0.7,0.1], [0.6,0.3,0.1]])
    
    f = softmax()
    f.init_buffers(2, (3,))
    
    vb = np.empty((2, 3))
    db = np.empty((2, 3, 3))
    
    v = f.value(b)
    d = f.derivative(b)
    
    f.value_buffered(b, vb)
    f.derivative_buffered(b, db)
    
# %%

if __name__ == "__main__":
    a = np.array([[[[-2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    f = linear()
    r = f.value(a)
    dr = f.derivative(a)

# %%

if __name__ == "__main__":
    in_channels = 3
    out_channels = 128
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = (4, in_channels, 12, 10)  # expected input size

    np.random.seed(42)  # for reproducibility

    x = np.random.random(batch_size)  # create data for forward pass
    
    x = np.transpose(x, axes=(0,2,3,1))
    
    f = linear()
    r = f.value(x)
    dr = f.derivative(x)

























