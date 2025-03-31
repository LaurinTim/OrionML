import numpy as np
import math
import copy

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ
import utils

class Linear():
    
    def __init__(self, dim1, dim2, activation, bias=True, alpha=0.1):
        '''
        Linear Layer.

        Parameters
        ----------
        dim1 : int
            Size of the input sample.
        dim2 : int
            Size of the output sample.
        activation : str
            Activation to use for this Layer. The available activations are: 
            {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.
        bias : TYPE, optional
            Whether or not the Layer contains a bias. The default is True.
        alpha : float
            Only used if the activation is "leakyrelu". Slope of the leaky ReLU when z<0. 
            The default is 0.1.

        '''
        self.dim1 = dim1
        self.dim2 = dim2
        self.bias = bias
        self.alpha = alpha
        self.w = np.zeros((dim1, dim2))
        self.b = np.zeros((1, dim2))
        
        self.activation = activation
        self.activation_function = self.get_activation_function()
        
        self.trainable = True
        self.dimension = np.array([dim1, dim2])
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to linear Layers.

        '''
        return "OrionML.Layer.Linear"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the linear Layer with information about the input and output 
            dimension and the activation.

        '''
        return f"OrionML.Layer.Linear  (shape:({self.dim1, self.dim2}), activation: {self.activation})"
    
    def get_activation_function(self):
        '''
        Get the correct activation function from the string input activation.

        Returns
        -------
        Correct activation function class from OrionMl.activation. If the value of self.activation 
        is not valid, an error is printed and a linear activation is used.

        '''
        if self.activation == "linear": return activ.linear()
        elif self.activation == "relu": return activ.relu()
        elif self.activation == "elu": return activ.elu()
        elif self.activation == "leakyrelu": return activ.leakyrelu(alpha=self.alpha)
        elif self.activation == "softplus": return activ.softplus()
        elif self.activation == "sigmoid": return activ.sigmoid()
        elif self.activation == "tanh": return activ.tanh()
        elif self.activation == "softmax": return activ.softmax()
        else:
            print("ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.")
            return activ.linear()
            
    def update_parameters(self, w_new, b_new=None):
        '''
        Updade the weights and bias of the current Layer.

        Parameters
        ----------
        w_new : ndarray, shape: (self.dim1, self.dim2)
            New weights to replace the current ones of the linear Layer.
        b_new : ndarray/None, optional, shape: (1, self.dim2)
            New bias to replace the current one of the linear Layer. If self.bias is set to False 
            this must be None, otherwise a ndarray. The default is None.

        '''
        assert not (self.bias==True and b_new is None), "ERROR: Expected a bias when updating the parameters."
        assert not (self.bias == False and not b_new is None), "ERROR: This Layer has no bias but a new bias was passed along."
        
        self.w = w_new
        if self.bias==True: self.b = b_new
        
        return
        
    def value(self, x, training=None):
        '''
        Pass an input to the linear Layer to get the output after the weights, bias and 
        activation function is applied.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, self.dim1)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        ndarray, shape: (number of samples, self.dim2)
            Array after the weights, bias and activation function of the linear Layer are 
            applied to the input data.
        z : ndarray, shape: (number of samples, self.dim2)
            Array after the weights and bias of the linear Layer are applied to the input data.

        '''
        z = np.matmul(x, self.w) + self.b
        return self.activation_function.value(z), z
    
    def derivative(self, x, training=None):
        '''
        Get the derivative of the activation function for the values after applying the 
        weights and bias of the linear Layer to the input data.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, self.dim1)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        d_activ : ndarray, shape: (input size, output size, output size)
            Derivative of the activation function for the values after applying the 
            weights and bias of the linear Layer to the input data.

        '''
        z = np.matmul(x, self.w) + self.b
        d_activ = self.activation_function.derivative(z)
        return d_activ
    
    def forward(self, prev_A, training=None):
        '''
        Forward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Data before the current linear Layer is applied.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim2)
            Data after the current linear Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
                    Input for the current forward step.
                self.w : ndarray, shape: (self.dim1, self.dim2)
                    Weights of the current linear Layer.
                self.b : ndarray, shape: (1, self.dim2)
                    Bias of the current linear Layer. If the current Layer has no bias, an array 
                    with shape contianing 0's is returned.
                Z : ndarray, shape: (number of samples, self.dim2)
                    Array after the weights and bias of the linear Layer are applied to the input data.

        '''
        curr_A, Z = self.value(prev_A)
        cache = (prev_A, self.w, self.b, Z)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=None):
        '''
        Backward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples passed to the Neural Network, self.dim2)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current linear Layer. 
            For its contens, refer to the return of self.forward.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        prev_dA : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Derivative of all Layers in the Neural Network starting from the current Layer.
        curr_dw : ndarray, shape: (self.dim1, self.dim2)
            Derivative of the weights of the current Layer given dA and the values in the cache.
        curr_db : ndarray, shape: (1, self.dim2)
            Derivative of the bias of the current Layer given dA and the values in the cache.

        '''
        prev_A, curr_w, curr_b, curr_Z = cache
        d_activation = self.derivative(prev_A)
        curr_dA = np.einsum('ijk,ik->ij', d_activation, dA)
        curr_dw = 1/prev_A.shape[0] * np.matmul(prev_A.T, curr_dA)
        curr_db = 1/prev_A.shape[0] * np.sum(curr_dA, axis=0, keepdims=True)
        prev_dA = np.matmul(curr_dA, curr_w.T)
        
        return prev_dA, curr_dw, curr_db
        
       
class Dropout():
    def __init__(self, dropout_probability=0.3, scale=True):
        '''
        Dropout Layer.

        Parameters
        ----------
        dropout_probability : TYPE, optional
            Probability for each node to be set to 0. The default is 0.3.
        scale : bool, optional
            Whether the remaining weights should be scaled by 1/dropout_probability.
            The default is True.


        '''
        self.dropout_probability = dropout_probability   
        self.scale = scale
        
        self.trainable = False
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to dropout Layers.

        '''
        return "OrionML.Layer.Dropout"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the dropout Layer with information about dropout probability.

        '''
        return f"OrionML.Layer.Dropout (dropout probability: {self.dropout_probability})"
        
    def value(self, activation_output, training=False):
        '''
        Set each element in activation_output with probability dropout_probability to 0 if training is True.
        
        Parameters
        ----------
        activation_output : ndarray, shape: (input size, output size)
            Output after an activation function to pass through the dropout Layer.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Copy of activation_output but each element set to 0 with probability dropout_probability.
        mask : ndarray, shape: (input size, output size)
            Is only returned if training is set to True. An array that is 0 at every element in 
            activation_output that was set to 0, otherwise 1.

        '''
        if training==False:
            return activation_output, np.zeros(1)
        
        mask = np.random.rand(activation_output.shape[0], activation_output.shape[1]) > self.dropout_probability
        res = mask*activation_output
        if self.scale==True:
            res = res * 1/(1-self.dropout_probability)
            
        return res, mask
    
    def derivative(self, mask, training=False):
        '''
        Get the derivative of the dropout Layer.

        Parameters
        ----------
        mask : ndarray, shape: (input size, output size)
            Mask from the dropout Layer when it was applied
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied and the derivative is the same as for linear activation. The default is False.

        Returns
        -------
        ndarray, shape: (input size, output size)
            If training is False, return an array filled with ones. Otherwise return mask.

        '''
        if training==False: return np.ones(mask.shape)
        return mask
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a dropout Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (input size, output size)
            Data before the current dropout Layer is applied.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        curr_A : ndarray, shape: (input size, output size)
            Data after the current dropout Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (input size, output size)
                    Input for the current forward step.
                curr_mask : ndarray, shape: (input size, output size)
                    Mask used in the dropout Layer.
                
        '''
        curr_A, curr_mask = self.value(prev_A, training=training)
        cache = (prev_A, curr_mask)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step of a dropout Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current dropout Layer. 
            For its contens, refer to the return of self.forward.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting from the current Layer.

        '''
        prev_A, curr_mask = cache
        d_Layer = self.derivative(curr_mask, training=training)
        prev_dA = d_Layer * dA
        return prev_dA


class Pool():
    def __init__(self, kernel_size, stride, padding=0, pool_mode="max"):
        '''
        Pooling Layer.

        Parameters
        ----------
        kernel_size : int
            Size of the window over which the pooling is applied.
        stride : int
            Stride of the window.
        padding : int, optional
            Zeros padding on the sides of the input. The default is 0.
        pool_mode : str, optional
            What type of pooling to apply. Has to be one of {max, avg}. The default is "max".

        '''
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_mode = pool_mode
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to pooling Layers.

        '''
        return "OrionML.Layer.Pool"
    
    def description(self):
        '''

        Returns
        -------
        str
            Description of the pooling Layer with information about the kernel size, stride, 
            padding and what type of pooling is applied.

        '''
        return f"OrionML.Layer.Pool (kernel size: {self.kernel_size}, stride: {self.stride}, padding: {self.padding}, pooling mode: {self.pool_mode})"
    
    
    def value(self, A, training=False):
        '''
        Apply pooling of type self.pool_mode to the input.
        
        Parameters
        ----------
        A : ndarray, shape: (input size, number of filters, dim1, dim2)
            array to apply the pooling to the second dimension
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Copy of activation_output but each element set to 0 with probability dropout_probability.

        '''
        A_w_cols = utils.im2col(A, self.kernel_size, self.kernel_size, self.stride)
        max_idx = np.argmax(A_w_cols.reshape(A.shape[1], self.kernel_size**2, -1), axis=1)
        
        #A_w = np.reshape(np.array([A_w_cols[:, 0::2], A_w_cols[:, 1::2]]), (A.shape[0], A.shape[1], self.kernel_size**2, ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1) * ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1)))
        A_w = np.reshape(np.array([A_w_cols[:, val::A.shape[0]] for val in range(A.shape[0])]), (A.shape[0], A.shape[1], self.kernel_size**2, ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1) * ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1)))
        
        if self.pool_mode=="max":
            A_w = np.max(A_w, axis=2)
            
        if self.pool_mode=="avg":
            A_w = np.mean(A_w, axis=2)
            
        cache = (A, A_w_cols.reshape(A.shape[1], self.kernel_size**2, -1), max_idx)
        
        return A_w.reshape(A.shape[0], A.shape[1], (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1, (A.shape[3] + 2*self.padding - self.kernel_size)//self.stride + 1), cache
    
    def derivative(self, A_prev, dA, x_cols, max_idx, training=False):
        '''
        Get the derivative of the pooling Layer.

        Parameters
        ----------
        mask : ndarray, shape: (input size, output size)
            Mask from the dropout Layer when it was applied
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied and the derivative is the same as for linear activation. The default is False.

        Returns
        -------
        ndarray, shape: (input size, output size)
            If training is False, return an array filled with ones. Otherwise return mask.

        '''        
        N, C, H, W = A_prev.shape

        # Reshape dout to match the dimensions of x_cols:
        # dout: (N, C, out_height, out_width) -> (C, 1, N*out_height*out_width)
        dA_reshaped = dA.transpose(1, 2, 3, 0).reshape(C, -1)
        
        if self.pool_mode=="max":
            # Initialize gradient for x_cols as zeros.
            dmax = np.zeros_like(x_cols)
            
            # Scatter the upstream gradients to the positions of the max indices.
            # For each channel and each pooling window, place the corresponding gradient at the max index.
            dmax[np.arange(C)[:, None], max_idx, np.arange(x_cols.shape[2])] = dA_reshaped
            
            # Reshape dmax back to the 2D column shape expected by col2im.
            dmax = dmax.reshape(C * self.kernel_size**2, -1)
            
            # Convert the columns back to the original image shape.
            dx = utils.col2im(dmax, A_prev.shape, self.kernel_size, self.kernel_size, self.stride)
            
        if self.pool_mode=="avg":
            dcols = np.repeat(dA_reshaped, self.kernel_size**2, axis=0) / (self.kernel_size**2)
            dx = utils.col2im(dcols, A_prev.shape, self.kernel_size, self.kernel_size, self.stride)

        return dx
    
    def forward(self, prev_A, training=False):
        curr_A, cache = self.value(prev_A, training=training)
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        A_prev, A_w_cols, max_idx = cache
        dx = self.derivative(A_prev, dA, A_w_cols, max_idx, training=training)
        return dx

# %%    

if __name__ == "__main__":
    a = np.array([[[[1, 1, 2, 4],
                    [5, 6, 7, 8],
                    [3, 2, 1, 0],
                    [1, 2, 3, 4]], 
                  
                   [[3, 2, 7, 4],
                    [8, 1, 4, 2],
                    [3, 1, 1, 2],
                    [5, 6, 2, 3]]]])
    
    l = Pool(3, 2, 0, pool_mode="max")
    aw, c = l.value(a)
    daw = l.backward(np.ones_like(aw), c)
    
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
    
    l = Pool(2, 2, 0, pool_mode="max")
    aw, c = l.value(a)
    daw = l.backward(np.ones_like(aw), c)
























































































