import numpy as np
import math
import copy
from time import time
import numba as nb

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ
import utils

    
@nb.njit
def back_gradient(dA_strided, weights):
    B, H, W, k, l, o = dA_strided.shape
    
    _, _, i, _ = weights.shape
    
    dA = np.zeros((B, H, W, i), dtype=dA_strided.dtype)
    
    for b in range(B):
        for h in range(H):
            for w in range(W):
                for ki in range(k):
                    for li in range(l):
                        for oi in range(o):
                            a_val = dA_strided[b, h, w, ki, li, oi]
                            for ii in range(i):
                                dA[b, h, w, ii] += a_val*weights[ki, li, ii, oi]
        
    return dA

class Conv_old():
    def __init__(self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0, flatten=False, bias=True):
        '''
        Convolutional Layer.
        The shape of the weights is (kernel_size, kernel_size, in_channels, out_channels).
        The shape of the bias is (1, out_channels).
        The last columns and rows of the input to the layer get ignored if the kernel does not 
        fit nicely to the input data in the convolution.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input data.
        out_channels : int
            Number of channels in the output data.
        kernel_size : int
            Size of the kernel. The kernel is of shape (kernel_size, kernel_size).
        activation : str
            Activation to use for this Layer. The available activations are: 
                {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh}.
        stride : int, optional
            Stride of the convolution. The default is 1.
        padding : int, optional
            Padding used in the convolution. The default is 0.
        flatten : bool, optional
            Whether or not the output of the layer should be flattened. Has to be True if the 
            following layer is a Linear layer. The default is False.
        bias : bool, optional DOES NOT WORK YET
            Whether or not the layer has a bias term. The default is True.

        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.flatten = flatten
        self.bias = bias
        
        self.activation = activation
        self.activation_function = self.get_activation_function()
        
        self.trainable = True
        self.dimension = np.array([self.kernel_size, self.kernel_size, self.in_channels, self.out_channels])
        
        self.w = np.zeros((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
        self.b = np.zeros((1, self.out_channels))
        
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0
        self.t5 = 0
                
    def type(self):
        '''

        Returns
        -------
        str
            String unique to convolutional Layers.

        '''
        return "OrionML.Layer.Conv"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the convolutional Layer with information about the input and output 
            channels, the kernel size, the stride and the padding.

        '''
        return f"OrionML.Layer.Conv  (shape:({self.kernel_size, self.kernel_size, self.in_channels, self.out_channels}), stride: {self.stride}, padding: {self.padding})"
    
    def get_activation_function(self):
        '''
        Get the correct activation function from the string input activation.

        Returns
        -------
        Correct activation function class from OrionMl.activation.

        '''
        if self.activation == "linear": return activ.linear()
        elif self.activation == "relu": return activ.relu()
        elif self.activation == "elu": return activ.elu()
        elif self.activation == "leakyrelu": return activ.leakyrelu(alpha=self.alpha)
        elif self.activation == "softplus": return activ.softplus()
        elif self.activation == "sigmoid": return activ.sigmoid()
        elif self.activation == "tanh": return activ.tanh()
        else:
            assert False, "ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh}."
            
            
    def update_parameters(self, w_new, b_new=None):
        '''
        Updade the weights and bias of the current Layer.

        Parameters
        ----------
        w_new : ndarray, shape: (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
            New weights to replace the current ones of the convolutional Layer.
        b_new : ndarray/None, optional, shape: (1, self.out_channels)
            New bias to replace the current one of the convolutional Layer. If self.bias is set to False 
            this must be None, otherwise a ndarray. The default is None.

        '''
        assert not (self.bias==True and b_new is None), "ERROR: Expected a bias when updating the parameters."
        assert not (self.bias == False and not b_new is None), "ERROR: This Layer has no bias but a new bias was passed along."
        
        self.w = w_new
        if self.bias==True: self.b = b_new
        
        return
        
    def value(self, A, training=False):
        '''
        Pass an input to the convolutional Layer to get the output after the weights and bias  
        are applied.

        Parameters
        ----------
        A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.

        Returns
        -------
        prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.
        prev_A_strided : ndarray, shape: (number of samples, output height, output width, kernel size, kernel size, input channels)
            Array containing the sliding windows of prev_A used for the convolution.

        '''
        # =============================================================================
        #  As a reminder: ndarray.strides gives the number of bytes to step until the next element is reached in each dimension. Each number in the arrays is of type np.float64, the last 
        #  Dimension of A.strides will be 64/8=4. 
        #  For the array np.array([[0,1,2],[3,4,5]]) b.strides is (12, 4) since each number is a 32 bit integer and thus there are 4 bytes for each number.
        #  The first dimension is filled with three 32 bit integers and thus the stride for the first dimension is 3*4=12.
        #  For the array np.array([[0,1,2],[3,4,5]], dtype=float) b.strides is (24, 8) since each number is a 64 bit float and thus there are 8 bytes for each number.
        #  The first dimension is filled with three 64 bit floats and thus the stride for the first dimension is 3*8=24. 
        # =============================================================================
            
        h_out = (A.shape[1] + 2*self.padding - self.kernel_size)//self.stride + 1
        w_out = (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1
        
        A_padded = np.copy(A)
        
        if self.padding > 0:
            A_padded = np.pad(A_padded, pad_width=((0,), (self.padding,), (self.padding,), (0,)), mode="constant", constant_values=(0.,))
        
        batch_stride, kern_h_stride, kern_w_stride, channel_stride = A_padded.strides
        strides = (batch_stride, self.stride*kern_h_stride, self.stride*kern_w_stride, kern_h_stride, kern_w_stride, channel_stride)
        
        A_strided = np.lib.stride_tricks.as_strided(A_padded, (A.shape[0], h_out, w_out, self.w.shape[0], self.w.shape[1], A.shape[3]), strides)
        
        #A_convoluted = np.einsum('abcijk,ijkd', A_strided, self.w) + self.b
        A_convoluted = np.tensordot(A_strided, self.w, axes=3) + self.b
        
        output = self.activation_function.value(A_convoluted)
        
        if self.flatten:
            output = output.reshape(output.shape[0], -1)
        
        return output, (A_strided, A_convoluted)
    
    def forward(self, prev_A, training=None):
        '''
        Forward step of a convolutional Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for convolutional 
            Layers. The default is None.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples, output height, output width, output channels) if not flatten,
                                else (number of samples, output height*output width*output channels)
            Data after the convolutional Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
                    Input data.
                prev_A_strided : ndarray, shape: (number of samples, output height, output width, kernel size, kernel size, input channels)
                    Array containing the sliding windows of prev_A used for the convolution.

        '''
        curr_A, (prev_A_strided, A_convoluted) = self.value(prev_A)
        cache = (prev_A, prev_A_strided, A_convoluted)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step for a convolutional Layer.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples, output height, output width, output channels) if not flatten, 
                            else (number of samples, output height*output width*output channels)
            Upstream gradient.
        cache : tuple
            Cache containing information from the forwards propagation used in the backwards propagation.

        Returns
        -------
        curr_dA : ndarray, shape: (number of samples, input height, input width, input channels)
            Gradient to be passed as upstream gradient to the next layer in backwards propagation.
        db : ndarray, shape: (1, input channels)
            Gradient of the bias.
        dw : ndarray, shape: (kernel size, kernel size, input channels, output channels)
            Gradient of the weights.

        '''
        st1 = time()
        
        A, A_strided, A_convoluted = cache
        
        #If flatten is True, dA is of shape (number of samples, output height*output width*output channels), so dA needs to be reshaped to 
        #(number of samples, output height, output width, output channels). The height and width of the output can be calculated with:
        #output height = (A.shape[1] + 2*self.padding - self.kernel_size)//self.stride + 1
        #output width = (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1
        #where A is the input for the forward function in this layer called previously, saved as cache[0].
        if self.flatten:
            dA = dA.reshape(dA.shape[0], (A.shape[1] + 2*self.padding - self.kernel_size)//self.stride + 1, (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1, self.out_channels)
        
        d_activ = self.activation_function.derivative(A_convoluted)
        dA = d_activ * dA
        
        dA_padded = np.copy(dA)
        back_padding = self.kernel_size - 1 if self.padding==0 else self.padding
                
        if self.stride > 1:
            dA_padded = np.insert(dA, range(1, dA.shape[1]), 0, axis=1)
            dA_padded = np.insert(dA_padded, range(1, dA.shape[2]), 0, axis=2)
                
        if back_padding > 0:
            dA_padded = np.pad(dA_padded, pad_width=((0,), (back_padding,), (back_padding,), (0,)), mode="constant", constant_values=(0.,))
        
        batch_stride, kern_h_stride, kern_w_stride, channel_stride = dA_padded.strides
        strides = (batch_stride, 1*kern_h_stride, 1*kern_w_stride, kern_h_stride, kern_w_stride, channel_stride)
        
        dA_strided = np.lib.stride_tricks.as_strided(dA_padded, (dA.shape[0], A.shape[1], A.shape[2], self.w.shape[0], self.w.shape[1], dA.shape[3]), strides)
                
        db = np.array([np.sum(dA, axis=(0, 1, 2))])
        
        #dw = np.einsum('bhwkli,bhwo->klio', A_strided, dA, optimize="greedy")
        st2 = time()
        dw = np.tensordot(A_strided, dA, axes=([0,1,2], [0,1,2]))
        self.t2 += time()-st2
        
        #Try and use numba for dA_flat =  np.ascontiguousarray(dA_strided).reshape(N, -1)
        #curr_dA = np.einsum('bhwklo,klio->bhwi', dA_strided, np.rot90(self.w, 2, axes=(0,1)), optimize="greedy")
        st3 = time()
        #curr_dA = np.tensordot(dA_strided, self.w, axes=([3,4,5], [0,1,3]))
        
        curr_dA = back_gradient(dA_strided, self.w)
        
        '''
        w_flat = self.w.transpose(0, 1, 3, 2).reshape(-1, self.in_channels)
        
        N = dA_strided.shape[0] * dA_strided.shape[1] * dA_strided.shape[2]
        
        if self.t4==0 and False:
            print(dA_strided.shape, np.size(dA_strided))
                        
        st4 = time()
        dA_flat =  np.ascontiguousarray(dA_strided).reshape(N, -1)
        self.t4 += time()-st4
        
        curr_dA = dA_flat.dot(w_flat).reshape(dA_strided.shape[0], dA_strided.shape[1], dA_strided.shape[2], -1)
        '''
                        
        self.t3 += time()-st3
        
        
        #b, h, w, k, l, i = A_strided.shape
        #_, _, _, o = dA.shape
        #N = b*h*w
        #A_flat = A_strided.reshape(N, k*l*i)
        #dA_flat = dA.reshape(N, o)
        #dw = np.matmul(A_flat.T, dA_flat).reshape(k, l, i, o)
        
        self.t1 += time()-st1-1000000000
        
        return curr_dA, dw, db
    