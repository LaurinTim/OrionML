import numpy as np
import math
import copy
from time import time
from pathlib import Path
import sys

import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

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
        bias : TYPE, optional DOES NOT WORK YET
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
        Correct activation function class from OrionMl.activation.

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
            assert False, "ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}."
            
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
        
    def value(self, x, training=False):
        '''
        Pass an input to the linear Layer to get the output after the weights, bias and 
        activation function is applied.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, self.dim1)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is False.

        Returns
        -------
        ndarray, shape: (number of samples, self.dim2)
            Array after the weights, bias and activation function of the linear Layer are 
            applied to the input data.
        z : ndarray, shape: (number of samples, self.dim2)
            Array after the weights and bias of the linear Layer are applied to the input data.

        '''
        z = np.matmul(x, self.w) + self.b
        out = self.activation_function.value(z)
                    
        return out, z
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Data before the current linear Layer is applied.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is False.

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
        curr_A, Z = self.value(prev_A, training=training)
        cache = (prev_A, self.w, self.b, Z)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
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
            Layers. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Derivative of all Layers in the Neural Network starting from the current Layer.
        curr_dw : ndarray, shape: (self.dim1, self.dim2)
            Derivative of the weights of the current Layer given dA and the values in the cache.
        curr_db : ndarray, shape: (1, self.dim2)
            Derivative of the bias of the current Layer given dA and the values in the cache.

        '''
        prev_A, curr_w, curr_b, curr_z = cache
        
        d_activation = self.activation_function.derivative(curr_z)
        
        if self.activation != "softmax":
            curr_dA = dA * d_activation
            
        else:
            curr_dA = np.einsum('ijk,ik->ij', d_activation, dA, optimize="optimal")
                        
        #curr_dw = 1/prev_A.shape[0] * np.matmul(prev_A.T, curr_dA)
        #curr_db = 1/prev_A.shape[0] * np.sum(curr_dA, axis=0, keepdims=True)
        curr_dw = np.matmul(prev_A.T, curr_dA)
        curr_db = np.sum(curr_dA, axis=0, keepdims=True)
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
        activation_output : ndarray
            Output after an activation function to pass through the dropout Layer.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: activation_output.shape
            Copy of activation_output but each element set to 0 with probability dropout_probability.
        mask : ndarray, shape: activation_output.shape
            Is only returned if training is set to True. An array that is 0 at every element in 
            activation_output that was set to 0, otherwise 1.

        '''
        if not training:
            return activation_output, np.zeros(1)
        
        mask = np.random.rand(*activation_output.shape) > self.dropout_probability
        res = mask*activation_output
        if self.scale:
            res = res * 1/(1-self.dropout_probability)
            
        return res, mask
    
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
        assert training, "Training set to False in the backward pass of a Dropout layer."
        
        prev_A, curr_mask = cache
        prev_dA = curr_mask * dA
        
        if self.scale:
            prev_dA = prev_dA * 1/(1-self.dropout_probability)
        
        return prev_dA

    
class BatchNorm():
    def __init__(self, sample_dim, momentum=0.9, epsilon=1e-8):
        '''
        Batch normalization Layer.

        Parameters
        ----------
        sample_dim : int
            Number of features in the input data.
        momentum : float, optional
            Momentum used to calculate the running mean. The default is 0.9.
        epsilon : float, optional
            Epsilon value used to avoid division by 0. The default is 1e-8.

        Returns
        -------
        None.

        '''
        self.epsilon = epsilon
        self.momentum = momentum
        self.sample_dim = sample_dim
        self.dimension = np.array([self.sample_dim])
        
        self.gamma = np.random.randn(1, self.sample_dim)
        self.beta = np.zeros((1, self.sample_dim))
        self.running_mean = np.zeros((1, self.sample_dim))
        self.running_variance = np.zeros((1, self.sample_dim))
        
        self.trainable = True
                
    def type(self):
        '''

        Returns
        -------
        str
            String unique to Batch normalization Layers.

        '''
        return "OrionML.Layer.BatchNorm"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the Batch normalization Layer with information about the number of input features.

        '''
        return f"OrionML.Layer.BatchNorm (number of input features: {self.sample_dim})"
        
    def value(self, x, training=False):
        '''
        
        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input for the batch normalization.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: (number of samples, number of features)
            Output of the batch normalization layer.

        '''
        if training==True:
            mean = np.mean(x, axis=0, keepdims=True)
            variance = np.var(x, axis=0, keepdims=True)
            x_normalized = (x-mean)/np.sqrt(variance + self.epsilon)
            
            out = self.gamma*x_normalized + self.beta
                    
            return out, x_normalized, mean, variance
        
        elif training==False:
            x_normalized = (x-self.running_mean)/np.sqrt(self.running_variance + self.epsilon)
            out = self.gamma*x_normalized + self.beta
                        
            return out, ()
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples, number of features)
            Data before the current dropout Layer is applied.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples, number of features)
            Data after the Batch normalization Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples, number of features)
                    Input data.
                x_normalized : ndarray, shape: (number of samples, input height, input width, input channels)
                    Normalized input data, before gamma and beta are applied.
                batch_mean : ndarray, shape: (1, number of features)
                    Mean of the elements in each channel of prev_A.
                batch_variance : ndarray, shape: (1, number of features)
                    Variance of the elements in each channel of prev_A.
                
        '''
        if training==True:
            curr_A, x_normalized, batch_mean, batch_variance = self.value(prev_A, training=training)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*batch_mean
            self.running_variance = self.momentum*self.running_variance + (1-self.momentum)*batch_variance
            
            cache = (prev_A, x_normalized, batch_mean, batch_variance)
            
        elif training==False:
            curr_A, _ = self.value(prev_A, training=training)
            
            cache = (prev_A)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples, number of features)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current dropout Layer. 
            For its contens, refer to the return of self.forward.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (number of samples, number of features)
            Derivative of all Layers in the Neural Network starting from the current Layer.
        dgamma : ndarray, shape: (1, number of features)
            Gradient of gamma.
        dbeta : ndarray, shape: (1, number of features)
            Gradient of beta.

        '''
        assert training, "Training should be True for backward propagation."
        
        prev_A, x_normalized, batch_mean, batch_variance = cache
        
        dgamma = np.sum(dA*x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(dA, axis=0, keepdims=True)
        
        m = prev_A.shape[0]
        t = 1/np.sqrt(batch_variance + self.epsilon)
        curr_dA = (self.gamma * t/m) * (m*dA - np.sum(dA, axis=0) - t**2 * (prev_A-batch_mean) * np.sum(dA * (prev_A - batch_mean), axis=0))
        
        return curr_dA, dgamma, dbeta


class BatchNorm2D():
    def __init__(self, channels, momentum=0.9, epsilon=1e-8):
        '''
        Batch normalization Layer for the output of a convolutional layer.

        Parameters
        ----------
        channels : int
            Number of channels in the input data.
        momentum : float, optional
            Momentum used to calculate the running mean. The default is 0.9.
        epsilon : float, optional
            Epsilon value used to avoid division by 0. The default is 1e-8.

        Returns
        -------
        None.

        '''
        self.epsilon = epsilon
        self.momentum = momentum
        self.channels = channels
        self.sample_dim = channels
        self.dimension = np.array([self.channels])
        
        self.gamma = np.random.randn(1, self.channels)
        self.beta = np.zeros((1, self.channels))
        self.running_mean = np.zeros((1, self.channels))
        self.running_variance = np.zeros((1, self.channels))
        
        self.trainable = True
                
    def type(self):
        '''

        Returns
        -------
        str
            String unique to 2D Batch normalization Layers.

        '''
        return "OrionML.Layer.BatchNorm2D"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the 2D Batch normalization Layer with information about the number of input channels.

        '''
        return f"OrionML.Layer.BatchNorm2D (input channels: {self.channels})"
        
    def value(self, x, training=False):
        '''
        
        Parameters
        ----------
        x : ndarray, shape: (number of samples, input height, input width, input channels)
            Input for the 2D batch normalization.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: (number of samples, input height, input width, input channels)
            Output of the 2D batch normalization layer.

        '''
        #In Batch normalization for convolutional layers, the normalization should be performed over each channel
        
        if training==True:
            #Get array with shape (x.shape[0]*x.shape[1]*x.shape[2], x.shape[3]) where the ith column corresponds to all
            #elements in x[:,:,:,i], so all elements in the ith channel. The normalization is then done the same way as 
            #for linear layers using this new array as input.
            x_channels = x.copy().reshape(-1, x.shape[3])
            mean = np.mean(x_channels, axis=0, keepdims=True)
            variance = np.var(x_channels, axis=0, keepdims=True)
            x_normalized = (x-mean)/np.sqrt(variance + self.epsilon)
            out = self.gamma*x_normalized + self.beta
                    
            return out, x_normalized, mean, variance
        
        elif training==False:
            x_normalized = (x-self.running_mean)/np.sqrt(self.running_variance + self.epsilon)
            out = self.gamma*x_normalized + self.beta
            
            return out, ()
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a 2D Batch normalization layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Data before the 2D batch normalization layer is applied.
        training : bool, optional
            Whether the layer is currently in training or not. The default is False.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Data after the Batch normalization Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
                    Input data.
                x_normalized : ndarray, shape: (number of samples, input height, input width, input channels)
                    Normalized input data, before gamma and beta are applied.
                batch_mean : ndarray, shape: (1, input channels)
                    Mean of the elements in each channel of prev_A.
                batch_variance : ndarray, shape: (1, input channels)
                    Variance of the elements in each channel of prev_A.
                
        '''
        if training==True:
            curr_A, x_normalized, batch_mean, batch_variance = self.value(prev_A, training=training)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*batch_mean
            self.running_variance = self.momentum*self.running_variance + (1-self.momentum)*batch_variance
            cache = (prev_A, x_normalized, batch_mean, batch_variance)
            
        elif training==False:
            curr_A = self.value(prev_A, training=training)
            cache = (prev_A)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples, input height, input width, input channels)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current dropout Layer. 
            For its contens, refer to the return of self.forward.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (number of samples, input height, input width, input channels)
            Derivative of all Layers in the Neural Network starting from the current Layer.
        dgamma : ndarray, shape: (1, input channels)
            Gradient of gamma.
        dbeta : ndarray, shape: (1, input channels)
            Gradient of beta.

        '''
        assert training, "Training should be True for backward propagation."
        
        prev_A, x_normalized, batch_mean, batch_variance = cache
        
        dA_channels = dA.copy().reshape(-1, dA.shape[3])
        x_normalized_channels = x_normalized.copy().reshape(-1, x_normalized.shape[3])
        
        dgamma = np.sum(dA_channels*x_normalized_channels, axis=0, keepdims=True)
        dbeta = np.sum(dA_channels, axis=0, keepdims=True)
        
        m = dA_channels.shape[0]
        t = 1/np.sqrt(batch_variance + self.epsilon)
        curr_dA = (self.gamma * t/m) * (m*dA - np.sum(dA_channels, axis=0) - t**2 * (prev_A-batch_mean) * np.sum(dA_channels * (prev_A - batch_mean).reshape(-1, dA.shape[3]), axis=0))
        
        return curr_dA, dgamma, dbeta


class Conv():
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
        
        self.im2col_indices = None
                        
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
        """
        Convolution of layer using im2col.
    
        Parameters
        ----------
        x : ndarray
            Input data of shape (N, H, W, C).
        w : ndarray
            Filter weights of shape (FH, FW, C, out_channels).
        b : ndarray
            Biases of shape (1, out_channels).
        stride : int, optional
            Stride for the convolution.
        padding : int, optional
            Number of pixels to pad around the input height and width.
        activation : function or None, optional
            Activation function to apply elementwise. If None, no nonlinearity is applied.
    
        Returns
        -------
        out : ndarray
            Output data of shape (N, H_out, W_out, out_channels) where
            H_out = (H + 2*padding - FH)//stride + 1 and 
            W_out = (W + 2*padding - FW)//stride + 1.
        
        """
        
        N, H_prev, W_prev, C_prev = A.shape

        H_out = (H_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        A_col = utils.im2col(A, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding, indices=self.im2col_indices)
        
        w_col = self.w.reshape(-1, self.out_channels)
        # Perform matrix multiplication.
        out_col = np.matmul(w_col.T, A_col) + self.b.T
        # Reshape back matrix to image.
        out_convoluted = out_col.T.reshape(N, H_out, W_out, self.out_channels)
        
        output = self.activation_function.value(out_convoluted)
        
        if self.flatten:
            output = output.reshape(output.shape[0], -1)
                                            
        return output, (A_col, out_convoluted)
    
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
        if self.padding=="same":
            self.padding = math.floor(1/2 * (self.kernel_size + (self.stride - 1) * (prev_A.shape[1] - 1)))
        
        if self.im2col_indices is None:
            self.im2col_indices = utils.im2col_indices(prev_A.shape, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding)

        curr_A, (A_col, A_convoluted) = self.value(prev_A, training=training)
        cache = (prev_A, A_col, A_convoluted)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        """
        Backward pass for a convolutional layer using im2col and col2im.
    
        Parameters
        ----------
        dout : ndarray
            Upstream gradients of shape (N, H_out, W_out, out_channels).
        cache : tuple
            Values from the forward pass (x, w, b, stride, padding, X_col).
    
        Returns
        -------
        dx : ndarray
            Gradient with respect to the input, of shape (N, H, W, C).
        dw : ndarray
            Gradient with respect to the weights, of shape (FH, FW, C, out_channels).
        db : ndarray
            Gradient with respect to the biases, of shape (1, out_channels).
        """
        
        A_prev, A_col, A_convoluted = cache
        N, H, W, C = A_prev.shape
        
        if self.flatten:
            dA = dA.reshape(dA.shape[0], (A_prev.shape[1] + 2*self.padding - self.kernel_size)//self.stride + 1, (A_prev.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1, self.out_channels)
        
        d_activ = self.activation_function.derivative(A_convoluted)
        dA = d_activ * dA
    
        # Reshape upstream gradients:
        # dout is of shape (N, H_out, W_out, out_channels); convert it into 
        # shape (out_channels, N*H_out*W_out) for easier matrix multiplication.
        dA_reshaped = dA.transpose(0, 3, 1, 2).reshape(self.out_channels, -1)
    
        # Gradient with respect to bias: sum over all spatial locations and examples.
        db = np.sum(dA, axis=(0, 1, 2)).reshape(1, -1)
    
        # Gradient with respect to weights.
        # Compute dW_col = X_col dot (dout reshaped transposed), then reshape.
        dw_col = A_col.dot(dA_reshaped.T)  # shape: (FH*FW*C, out_channels)
        dw = dw_col.reshape(self.w.shape)
    
        # Gradient with respect to the input:
        # Compute dX_col = weights (reshaped) dot dout_reshaped.
        w_col = self.w.reshape(-1, self.out_channels)  # shape: (FH*FW*C, out_channels)
        dA_col = np.matmul(w_col, dA_reshaped) #w_col.dot(dA_reshaped)      # shape: (FH*FW*C, N*H_out*W_out)
        
        # Use col2im to reshape dX_col back to input shape.
        curr_dA = utils.col2im(dA_col, A_prev.shape, self.kernel_size, self.kernel_size, self.stride, self.padding, indices = self.im2col_indices)
            
        return curr_dA, dw, db


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
        
        self.trainable = False
                
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
        A : ndarray, shape: (N, H, W, C)
            Input images. N is the number of images, H and W are the height and width of 
            each image and C is the number of channels of each image.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for pooling 
            Layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: (N, H, W, C)
            Input images with the pooling applied.
        cache : tuple
            Cache containing information required in backward step. The cache contains:
                A : ndarray, shape: (N, H, W, C)
                    Input images.
                A_cols : ndarray, shape: (C * field_height * field_width, N * H_out * W_out)
                    Column representation of each window in A. The field_height and field_width are the kernel 
                    dimensions and  out_height and out_width the dimensions of each image in the output given by:
                        out_height = (H + 2*padding - field_height)//stride + 1)
                        out_width = (W + 2*padding - field_width)//stride + 1
                max_idx : ndarray/None, shape: (C, N * out_height * out_width)
                    Position of the maximum value in each column at dimension 1 of A_cols_reshaped.

        '''
        N, H, W, C = A.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
    
        # Use im2col to extract all patches.
        # im2col should return an array of shape:
        #     (C * kernel_size * kernel_size, N * H_out * W_out)
        A_cols = utils.im2col(A, self.kernel_size, self.kernel_size, self.stride, self.padding)
        
        # Reshape so that we group elements for each channel separately.
        # New shape: (C, kernel_size*kernel_size, N * H_out * W_out)
        A_cols_reshaped = A_cols.reshape(C, self.kernel_size * self.kernel_size, -1)
                
        if self.pool_mode == "max":
            # For each channel and each patch, take the maximum.
            out_cols = np.max(A_cols_reshaped, axis=1)  # shape: (C, N * H_out * W_out)
            # Record which index within each patch was maximal (for backprop).
            max_idx = np.argmax(A_cols_reshaped, axis=1)  # shape: (C, N * H_out * W_out)
        elif self.pool_mode == "avg":
            out_cols = np.mean(A_cols_reshaped, axis=1)   # shape: (C, N * H_out * W_out)
            max_idx = None
        else:
            raise ValueError("pool_mode must be either 'max' or 'avg'")
        
        # Reshape out_cols to (C, N, H_out, W_out) and then transpose to (N, H_out, W_out, C)
        A_curr = out_cols.reshape(C, N, H_out, W_out).transpose(1, 2, 3, 0)
        
        cache = (A, A_cols, max_idx)
        
        return A_curr, cache
    
    def forward(self, prev_A, training=False):
        '''
        Forward pass for a pooling Layer. This is evaluated using im2col and col2im algorithms 
        from OrionML.utils.

        Parameters
        ----------
        prev_A : ndarray, shape: (N, H, W, C)
            Input images. N is the number of images, H and W are the height and width of 
            each image and C is the number of channels of each image.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for pooling 
            Layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: (N, H, W, C)
            Input images with the pooling applied.
        cache : tuple
            Cache containing information required in backward step. The cache contains:
                A : ndarray, shape: (N, H, W, C)
                    Input images.
                A_cols : ndarray, shape: (C * field_height * field_width, N * H_out * W_out)
                    Column representation of each window in A. The field_height and field_width are the kernel 
                    dimensions and  out_height and out_width the dimensions of each image in the output given by:
                        out_height = (H + 2*padding - field_height)//stride + 1)
                        out_width = (W + 2*padding - field_width)//stride + 1
                max_idx : ndarray/None, shape: (C, N * out_height * out_width)
                    Position of the maximum value in each column at dimension 1 of A_cols_reshaped.

        '''
        A_curr, cache = self.value(prev_A, training=training)
        return A_curr, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward pass for a pooling Layer. This is evaluated using im2col and col2im algorithms 
        from OrionML.utils.

        Parameters
        ----------
        dA : ndarray, shape: (N, H_out, W_out, C)
            Upstream gradient.
        training : bool, optional
            Whether the Layer is currently in training or not. This has no effect for pooling 
            Layers. The default is False.

        Returns
        -------
        dA_curr : ndarray, shape: (input size, output size)
            Gradient with respect to the input A.

        '''        
        A, A_cols, max_idx = cache
        N, H, W, C = A.shape
        
        # Initialize gradient in column shape: same shape as A_cols.
        dA_curr_cols = np.zeros_like(A_cols)  # shape: (C * kernel_size * kernel_size, N * H_out * W_out)
        
        # For backpropagation, we reshape A_cols so that pooling regions (patches) are grouped per channel.
        dA_curr_cols_reshaped = dA_curr_cols.reshape(C, self.kernel_size * self.kernel_size, -1)  # shape: (C, kernel_size*kernel_size, N*H_out*W_out)
        
        # Reshape upstream gradient: (N, H_out, W_out, C) --> (C, N*H_out*W_out)
        dA_reshaped = dA.transpose(3, 0, 1, 2).reshape(C, -1)
        
        if self.pool_mode == "max":
            # For max pooling, distribute upstream gradient only to the positions that had the maximum.
            # For each channel c and each patch (column), set the gradient at the max index.
            for c in range(C):
                # Here, max_idx[c] is a 1D array of length (N * H_out * W_out)
                # Use np.arange to index the patch dimension.
                dA_curr_cols_reshaped[c, max_idx[c], np.arange(dA_reshaped.shape[1])] = dA_reshaped[c]
                
        elif self.pool_mode == "avg":
            # For average pooling, distribute the gradient evenly over all elements in the pooling window.
            dA_curr_cols_reshaped += (dA_reshaped[:, None, :] / (self.kernel_size * self.kernel_size))
            
        else:
            raise ValueError("pool_mode must be either 'max' or 'avg'")
        
        # Flatten dA_cols_reshaped back to (C * kernel_size * kernel_size, N * H_out * W_out)
        dA_curr_cols = dA_curr_cols_reshaped.reshape(C * self.kernel_size * self.kernel_size, -1)
        
        # Use col2im to convert the column representation back to the image shape.
        dA_curr = utils.col2im(dA_curr_cols, A.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        
        return dA_curr


class Reshape():
    def __init__(self, output_shape):
        '''
        Layer to reshape data. Can be used if e.g. a linear layer follows a pooling layer.

        Parameters
        ----------
        output_shape : tuple
            Shape of the output data.

        '''
        self.input_shape = None
        self.output_shape = output_shape
                
        self.trainable = False
                
    def type(self):
        '''

        Returns
        -------
        str
            String unique to Reshape layers.

        '''
        return "OrionML.Layer.Reshape"
    
    def description(self):
        '''

        Returns
        -------
        str
            Description of the Reshape layer with information about the output and input shapes.

        '''
        return f"OrionML.Layer.Reshape (input shape: {self.input_shape}, output shape: {self.output_shape})"
    
    def value(self, A, training=False):
        '''
        Reshape the input.
        
        Parameters
        ----------
        A : ndarray, shape: self.input_shape
            Input Data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for Reshape 
            layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: self.output_data
            Reshaped input data.
        cache : tuple
            Empty tuple since no additional information is required for backwards propagation.

        '''
        assert A.shape[1:] == self.input_shape, "Shape of upstream gradient passed to forwards propagation in Reshape layer does not match the predefined input shape."
        
        A_curr = A.reshape(-1, *self.output_shape)
        
        cache = ()
        
        return A_curr, cache
    
    def forward(self, prev_A, training=False):
        '''
        Forward pass for a Reshape layer.

        Parameters
        ----------
        A : ndarray, shape: self.input_shape
            Input Data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for Reshape 
            layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: self.output_data
            Reshaped input data.
        cache : tuple
            Empty tuple since no additional information is required for backwards propagation.

        '''
        self.input_shape = prev_A.shape[1:]
        
        A_curr, cache = self.value(prev_A, training=training)
        return A_curr, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward pass for a Reshape Layer.

        Parameters
        ----------
        dA : ndarray, shape: self.output_shape
            Upstream gradient.
        training : bool, optional
            Whether the Layer is currently in training or not. This has no effect for Reshape 
            layers. The default is False.

        Returns
        -------
        dA_curr : ndarray, shape: self.input_shape
            Reshaped upstream gradient.

        '''    
        assert dA.shape[1:] == self.output_shape, "Shape of upstream gradient passed to backwards propagation in Reshape layer does not match the predefined output shape."
        
        dA_curr = dA.reshape(-1, *self.input_shape)
        
        return dA_curr


class Flatten():
    def __init__(self):
        '''
        Layer to flatten data. Can be used if e.g. a linear layer follows a pooling layer.
        

        '''
                
        self.trainable = False
        
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to Flatten layers.

        '''
        return "OrionML.Layer.Flatten"
    
    def description(self):
        '''

        Returns
        -------
        str
            Description of the Flatten layer.

        '''
        return "OrionML.Layer.Flatten"
    
    def value(self, A, training=False):
        '''
        Flatten the input.
        
        Parameters
        ----------
        A : ndarray, shape: self.input_shape
            Input Data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for Flatten 
            layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: self.output_data
            Reshaped input data.
        cache : tuple
            Tuple containing the shape of the input data starting from the first dimension.

        '''        
        A_curr = A.reshape((A.shape[0], -1))
        
        cache = (A.shape[1:])
        
        return A_curr, cache
    
    def forward(self, prev_A, training=False):
        '''
        Forward pass for a Flatten layer.

        Parameters
        ----------
        A : ndarray, shape: self.input_shape
            Input Data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for Flatten 
            layers. The default is False.

        Returns
        -------
        A_curr : ndarray, shape: self.output_data
            Reshaped input data.
        cache : tuple
            Tuple containing the shape of the input data starting from the first dimension.

        '''
        A_curr, cache = self.value(prev_A, training=training)
        return A_curr, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward pass for a Flatten Layer.

        Parameters
        ----------
        dA : ndarray, shape: self.output_shape
            Upstream gradient.
        training : bool, optional
            Whether the Layer is currently in training or not. This has no effect for Reshape 
            layers. The default is False.

        Returns
        -------
        dA_curr : ndarray, shape: self.input_shape
            Unflattened upstream gradient.

        '''    
        input_shape = cache
        dA_curr = dA.reshape((-1, *input_shape))
        
        return dA_curr























































































