import numpy as np
import math
import copy
from time import time
import numba as nb

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
        out = self.activation_function.value(z)
                    
        return out, z
    
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
        curr_A, Z = self.value(prev_A, training=training)
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
        if training==False:
            return activation_output, np.zeros(1)
        
        mask = np.random.rand(*activation_output.shape) > self.dropout_probability
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
        st1 = time()
        
        N, H_prev, W_prev, C_prev = A.shape

        H_out = (H_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        A_col = utils.im2col(A, self.kernel_size, self.kernel_size, self.stride, self.padding)
        
        w_col = self.w.reshape(-1, self.out_channels)
        # Perform matrix multiplication.
        out_col = np.matmul(w_col.T, A_col) + self.b.T
        # Reshape back matrix to image.
        out_convoluted = out_col.T.reshape(N, H_out, W_out, self.out_channels)
        
        output = self.activation_function.value(out_convoluted)
        
        if self.flatten:
            output = output.reshape(output.shape[0], -1)
                        
        if training == True: self.t1 += time()-st1
        
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
        curr_dA = utils.col2im(dA_col, A_prev.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
    
        return curr_dA, dw, db
    
# %%

if __name__ == "__main__":
    a = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    c = Conv(in_channels=2, out_channels=3, kernel_size=2, activation="linear")
    
    ac, ca = c.forward(a)
    ac1, ca1 = c.forward1(a)
    
    dac, dw, db = c.backward(np.ones_like(ac), ca)
    dac1, dw1, db1 = c.backward1(np.ones_like(ac1), ca1)
    
    #d = Dropout(dropout_probability=0.5)
    
    #ad = d.value(a, training=True)

# %%

if __name__ == "__main__":
    in_channels = 3
    out_channels = 128
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = (4, in_channels, 12, 10)  # expected input size
    dout_size = (4, out_channels, 6, 5)   # expected to match the size of the layer outputs

    np.random.seed(42)  # for reproducibility

    x = np.random.random(batch_size)  # create data for forward pass
    dout = np.random.random(dout_size)  # create random data for backward
    
    xl = np.transpose(x, axes=(0,2,3,1))
    doutl = np.transpose(dout, axes=(0,2,3,1))
    '''
    print('x: ', x.shape)
    print('d_out: ', dout.shape)'''
    
    conv = Conv(in_channels, out_channels, kernel_size, stride, padding)
    
    conv_outl, c = conv.forward(xl)
    
# %%

if __name__ == "__main__":
    dxl, dbl, dwl = conv.derivative(doutl, c)
    '''
    print('conv_out: ', conv_outl.shape)
    print('db: ', dbl.shape)
    print('dw: ', dwl.shape)
    print('dx: ', dxl.shape)'''
    
# %%

if __name__ == "__main__":
    a = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    dal = np.ones((2, 2, 2, 3))
    
    l = Conv(2, 3, 2, "linear")
    l.w = np.ones((2,2,2,3))
    arl, c = l.forward(a)
    darl = l.backward(dal, c)
    
# %%

if __name__ == "__main__":
    #2 samples, 2 channels, height=width=3
    a = np.array([[[[2,0,2],
                    [3,3,1],
                    [4,1,3]],
                   [[5,2,0],
                    [0,1,4],
                    [4,0,4]]],
                  
                  [[[5,2,3],
                    [0,0,2],
                    [4,5,2]],
                   [[1,4,4],
                    [3,5,0],
                    [0,2,1]]]])
    
    a = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    dal = np.ones((2, 2, 2, 3))
    
    ns = 2
    oc = 2
    nc = 3
    ks = 2
    st = 1
    p = 0
    nh = (a.shape[2] + 2*p - ks)/st + 1
    nw = (a.shape[3] + 2*p - ks)/st + 1
    
    w = np.ones((ks, ks, oc, nc))
    #w[0][0][0][0] = 0
    ww = w*np.array([[[[1,10,100],[1,10,100]],[[1,10,100],[1,10,100]]],[[[1,10,100],[1,10,100]],[[1,10,100],[1,10,100]]]])
    '''w[:,0,0,1]=0
    w[:,0,1,0]=0
    w[:,1,0,0]=0
    w[:,1,1,1]=0'''
    
    b = np.zeros((nc, int(nh), int(nw)))
    
# %%

if __name__ == "__main__":
    Hout = a.shape[1] - w.shape[0] + 1
    Wout = a.shape[2] - w.shape[1] + 1
    
    aa = np.lib.stride_tricks.as_strided(a, (a.shape[0], Hout, Wout, w.shape[0], w.shape[1], a.shape[3]), a.strides[:3] + a.strides[1:])
    
    r = np.einsum('abcijk,ijkd', aa, w)
    rr = np.tensordot(aa, w, axes=3)
    
# %%

if __name__ == "__main__":
    l = Conv(2, 3, 2)
    l.w = w
    rl, c = l.value(a)
    drl = l.derivative(dal, c)
    
# %%

def value1(self, A, training=False):
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
 # =============================================================================
 #       As a reminder: ndarray.strides gives the number of bytes to step until the next element is reached in each dimension. Each number in the arrays is of type np.float64, the last 
 #       Dimension of A.strides will be 64/8=4. 
 #       For the array np.array([[0,1,2],[3,4,5]]) b.strides is (12, 4) since each number is a 32 bit integer and thus there are 4 bytes for each number.
 #       The first dimension is filled with three 32 bit integers and thus the stride for the first dimension is 3*4=12.
 #       For the array np.array([[0,1,2],[3,4,5]], dtype=float) b.strides is (24, 8) since each number is a 64 bit float and thus there are 8 bytes for each number.
 #       The first dimension is filled with three 64 bit floats and thus the stride for the first dimension is 3*8=24. 
 # =============================================================================
         A_num_dims = A.ndim
         
         assert A_num_dims == 4, "Input to pooling layer has wrong number of dimensions. A needs 4 dimensions."
         
         A = np.pad(A, self.padding, mode="constant")
         
         output_shape = (A.shape[0], A.shape[1], (A.shape[2] + 2*self.padding - self.kernel_size) // self.stride + 1, (A.shape[3] + 2*self.padding - self.kernel_size) // self.stride + 1)
         w_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3], self.kernel_size, self.kernel_size)
         w_strides = (A.strides[0], A.strides[1], self.stride*A.strides[2], self.stride*A.strides[3], A.strides[2], A.strides[3])
         
         A_w = np.lib.stride_tricks.as_strided(A, w_shape, w_strides)
 
         if self.pool_mode=="max":
             A_w = A_w.max(axis=(4, 5))
         elif self.pool_mode=="avg":
             A_w = A_w.mean(axis=(4, 5))
         
         return A_w

# %%

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

# %%

if __name__ == "__main__":
    a = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    l = Pool(kernel_size=2, stride=1, padding=0, pool_mode="max")
    
    av, c = l.value(a)
    
    ad = l.backward(np.ones_like(av), c)
























































































