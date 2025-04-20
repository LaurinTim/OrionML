import numpy as np
import math
import copy
from time import time

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
        
        self.in_dim = None
        self.out_dim = None
        
        self.buffers = {}
                
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
            
    def init_buffers(self, batch_size):
        self.buffers["z"] = np.empty((batch_size, *self.out_dim))
        self.buffers["A_prev"] = np.empty((batch_size, *self.in_dim))
        
        if self.activation == "softmax":
            self.buffers["d_activation"] = np.empty((batch_size, *self.out_dim, *self.out_dim))
            
        else:
            self.buffers["d_activation"] = np.empty((batch_size, *self.out_dim))
            
        self.buffers["dA_activation"] = np.empty((batch_size, *self.out_dim))
        
        self.activation_function.init_buffers(batch_size, self.out_dim)
        
    
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
                    
        return out
    
    def forward(self, A_prev, out_buffer, training=False):
        '''
        Forward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        A_prev : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
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
        np.copyto(self.buffers["A_prev"], A_prev)
        z_buffer = self.buffers["z"]
        
        np.matmul(self.buffers["A_prev"], self.w, out=z_buffer)
        np.add(z_buffer, self.b, out=z_buffer)
        self.activation_function.value_buffered(z_buffer, out_buffer=out_buffer)
        
        return
    
    def backward(self, dA, dw_buffer, db_buffer, dout_buffer, training=False):
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
        z_buffer = self.buffers["z"]
        d_activation_buffer = self.buffers["d_activation"]
        dA_activation_buffer = self.buffers["dA_activation"]
        A_prev_buffer = self.buffers["A_prev"]
        
        self.activation_function.derivative_buffered(z_buffer, out_buffer=d_activation_buffer)
                
        if self.activation != "softmax":
            np.multiply(dA, d_activation_buffer, out=dA_activation_buffer)
            
        else:
            np.einsum('ijk,ik->ij', d_activation_buffer, dA, optimize="optimal", out=dA_activation_buffer)
                        
        np.matmul(A_prev_buffer.T, dA_activation_buffer, out=dw_buffer)
        np.sum(dA_activation_buffer, axis=0, keepdims=True, out=db_buffer)
        np.matmul(dA_activation_buffer, self.w.T, out=dout_buffer)
                
        return
    
       
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
        
        self.in_dim = None
        self.out_dim = None
        
        self.buffers = {}
        
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
    
    def init_buffers(self, batch_size):
        '''
        Allocate mask and output buffers once shapes are known.
        '''
        self.buffers["mask"] = np.empty((batch_size, *self.in_dim), dtype=bool)
        self.buffers["out"]  = np.empty((batch_size, *self.in_dim), dtype=float)
        self.buffers["random"]  = np.empty((batch_size, *self.in_dim), dtype=float)
        
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
        return activation_output
    
    def forward(self, A_prev, out_buffer, training=False):
        '''
        Forward step of a dropout Layer in a Neural Network.

        Parameters
        ----------
        A_prev : ndarray, shape: (input size, output size)
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
        random_buffer = self.buffers["random"]
        mask = self.buffers["mask"]
        
        random_buffer = np.random.rand(*A_prev.shape)
        np.greater(random_buffer, self.dropout_probability, out=mask)
                
        np.multiply(A_prev, mask, out=out_buffer)
                
        if self.scale:
            np.divide(out_buffer, 1.0 - self.dropout_probability, out=out_buffer)
                                
        return
    
    def backward(self, dA, dout_buffer, training=False):
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
        mask = self.buffers["mask"]
        np.multiply(dA, mask, out=dout_buffer)
                
        if self.scale:
            np.divide(dout_buffer, 1.0 - self.dropout_probability, out=dout_buffer)
                        
        return

    
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
        
        self.in_dim = None
        self.out_dim = None
        
        self.buffers = {}
        
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
    
    def init_buffers(self, batch_size):
        self.buffers["A_prev"] = np.empty((batch_size, *self.in_dim))
        self.buffers["mean"] = np.empty((1, *self.out_dim))
        self.buffers["variance"] = np.empty((1, *self.out_dim))
        self.buffers["A_shifted"] = np.empty((batch_size, *self.in_dim))
        self.buffers["add1"] = np.empty((1, *self.in_dim))
        self.buffers["denom"] = np.empty((1, *self.in_dim))
        self.buffers["A_normalized"] = np.empty((batch_size, *self.in_dim))
        self.buffers["mult1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["mult2"] = np.empty((1, *self.in_dim))
        self.buffers["mult3"] = np.empty((1, *self.in_dim))
        
        self.buffers["dmult1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dadd1"] = np.empty((1, *self.in_dim))
        self.buffers["ddenom"] = np.empty((1, *self.in_dim))
        self.buffers["t"] = np.empty((1, *self.in_dim))
        self.buffers["ddiv1"] = np.empty((1, *self.in_dim))
        self.buffers["dmult2"] = np.empty((1, *self.in_dim))
        self.buffers["dmult3"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsum1"] = np.empty((1, *self.in_dim))
        self.buffers["dsub1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["t_squared"] = np.empty((1, *self.in_dim))
        self.buffers["dsub2"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dmult4"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsub3"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dmult5"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsum2"] = np.empty((1, *self.in_dim))
        self.buffers["dmult6"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsub4"] = np.empty((batch_size, *self.in_dim))
        
        
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
        x_normalized = (x-self.running_mean)/np.sqrt(self.running_variance + self.epsilon)
        out = self.gamma*x_normalized + self.beta
                    
        return out
    
    def forward(self, A_prev, out_buffer, training=False):
        '''
        Forward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        A_prev : ndarray, shape: (number of samples, number of features)
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
        np.copyto(self.buffers["A_prev"], A_prev)
        A_prev_buffer = self.buffers["A_prev"]
        
        if training==True:
            mean_buffer = self.buffers["mean"]
            variance_buffer = self.buffers["variance"]
            A_shifted_buffer = self.buffers["A_shifted"]
            add1_buffer = self.buffers["add1"]
            denom_buffer = self.buffers["denom"]
            A_normalized_buffer = self.buffers["A_normalized"]
            mult1_buffer = self.buffers["mult1"]
            mult2_buffer = self.buffers["mult2"]
            mult3_buffer = self.buffers["mult3"]
            
            np.mean(A_prev_buffer, axis=0, keepdims=True, out=mean_buffer)
            np.var(A_prev_buffer, axis=0, keepdims=True, out=variance_buffer)
            np.subtract(A_prev_buffer, mean_buffer, out=A_shifted_buffer)
            np.add(variance_buffer, self.epsilon, out=add1_buffer)
            np.sqrt(add1_buffer, out=denom_buffer)
            np.divide(A_shifted_buffer, denom_buffer, out=A_normalized_buffer)
            
            np.multiply(self.gamma, A_normalized_buffer, out=mult1_buffer)
            np.add(mult1_buffer, self.beta, out=out_buffer)
            
            np.multiply(self.momentum, self.running_mean, out=mult2_buffer)
            np.add(mult2_buffer, (1-self.momentum) * mean_buffer, out=self.running_mean)
            
            np.multiply(self.momentum, self.running_variance, out=mult3_buffer)
            np.add(mult3_buffer, (1-self.momentum) * variance_buffer, out=self.running_variance)
        
        return
    
    def backward(self, dA, dgamma_buffer, dbeta_buffer, dout_buffer, training=False):
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
        
        A_normalized_buffer = self.buffers["A_normalized"]
        mean_buffer = self.buffers["mean"]
        variance_buffer = self.buffers["variance"]
        A_prev_buffer = self.buffers["A_prev"]
        dmult1_buffer = self.buffers["dmult1"]
        dadd1_buffer = self.buffers["dadd1"]
        ddenom_buffer = self.buffers["ddenom"]
        t_buffer = self.buffers["t"]
        ddiv1_buffer = self.buffers["ddiv1"]
        dmult2_buffer = self.buffers["dmult2"]
        dmult3_buffer = self.buffers["dmult3"]
        dsum1_buffer = self.buffers["dsum1"]
        dsub1_buffer = self.buffers["dsub1"]
        t_squared_buffer = self.buffers["t_squared"]
        dsub2_buffer = self.buffers["dsub2"]
        dmult4_buffer = self.buffers["dmult4"]
        dsub3_buffer = self.buffers["dsub3"]
        dmult5_buffer = self.buffers["dmult5"]
        dsum2_buffer = self.buffers["dsum2"]
        dmult6_buffer = self.buffers["dmult6"]
        dsub4_buffer = self.buffers["dsub4"]
        
        np.multiply(dA, A_normalized_buffer, out=dmult1_buffer)
        np.add.reduce(dmult1_buffer, axis=0, keepdims=True, out=dgamma_buffer)
        np.add.reduce(dA, axis=0, keepdims=True, out=dbeta_buffer)
        
        m = A_prev_buffer.shape[0]
        
        np.add(variance_buffer, self.epsilon, out=dadd1_buffer)
        np.sqrt(dadd1_buffer, out=ddenom_buffer)
        np.divide(1, ddenom_buffer, out=t_buffer)
        np.divide(t_buffer, m, out=ddiv1_buffer)
                
        np.multiply(self.gamma, ddiv1_buffer, out=dmult2_buffer)
        np.multiply(m, dA, out=dmult3_buffer)
        np.add.reduce(dA, axis=0, keepdims=True, out=dsum1_buffer)
        np.subtract(dmult3_buffer, dsum1_buffer, out=dsub1_buffer)
        np.square(t_buffer, out=t_squared_buffer)
        np.subtract(A_prev_buffer, mean_buffer, out=dsub2_buffer)
        np.multiply(t_squared_buffer, dsub2_buffer, out=dmult4_buffer)
        np.subtract(A_prev_buffer, mean_buffer, out=dsub3_buffer)
        np.multiply(dA, dsub3_buffer, out=dmult5_buffer)
        np.add.reduce(dmult5_buffer, axis=0, keepdims=True, out=dsum2_buffer)
        np.multiply(dmult4_buffer, dsum2_buffer, out=dmult6_buffer)
        np.subtract(dsub1_buffer, dmult6_buffer, out=dsub4_buffer)
                
        np.multiply(dmult2_buffer, dsub4_buffer, out=dout_buffer)
        
        #prev_A, x_normalized, batch_mean, batch_variance = cache
        
        #dgamma = np.sum(dA*x_normalized, axis=0, keepdims=True)
        #dbeta = np.sum(dA, axis=0, keepdims=True)
        
        #m = prev_A.shape[0]
        #t = 1/np.sqrt(batch_variance + self.epsilon)
        #curr_dA = (self.gamma * t/m) * (m*dA - np.sum(dA, axis=0) - t**2 * (prev_A-batch_mean) * np.sum(dA * (prev_A - batch_mean), axis=0))
        
        return


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
        
        self.in_dim = None
        self.out_dim = None
        
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
    
    def init_buffers(self, batch_size):
        self.buffers["A_prev"] = np.empty((batch_size, *self.in_dim))
        self.buffers["A_prev_cols"] = np.empty((batch_size * self.in_dim[0] * self.in_dim[1], self.in_dim[2]))
        self.buffers["mean"] = np.empty((1, *self.out_dim))
        self.buffers["variance"] = np.empty((1, *self.out_dim))
        self.buffers["A_shifted"] = np.empty((batch_size, *self.in_dim))
        self.buffers["add1"] = np.empty((1, *self.in_dim))
        self.buffers["denom"] = np.empty((1, *self.in_dim))
        self.buffers["A_normalized"] = np.empty((batch_size, *self.in_dim))
        self.buffers["mult1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["mult2"] = np.empty((1, *self.in_dim))
        self.buffers["mult3"] = np.empty((1, *self.in_dim))
        
        self.buffers["dA_prev_cols"] = np.empty((batch_size * self.in_dim[0] * self.in_dim[1], self.in_dim[2]))
        self.buffers["dmult1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dadd1"] = np.empty((1, *self.in_dim))
        self.buffers["ddenom"] = np.empty((1, *self.in_dim))
        self.buffers["t"] = np.empty((1, *self.in_dim))
        self.buffers["ddiv1"] = np.empty((1, *self.in_dim))
        self.buffers["dmult2"] = np.empty((1, *self.in_dim))
        self.buffers["dmult3"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsum1"] = np.empty((1, *self.in_dim))
        self.buffers["dsub1"] = np.empty((batch_size, *self.in_dim))
        self.buffers["t_squared"] = np.empty((1, *self.in_dim))
        self.buffers["dsub2"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dmult4"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsub3"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dmult5"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsum2"] = np.empty((1, *self.in_dim))
        self.buffers["dmult6"] = np.empty((batch_size, *self.in_dim))
        self.buffers["dsub4"] = np.empty((batch_size, *self.in_dim))
        
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
        x_normalized = (x-self.running_mean)/np.sqrt(self.running_variance + self.epsilon)
        out = self.gamma*x_normalized + self.beta
            
        return out
    
    def forward(self, A_prev, out_buffer, training=False):
        '''
        Forward step of a 2D Batch normalization layer in a Neural Network.

        Parameters
        ----------
        A_prev : ndarray, shape: (number of samples, input height, input width, input channels)
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
        #In Batch normalization for convolutional layers, the normalization should be performed over each channel
        np.copyto(self.buffers["A_prev"], A_prev)
        A_prev_buffer = self.buffers["A_prev"]
        A_prev_cols_buffer = self.buffers["A_prev_cols"]
        
        if training==True:
            mean_buffer = self.buffers["mean"]
            variance_buffer = self.buffers["variance"]
            A_shifted_buffer = self.buffers["A_shifted"]
            add1_buffer = self.buffers["add1"]
            denom_buffer = self.buffers["denom"]
            A_normalized_buffer = self.buffers["A_normalized"]
            mult1_buffer = self.buffers["mult1"]
            mult2_buffer = self.buffers["mult2"]
            mult3_buffer = self.buffers["mult3"]
            
            A_prev_cols_buffer = A_prev_buffer.copy().reshape((-1, A_prev_buffer.shape[3]))
            
            np.mean(A_prev_cols_buffer, axis=0, keepdims=True, out=mean_buffer)
            np.var(A_prev_cols_buffer, axis=0, keepdims=True, out=variance_buffer)
            np.subtract(A_prev_cols_buffer, mean_buffer, out=A_shifted_buffer)
            np.add(variance_buffer, self.epsilon, out=add1_buffer)
            np.sqrt(add1_buffer, out=denom_buffer)
            np.divide(A_shifted_buffer, denom_buffer, out=A_normalized_buffer)
            
            np.multiply(self.gamma, A_normalized_buffer, out=mult1_buffer)
            np.add(mult1_buffer, self.beta, out=out_buffer)
            
            np.multiply(self.momentum, self.running_mean, out=mult2_buffer)
            np.add(mult2_buffer, (1-self.momentum) * mean_buffer, out=self.running_mean)
            
            np.multiply(self.momentum, self.running_variance, out=mult3_buffer)
            np.add(mult3_buffer, (1-self.momentum) * variance_buffer, out=self.running_variance)
        
        return
    
    def backward(self, dA, dgamma_buffer, dbeta_buffer, dout_buffer, training=False):
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
        
        dA_prev_cols_buffer = self.buffers("dA_prev_cols")
        A_normalized_buffer = self.buffers["A_normalized"]
        mean_buffer = self.buffers["mean"]
        variance_buffer = self.buffers["variance"]
        A_prev_cols_buffer = self.buffers["A_prev_cols"]
        dmult1_buffer = self.buffers["dmult1"]
        dadd1_buffer = self.buffers["dadd1"]
        ddenom_buffer = self.buffers["ddenom"]
        t_buffer = self.buffers["t"]
        ddiv1_buffer = self.buffers["ddiv1"]
        dmult2_buffer = self.buffers["dmult2"]
        dmult3_buffer = self.buffers["dmult3"]
        dsum1_buffer = self.buffers["dsum1"]
        dsub1_buffer = self.buffers["dsub1"]
        t_squared_buffer = self.buffers["t_squared"]
        dsub2_buffer = self.buffers["dsub2"]
        dmult4_buffer = self.buffers["dmult4"]
        dsub3_buffer = self.buffers["dsub3"]
        dmult5_buffer = self.buffers["dmult5"]
        dsum2_buffer = self.buffers["dsum2"]
        dmult6_buffer = self.buffers["dmult6"]
        dsub4_buffer = self.buffers["dsub4"]
        
        dA_prev_cols_buffer = dA.copy().reshape((-1, dA.shape[3]))
        
        np.multiply(dA_prev_cols_buffer, A_normalized_buffer, out=dmult1_buffer)
        np.add.reduce(dmult1_buffer, axis=0, keepdims=True, out=dgamma_buffer)
        np.add.reduce(dA_prev_cols_buffer, axis=0, keepdims=True, out=dbeta_buffer)
        
        m = dA_prev_cols_buffer.shape[0]
        
        np.add(variance_buffer, self.epsilon, out=dadd1_buffer)
        np.sqrt(dadd1_buffer, out=ddenom_buffer)
        np.divide(1, ddenom_buffer, out=t_buffer)
        np.divide(t_buffer, m, out=ddiv1_buffer)
                
        np.multiply(self.gamma, ddiv1_buffer, out=dmult2_buffer)
        np.multiply(m, dA_prev_cols_buffer, out=dmult3_buffer)
        np.add.reduce(dA_prev_cols_buffer, axis=0, keepdims=True, out=dsum1_buffer)
        np.subtract(dmult3_buffer, dsum1_buffer, out=dsub1_buffer)
        np.square(t_buffer, out=t_squared_buffer)
        np.subtract(A_prev_cols_buffer, mean_buffer, out=dsub2_buffer)
        np.multiply(t_squared_buffer, dsub2_buffer, out=dmult4_buffer)
        np.subtract(A_prev_cols_buffer, mean_buffer, out=dsub3_buffer)
        np.multiply(dA_prev_cols_buffer, dsub3_buffer, out=dmult5_buffer)
        np.add.reduce(dmult5_buffer, axis=0, keepdims=True, out=dsum2_buffer)
        np.multiply(dmult4_buffer, dsum2_buffer, out=dmult6_buffer)
        np.subtract(dsub1_buffer, dmult6_buffer, out=dsub4_buffer)
                
        np.multiply(dmult2_buffer, dsub4_buffer, out=dout_buffer)
        
        #prev_A, x_normalized, batch_mean, batch_variance = cache
        
        #dgamma = np.sum(dA*x_normalized, axis=0, keepdims=True)
        #dbeta = np.sum(dA, axis=0, keepdims=True)
        
        #m = prev_A.shape[0]
        #t = 1/np.sqrt(batch_variance + self.epsilon)
        #curr_dA = (self.gamma * t/m) * (m*dA - np.sum(dA, axis=0) - t**2 * (prev_A-batch_mean) * np.sum(dA * (prev_A - batch_mean), axis=0))
        
        return


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
        
        self.in_dim = None
        self.out_dim = None
                
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
        
        self.in_dim = None
        self.out_dim = None
        
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

# %%

class Reshape():
    def __init__(self, output_shape):
        '''
        Layer to reshape data. Can be used if e.g. a linear layer follows a pooling layer.

        Parameters
        ----------
        output_shape : tuple
            Shape of the output data.

        Returns
        -------
        None.

        '''
        self.input_shape = None
        self.output_shape = output_shape
                
        self.trainable = False
        
        self.in_dim = None
        self.out_dim = None
        
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

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data.
        output_shape : tuple
            Shape of the output data.

        Returns
        -------
        None.

        '''
                
        self.trainable = False
        
        self.in_dim = None
        self.out_dim = None
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to Reshape layers.

        '''
        return "OrionML.Layer.Flatten"
    
    def description(self):
        '''

        Returns
        -------
        str
            Description of the Reshape layer with information about the output and input shapes.

        '''
        return f"OrionML.Layer.Flatten"
    
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
        A_curr = A.reshape((A.shape[0], -1))
        
        cache = (A.shape[1:])
        
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
        input_shape = cache
        dA_curr = dA.reshape((-1, *input_shape))
        
        return dA_curr























































































