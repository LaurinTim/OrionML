import numpy as np
import math
import copy

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ

class Linear():
    
    def __init__(self, dim1, dim2, activation, bias=True):
        '''
        Linear layer.

        Parameters
        ----------
        dim1 : int
            Size of the input sample.
        dim2 : int
            Size of the output sample.
        activation : str
            Activation to use for this layer. The available activations are: 
            {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.
        bias : TYPE, optional
            Whether or not the layer contains a bias. The default is True.

        Returns
        -------
        None.

        '''
        self.dim1 = dim1
        self.dim2 = dim2
        self.activation = activation
        self.bias = bias
        self.w = np.random.rand(dim1, dim2) * 1e-3
        self.b = np.random.rand(1, dim2) if bias==True else np.zeros(1, dim2)
        
        self.activation_function = self.get_activation_function()
        
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
        elif self.activation == "leakyrelu": return activ.leakyrelu()
        elif self.activation == "softplus": return activ.softplus()
        elif self.activation == "sigmoid": return activ.sigmoid()
        elif self.activation == "tanh": return activ.tanh()
        elif self.activation == "softmax": return activ.softmax()
        else:
            print("Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.")
            
class Dropout():
    def __init__(self, wa, dropout_probability=0.3, scale=True):
        '''
        Dropout layer.

        Parameters
        ----------
        wa : ndarray, shape: (input size, output size)
            Activated weights to which dropout should be applied.
        dropout_probability : TYPE, optional
            DESCRIPTION. The default is 0.3.
        scale : bool, optional
            Whether the remaining weights should be scaled by 1/dropout_probability.
            The default is True.

        Returns
        -------
        None.

        '''
        self.wa = wa
        self.dropout_probability = dropout_probability
        self.set_zero()
        
    def set_zero(self) -> None:
        '''
        Set each element in wa with probability dropout_probability to 0.

        '''
        mask = np.random.rand(self.wa.shape[0], self.wa.shape[1]) > self.dropout_probability
        self.wa = mask*self.wa





























































