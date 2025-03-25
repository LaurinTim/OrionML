import numpy as np

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ

class Linear():
    
    def __init__(self, dim1, dim2, activation, bias=True):
        '''
        Linear layer for deep learning.

        Parameters
        ----------
        dim1 : int
            Size of the imput sample.
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
        
        if activation == "linear": self.activation_function = activ.linear()
        elif activation == "relu": self.activation_function = activ.relu()
        elif activation == "elu": self.activation_function = activ.elu()
        elif activation == "leakyrelu": self.activation_function = activ.leakyrelu()
        elif activation == "softplus": self.activation_function = activ.softplus()
        elif activation == "sigmoid": self.activation_function = activ.sigmoid()
        elif activation == "tanh": self.activation_function = activ.tanh()
        elif activation == "softmax": self.activation_function = activ.softmax()
        else:
            print("Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.")