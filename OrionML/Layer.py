import numpy as np
import math
import copy

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ

class Linear():
    
    def __init__(self, dim1, dim2, activation, bias=True, alpha=0.1):
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
        return "OrionML.Layer.Linear"
        
    def description(self):
        return f"OrionML.Layer.Linear  (shape:({self.dim1, self.dim2}), activation: {self.activaiton})"
    
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
            print("Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.")
            return activ.linear()
            
    def update_parameters(self, w_new, b_new=None):
        if self.bias==True and b_new is None:
            print("ERROR: Expected a bias when updating the parameters.")
        
        self.w = w_new
        if self.bias==True: self.b = b_new
        
    def value(self, x):
        z = np.matmul(x, self.w) + self.b
        return self.activation_function.value(z), z
    
    def derivative(self, x):
        z = np.matmul(x, self.w) + self.b
        d_activ = self.activation_function.derivative(z)
        return d_activ
    
    def forward(self, prev_A, training=False):
        curr_A, Z = self.value(prev_A)
        cache = (prev_A, self.w, self.b, Z)
        
        return curr_A, cache
        
       
class Dropout():
    def __init__(self, dropout_probability=0.3, scale=True):
        '''
        Dropout layer.

        Parameters
        ----------
        dropout_probability : TYPE, optional
            DESCRIPTION. The default is 0.3.
        scale : bool, optional
            Whether the remaining weights should be scaled by 1/dropout_probability.
            The default is True.


        '''
        self.dropout_probability = dropout_probability   
        self.scale = scale
        
        self.trainable = False
        
    def type(self):
        return "OrionML.Layer.Dropout"
        
    def description(self):
        return f"OrionML.Layer.Dropout (dropout probability: {self.dropout_probability})"
        
    def value(self, activation_output, training=False):
        '''
        Set each element in wa with probability dropout_probability to 0.
        
        Parameters
        ----------
        weights : ndarray, shape: (input size, output size)
            Activated weights to which dropout should be applied.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Copy of wa but each element set to 0 with probability dropout_probability.

        '''
        if training==False:
            return activation_output
        
        mask = np.random.rand(activation_output.shape[0], activation_output.shape[1]) > self.dropout_probability
        res = mask*activation_output
        if self.scale==True:
            res = res * 1/(1-self.dropout_probability)
            
        return res, mask
    
    def derivative(self, mask):
        return mask
    
    def forward(self, prev_A, training=False):
        curr_A, curr_mask = self.value(prev_A, training=training)
        cache = (prev_A, curr_mask)
        
        return curr_A, cache

            


























































