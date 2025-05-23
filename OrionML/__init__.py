import numpy as np

import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
from time import time
from pathlib import Path
import sys

import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import Loss
import utils
import method
import activation
import regularizer
import Layer
import initializer


class Sequential():
    def __init__(self, layers, initializer="uniform"):
        '''
        Sequence of layers for the input to a Neural Network.

        Parameters
        ----------
        layers : list
            List containing the Layers for the Neural Network.
        initializer : str
            How the parameters of the trainable layers should be initialized. If it is "glorot" 
            or "Xavier", glorot/Xavier initialization is used. If it is "he", he initialization 
            is used. For any other value, the parameters are initialized to small uniformally 
            distributed values. The default is "uniform".

        '''
        self.layers = layers
        self.initializer = initializer
        self.num_layers = len(layers)
        self.layer_dimensions = []
        self.trainable_layers = []
        self.bias_layers = []
        self.param_pos = []
        self.activations = []
        self.__get_layer_parameters()
        
        trainable_layers_dimensions = [val for val,bal in zip(self.layer_dimensions, self.trainable_layers) if bal]
        
        self.feature_num = trainable_layers_dimensions[0][0]
        self.output_dim = trainable_layers_dimensions[-1][1]
        
    def __str__(self):
        '''

        Returns
        -------
        str
            Description of the Sequential and the layers it contains.

        '''
        return "Sequential with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.layers)
    
    def __len__(self):
        '''
        If the length of a Sequential is calles, return the number of layers in the sequential.

        Returns
        -------
        int
            Number of layers in the Sequential.

        '''
        return self.num_layers
    
    def __repr__(self):
        '''
        
        Returns
        -------
        str
            Types of the layers in the Sequential.

        '''
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def __call__(self, x_in, training=False):
        '''
        Pass data through all the layers of the sequential.

        Parameters
        ----------
        x_in : ndarray, shape: (number of samples, self.feature_num)
            Input data.
        training : bool
            If the sequential should be called for training. The default is True.

        Returns
        -------
        x_next : ndarray, shape: (number of samples, self.output_dim)
            Result after the input is passed through all layers in the Sequential.

        '''        
        x_next = x_in
        
        for layer in self.layers:
            x_curr = x_next
            x_next, _ = layer.value(x_curr, training=training)
        
        return x_next
    
    def __getitem__(self, index):
        '''
        Get the Layer at position index in the Sequential.

        Parameters
        ----------
        index : int
            Position of the desired Layer in the Sequential. Must be larger than 0 and 
            smaller than the number of Layers in the Sequential.

        Returns
        -------
        OrionML.Layer
            Layer at position index in the Sequential.

        '''
        return self.layers[index]
    
    def __iter__(self):
        '''
        Iterating through the Sequential gives its different layers.
        
        Yields
        ------
        layer : OrionML.Layer
            Layer in the Sequential.

        '''
        for layer in self.layers:
            yield layer
    
    def __get_layer_parameters(self) -> None:
        '''
        Get information about the Layers in the Sequential.

        '''
        for i in range(self.num_layers):
            layer = self.layers[i]
            self.trainable_layers.append(layer.trainable)
            if layer.trainable==True:
                self.layer_dimensions.append(layer.dimension)
                if layer.type() in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                    self.bias_layers.append(layer.bias)
                    self.param_pos.append(i)
                    self.activations.append(layer.activation)
                else:
                    self.bias_layers.append(False)
                    self.param_pos.append(-1)
            else:
                if i>0:
                    self.layer_dimensions.append(self.layer_dimensions[-1])
                else:
                    self.layer_dimensions.append(np.array([0,0]))
                
                self.bias_layers.append(False)
                self.param_pos.append(-1)
                
        return
    
    def initialize_parameters(self):
        '''
        Initialize the parameters in the trainable Layers of the Sequential.

        Returns
        -------
        params : dict
            Dictionary containing the parameters of all trainable layers.
        derivs : dict
            Dictionary containing arrays where the derivatives for all trainable parameters 
            in the layers can be stored.

        '''
        
        if self.initializer in ["glorot", "Xavier"]:
            params, derivs = initializer.glorot(self.layers)
            
        elif self.initializer=="he":
            params, derivs = initializer.he(self.layers)
            
        else:
            params, derivs = initializer.uniform(self.layers)
        
        return params, derivs


class NeuralNetwork():
    def __init__(self, sequential, loss="mse", optimizer="gradient_descent", learning_rate=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=False):
        '''
        Create a Neural Network with Layers defined in a Sequential.

        Parameters
        ----------
        sequential : OrionML.Sequential
            Sequential Object with the Layers of the Neural Network.
        loss : str, optional
            What loss function to use. Has to be one of the following: 
                {mse, mae, mbe, cross_entropy, hinge, squared_hinge, L1loss, L2loss, huber}. 
            The default is "mse".
        optimizer : str, optional
            What optimizer to use. Has to be one of the following:
                {gradient_descent, adam}.
            The default is "gradient_descent".
        learning_rate : float, optional
            Learning rate of the Neural Network. The default is 1e-2.
        beta1 : float, optional
            Value of beta1 used in the adam optimizer. The default is 0.9.
        beta2 : float, optional
            Value of beta1 used in the adam optimizer. The default is 0.999.
        epsilon : float, optional
            Value of epsilon used in the adam optimizer. The default is 1e-8.
        verbose : bool/int, optional
            Whether the progress of the model should be displayed in the column. If set to an 
            integer, it is the number of times that information about the training is shown 
            during training. If set to False, no information is displayed. Setting verbose to 
            True is equivalent to setting it to 10. The default is False.

        '''
        self.sequential = sequential
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        
        if type(verbose) is int:
            self.verbose = True
            self.verbose_num = verbose
        
        else:
            self.verbose = verbose
            self.verbose_num = 10 if self.verbose else 0
        
        self.optimizer_name = optimizer
        self.optimizer = self.__select_optimizer()
        
        self.loss_name = loss
        self.loss_function = self.__select_loss_function()
        
        self.params, self.derivs = self.sequential.initialize_parameters()
                        
        self.J_h = []
        self.epoch = 0
        self.times = []
                
    def __str__(self):
        '''

        Returns
        -------
        str
            Description of the Neural Network and the layers it contains.

        '''
        return f"Neural Network with {self.loss_name} Loss, {self.optimizer_name} optimizer and {self.sequential.num_layers} layers:\n" + "\n".join(val.description() for val in self.sequential)
    
    def __repr__(self):
        '''
        
        Returns
        -------
        str
            Types of the layers in the Sequential of the Neural Network.

        '''
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.sequential))
    
    def __select_optimizer(self):
        '''
        Returns the optimizer associated with the input for optimizer to the class.

        Returns
        -------
        OrionML.NeuralNetwork.optimizer
            Optimizer used during the training.

        '''
        if self.optimizer_name in ["Adam", "adam"]:
            self.m_dw, self.v_dw = list(np.zeros((len(self.sequential), 1))), list(np.zeros((len(self.sequential), 1)))
            self.m_db, self.v_db = list(np.zeros((len(self.sequential), 1))), list(np.zeros((len(self.sequential), 1)))
            return self.Adam
            
        elif self.optimizer_name in ["gradient descent", "gradient_descent", "gd"]:
            return self.gradient_descent
        
        else:
            assert False, "Invalid input for optimizer, please choose one of the following: {gradient_descent, Adam}."
            
    def __select_loss_function(self):
        '''
        Returns the loss function associated with the input for loss to the class.

        Returns
        -------
        OrionML.Loss.function
            Loss function.

        '''
        if self.loss_name in ["mse"]: return Loss.mse()
        elif self.loss_name in ["mae"]: return Loss.mae()
        elif self.loss_name in ["mbe"]: return Loss.mbe()
        elif self.loss_name in ["cross_entropy"]: return Loss.cross_entropy()
        elif self.loss_name in ["hinge"]: return Loss.hinge()
        elif self.loss_name in ["squared_hinge"]: return Loss.squared_hinge()
        elif self.loss_name in ["L1loss"]: return Loss.L1loss()
        elif self.loss_name in ["L2loss"]: return Loss.L2loss()
        elif self.loss_name in ["huber"]: return Loss.huber()
        else:
            assert False, "Invalid input for loss, please choose on of the following: {mse, mae, mbe, cross_entropy, hinge, squared_hinge, L1loss, L2loss, huber}."
                                                    
    def forward_step(self, prev_A, layer_pos):
        '''
        Evaluate a forward step over one Layer.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples passed to the Neural Network, dimension 1 of the current Layer)
            Input data put through all Layers before the current Layer.
        layer_pos : int
            Position of the layer for which the forward step should be performed.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples passed to the Neural Network, dimension 2 of the current Layer)
            Input data put through all Layers before the next Layer.
        cache : tuple
            Tuple containing information required for backwards propagation. For the contents refer to the 
            documentation of the Layers.

        '''
        curr_A, cache = self.sequential[layer_pos].forward(prev_A, training=True)
                    
        return curr_A, cache
    
    def forward(self, x):
        '''
        Forward propagation through the Layers of the Neural Network.

        Parameters
        ----------
        x : ndarray, shape: (number of samples passed to the Neural Network, number of features)
            Input of the Neural Network.

        Returns
        -------
        A : ndarray, shape: (number of samples passed to the Neural Network, output dimension of the Neural Network)
            Result of the input data passed through all Layers.
        caches : list
            List containing the caches from the forward propagation steps.

        '''
        caches = []
        A = x
        
        for i in range(self.sequential.num_layers):
            prev_A = A
            A, cache = self.forward_step(prev_A, i)
            caches.append(cache)
                
        return A, caches
    
    def backward(self, dA, caches):
        '''
        Backward propagation through the Layers of the Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples passed to the Neural Network, output dimension of the Neural Network)
            Derivative of the loss function at the values given by the forward propagation.
        caches : list
            List containing the same number of tuples as layers in the model. Each tuple contains information from the 
            forward pass of the layer used in backward propagation.

        Returns
        -------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.

        '''
        grads = []
        
        for i in reversed(range(self.sequential.num_layers)):            
            curr_dA = dA
            curr_layer_type = self.sequential[i].type()
            curr_cache = caches[i]
            
            if curr_layer_type in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                dA, curr_dw, curr_db = self.sequential[i].backward(curr_dA, curr_cache, training=True)
                grads = [[dA, curr_dw, curr_db]] + grads
                
            elif curr_layer_type in ["OrionML.Layer.Dropout", "OrionML.Layer.Reshape", "OrionML.Layer.Flatten", "OrionML.Layer.Pool"]:
                dA = self.sequential[i].backward(curr_dA, curr_cache, training=True)
                
            elif curr_layer_type in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                dA, curr_dgamma, curr_dbeta = self.sequential[i].backward(curr_dA, curr_cache, training=True)
                grads = [[dA, curr_dgamma, curr_dbeta]] + grads
                                        
        return [[val[0] for val in grads], [val[1] for val in grads], [val[2] for val in grads]]
    
    def update_parameters(self, grads) -> None:
        '''
        Update the weights and bias of the trainable Layers.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.


        '''
        self.optimizer(grads)
                
        return
    
    def gradient_descent(self, grads) -> None:
        '''
        Gradient descent optimizer.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.

        '''
        layer_pos = 0
        train_pos = 0
        
        for layer in self.sequential:
            if layer.trainable:
                curr_layer_type = layer.type()
                if curr_layer_type in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                    layer.w -= self.learning_rate * grads[1][train_pos]
                    layer.b -= self.learning_rate * grads[2][train_pos]
                    
                    self.params[f"w layer {layer_pos}"] = layer.w
                    self.params[f"b layer {layer_pos}"] = layer.b
                
                elif curr_layer_type=="OrionML.Layer.BatchNorm":
                    layer.gamma -= self.learning_rate * grads[1][train_pos]
                    layer.beta -= self.learning_rate * grads[2][train_pos]
                    self.params[f"gamma layer {layer_pos}"] = layer.gamma
                    self.params[f"beta layer {layer_pos}"] = layer.beta
                    
                train_pos += 1
            
            layer_pos += 1
            
        return
    
    def Adam(self, grads) -> None:
        '''
        Adam optimizer.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer followed by the derivations with 
            respect to the trainable parameters of the layer.

        '''
        layer_pos = 0
        train_pos = 0
        
        for layer in self.sequential:
            if layer.trainable:
                curr_layer_type = layer.type()
                if curr_layer_type in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                    self.m_dw[layer_pos] = (self.beta1*self.m_dw[layer_pos] + (1-self.beta1)*grads[1][train_pos])
                    self.m_db[layer_pos] = (self.beta1*self.m_db[layer_pos] + (1-self.beta1)*grads[2][train_pos])
                    self.v_dw[layer_pos] = (self.beta2*self.v_dw[layer_pos] + (1-self.beta2)*np.square(grads[1][train_pos]))
                    self.v_db[layer_pos] = (self.beta2*self.v_db[layer_pos] + (1-self.beta2)*np.square(grads[2][train_pos]))
                                    
                    layer.w -= (self.learning_rate * (self.m_dw[layer_pos] / (1-self.beta1**self.epoch))/(np.sqrt(self.v_dw[layer_pos] / (1-self.beta2**self.epoch)) + self.epsilon))
                    layer.b -= (self.learning_rate * (self.m_db[layer_pos] / (1-self.beta1**self.epoch))/(np.sqrt(self.v_db[layer_pos] / (1-self.beta2**self.epoch)) + self.epsilon))

                    self.params[f"w layer {layer_pos}"] = layer.w
                    self.params[f"b layer {layer_pos}"] = layer.b
                    
                elif curr_layer_type=="OrionML.Layer.BatchNorm":
                    layer.gamma -= self.learning_rate * grads[1][train_pos]
                    layer.beta -= self.learning_rate * grads[2][train_pos]
                    self.params[f"gamma layer {layer_pos}"] = layer.gamma
                    self.params[f"beta layer {layer_pos}"] = layer.beta
                
                train_pos += 1
                                
            layer_pos += 1
        
        return
    
    def fit(self, x, y, epochs, batch_size=None, validation = None):
        '''
        Fit input data to a target.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Training data.
        y : ndarray, shape: (number of samples, output dimension of the Neural Network)
            Target data.
        epochs : int
            Number of epochs in the training of the Neural Network.
        batch_size : int/None, optional
            Size of batches that should be used for the training. If set to None, the 
            whole training data is used at the same time. If the batch size does not 
            exactly divide the number of samples, the last batch is smaller than the 
            batch size. The default is None.
        validation : None/list, optional
            List containing the data for the validation at position 0 and the target of the 
            validation at position 1. If set to None, no validation is displayed during training. 
            If set to a list, each time information about the state of the model is displayed 
            during training, information from the validation data is also displayed. The default 
            is None.

        '''        
        num_samples = x.shape[0]
                        
        self.tfor = []
        self.tbak = []
        
        if batch_size==None:
            x_batches = [x]
            y_batches = [y]
        else:
            x_batches = [x[i:i+batch_size] for i in range(0, num_samples, batch_size)]
            y_batches = [y[i:i+batch_size] for i in range(0, num_samples, batch_size)]
                
        for i in range(epochs):
            self.epoch += 1
            start_time = time()
            
            curr_tfor = 0
            curr_tbak = 0
            
            for curr_x, curr_y in zip(x_batches, y_batches):
                stfor = time()
                A, caches = self.forward(curr_x)
                curr_tfor += time()-stfor
                                
                AL = self.loss_function.value(curr_y, A)
                dAL = self.loss_function.derivative(curr_y, A)
                
                stbak = time()
                grads = self.backward(dAL, caches)
                curr_tbak += time()-stbak
                                
                self.update_parameters(grads)
                
                tst = np.array([np.array([np.isnan(val).any() for val in bal]).any() for bal in grads]).any()
                assert not tst, "hello nan"
                                                                                
            self.tfor.append(curr_tfor)
            self.tbak.append(curr_tbak)
                            
            self.J_h.append(AL)
            #print(AL)
            
            if self.verbose and ((i+1)% math.ceil(epochs/self.verbose_num) == 0 or i==0):
                if not validation is None:
                    pred_val = np.array([np.random.multinomial(1,val) for val in self.sequential(validation[0])])
                    same_arr_val = np.array([np.array_equal(validation[1][i], pred_val[i]) for i in range(len(validation[1]))])
                    acc_val = np.sum(same_arr_val)/len(validation[1])
                    
                    pred_train = np.array([np.random.multinomial(1,val) for val in self.sequential(x)])
                    same_arr_train = np.array([np.array_equal(y[i], pred_train[i]) for i in range(len(y))])
                    acc_train = np.sum(same_arr_train)/len(y)
                    
                    print(f"Iteration {i+1:4}:")
                    #print(f"Training:   Loss {self.loss_function.value(y, pred_train):8.4} ({self.loss_function.value(y, self.sequential(x)):8.4}, {AL:8.4}), accuracy {100*acc_train:2.1f}%.\n")
                    print(f"Training   Loss: {self.loss_function.value(y, pred_train):8.4}, accuracy {100*acc_train:2.1f}%.")
                    print(f"Validation Loss: {self.loss_function.value(validation[1], pred_val):8.4}, accuracy {100*acc_val:2.1f}%.\n")
                    
                else:
                    print(f"Iteration {i+1:4} training Loss: {AL:1.4}")
                    
            self.times.append(time()-start_time)
                        
        return

















































































