import numpy as np
import matplotlib.pyplot as plt
import math
import copy

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import Loss
import utils
import method
import activation
import regularizer
import Layer

class NeuralNetwork():
    def __init__(self, layers, learning_rate=1e-2):
        self.layers = layers
        self.lr = learning_rate
        
        self.num_layers = len(layers)
        self.layer_dimensions = []
        self.trainable_layers = []
        self.bias_layers = []
        self.__get_layer_parameters()
        
        self.w = []
        self.b = []
        self.dw = []
        self.db = []
        self.__initialize_parameters()
        
        self.caches = []
        
        self.x = np.zeros((2,4))
        
    def __str__(self):
        return "Neural Network with layers:\n" + "\n".join(val.description() for val in self.layers)
    
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
        
    def __get_layer_parameters(self) -> None:
        for layer in self.layers:
            self.trainable_layers.append(layer.trainable)
            if self.trainable_layers[-1]==True:
                self.layer_dimensions.append(layer.dimension)
                self.bias_layers.append(layer.bias)
            else:
                self.layer_dimensions.append(self.layer_dimensions[-1])
                self.bias_layers.append(False)
        
    def __initialize_parameters(self) -> None:
        for i in range(self.num_layers):
            if self.trainable_layers[i]==True:
                self.w.append(np.random.rand(self.layer_dimensions[i][0], self.layer_dimensions[i][1]) * 1/np.sqrt(self.layer_dimensions[i][0]))
                if self.bias_layers[i]==True:
                    self.b.append(np.random.rand(1, self.layer_dimensions[i][1]) * 1/np.sqrt(self.layer_dimensions[i][0]))
                    self.layers[i].update_parameters(self.w[-1], self.b[-1])
                else:
                    self.b.append(np.array([[0]]))
                    self.layers[i].update_parameters(self.w[-1])
                                        
    def forward_step(self, prev_A, layer_pos):
        curr_layer_type = self.layers[layer_pos].type()
        curr_A = self.layers[layer_pos].value(prev_A)
        
        if curr_layer_type == "OrionML.Layer.Linear":
            cache = (prev_A, self.layers[layer_pos].w, self.layers[layer_pos].b)
        elif curr_layer_type == "OrionML.Layer.Activation":
            cache = (prev_A)
        elif curr_layer_type == "OrionML.Layer.Dropout":
            cache = (prev_A, self.layers[layer_pos].mask)
        
        return curr_A, cache
    
    def forward(self):
        A = self.x
        
        for i in range(self.num_layers):
            prev_A = A
            A, cache = self.forward_step(prev_A)
            self.caches.append(cache)
        
        return A
    
    def backward_step(self, dA, layer_pos):
        curr_layer_type = self.layers[layer_pos].type()
        cache = self.caches[layer_pos]
        prev_A = cache[0]
        curr_dA = self.layers[layer_pos].derivative(prev_A)
        
        if curr_layer_type == "OrionML.Layer.Linear":
            cache = (prev_A, self.layers[layer_pos].w, self.layers[layer_pos].b)
        elif curr_layer_type == "OrionML.Layer.Activation":
            cache = (prev_A)
        elif curr_layer_type == "OrionML.Layer.Dropout":
            cache = (prev_A, self.layers[layer_pos].mask)
        
        
        return
    
    

# %%

l = [Layer.Linear(2, 4, "linearn"), Layer.Linear(4, 1, "linear")]
n = NeuralNetwork(l)



















































































