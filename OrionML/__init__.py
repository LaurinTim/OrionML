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
        self.__initialize_parameters()
        
    def __str__(self):
        print("AAAAA")
        return f"Neural Network with layers: {[type(val) for val in self.layers]}"
        
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
                    
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    
                    

