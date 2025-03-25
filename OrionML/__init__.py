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
        self.learning_rate = learning_rate
        
        self.num_layers = len(layers)
        self.layer_dimensions = []
        self.trainable_layers = []
        self.bias_layers = []
        self.param_pos = []
        self.__get_layer_parameters()
        
        self.w = []
        self.b = []
        self.__initialize_parameters()
        
        self.dw = [0]*len(self.w)
        self.db = [0]*len(self.b)
        
        self.caches = []
                
    def __str__(self):
        return "Neural Network with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.layers)
    
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
        
    def __get_layer_parameters(self) -> None:
        for i in range(self.num_layers):
            layer = self.layers[i]
            self.trainable_layers.append(layer.trainable)
            if layer.trainable==True:
                self.layer_dimensions.append(layer.dimension)
                self.bias_layers.append(layer.bias)
                self.param_pos.append(i)
            else:
                self.layer_dimensions.append(self.layer_dimensions[-1])
                self.bias_layers.append(False)
                self.param_pos.append(-1)
        
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
                    
    def Linear_forward_step(self, prev_A, layer_pos):
        curr_A, Z = self.layers[layer_pos].value(prev_A)
        cache = (prev_A, self.layers[layer_pos].w, self.layers[layer_pos].b, Z)
        
        return curr_A, cache
    
    def Dropout_forward_step(self, prev_A, layer_pos):
        curr_A, curr_mask = self.layers[layer_pos].value(prev_A)
        cache = (prev_A, curr_mask)
        
        return curr_A, cache
                                        
    def forward_step(self, prev_A, layer_pos):
        curr_layer_type = self.layers[layer_pos].type()
        
        if curr_layer_type == "OrionML.Layer.Linear":
            curr_A, cache = self.Linear_forward_step(prev_A, layer_pos)
        elif curr_layer_type == "OrionML.Layer.Dropout":
            curr_A, cache = self.Dropout_forward_step(prev_A, layer_pos)
        else:
            print("ERROR: Layer of unknown type in layers")
        
        return curr_A, cache
    
    def forward(self, x):
        caches = []
        A = x
        
        for i in range(self.num_layers):
            prev_A = A
            A, cache = self.forward_step(prev_A, i)
            caches.append(cache)
                
        return A, caches
    
    def Linear_backward_step(self, dA, layer_pos, cache):
        prev_A, curr_w, curr_b, curr_Z = cache
        d_activation = self.layers[layer_pos].derivative(prev_A)
        curr_dA = np.einsum('ijk,ik->ij', d_activation, dA)
        curr_dw = np.matmul(prev_A.T, curr_dA)
        curr_db = np.sum(curr_dA, axis=0, keepdims=True)
        prev_dA = np.matmul(curr_dA, curr_w.T)
        return prev_dA, curr_dw, curr_db
    
    def Dropout_backward_step(self, dA, layer_pos, cache):
        prev_A, curr_mask = cache
        d_layer = self.layers[layer_pos].derivative(curr_mask)
        prev_dA = d_layer * dA
        return prev_dA
    
    def backward(self, dA, caches):
        grads = []
        
        for i in reversed(range(self.num_layers)):
            curr_dA = dA
            curr_layer_type = self.layers[i].type()
            curr_cache = caches[i]
            
            if curr_layer_type == "OrionML.Layer.Linear":
                dA, curr_dw, curr_db = self.Linear_backward_step(curr_dA, i, curr_cache)
                grads = [[dA, curr_dw, curr_db]] + grads
                
            elif curr_layer_type == "OrionML.Layer.Dropout":
                dA = self.Dropout_backward_step(curr_dA, i, curr_cache)
        
        return grads
    
    def update_parameters(self, grads):
        grad_pos = 0
        
        for layer in self.layers:
            if layer.trainable==True:
                layer.w -= self.learning_rate * grads[grad_pos][1]
                layer.b -= self.learning_rate * grads[grad_pos][2]
                self.w[grad_pos] = layer.w
                self.b[grad_pos] = layer.b
                grad_pos += 1
                
        return
    
    def fit(self, x, y, epochs):
        
        for i in range(epochs):
            A, caches = self.forward(x)
            AL = Loss.mse().value(y, A)
            dAL = Loss.mse().derivative(y, A)
            grads = self.backward(dAL, caches)
            self.update_parameters(grads)
            if i% math.ceil(epochs/10) == 0:
                print(f"Iteration {i:4}: Cost {AL:8.2f}, params: {self.w[0][0][0]:5.2f}, {self.b[0][0][0]:5.2f}")
        
        return
    
# %%

np.random.seed(0)

x = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]])

l = [Layer.Linear(6, 3, "linear")]
n = NeuralNetwork(l, learning_rate=0.2)
n.fit(x, y, epochs=100)

w = n.w
b = n.b

# %%

l = [Layer.Linear(1, 1, "linear")]
n = NeuralNetwork(l, learning_rate=0.1)

x = np.array([[0],[2],[4]])
y = np.array([[1],[5],[9]])

n.fit(x, y, epochs = 100)

w = n.w
b = n.b
















































































