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


class Sequential():
    def __init__(self, layers):
        self.layers = layers
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
        return "Sequential with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.layers)
    
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def __call__(self, x_in):
        assert x_in.shape[1] == self.feature_num, "Number of features in the input does not match with the model."
        
        x_next = x_in
        
        for layer in self.layers:
            x_curr = x_next
            x_next = layer.value(x_curr)
        
        return x_next
    
    def __getitem__(self, index):
        return self.layers[index]
    
    def __iter__(self):
        for layer in self.layers:
            yield layer
    
    def __get_layer_parameters(self) -> None:
        for i in range(self.num_layers):
            layer = self.layers[i]
            self.trainable_layers.append(layer.trainable)
            if layer.trainable==True:
                self.layer_dimensions.append(layer.dimension)
                self.bias_layers.append(layer.bias)
                self.param_pos.append(i)
                self.activations.append(layer.activation)
            else:
                if i>0:
                    self.layer_dimensions.append(self.layer_dimensions[-1])
                else:
                    self.layer_dimensions.append(np.array([0,0]))
                
                self.bias_layers.append(False)
                self.param_pos.append(-1)
                
    def initialize_parameters(self) -> None:
        w_temp = []
        b_temp = []
        
        for i in range(self.num_layers):
            if self.trainable_layers[i]==True:
                w_temp.append(np.random.rand(self.layer_dimensions[i][0], self.layer_dimensions[i][1]) * 1/np.sqrt(self.layer_dimensions[i][0]))
                if self.bias_layers[i]==True:
                    b_temp.append(np.random.rand(1, self.layer_dimensions[i][1]) * 1/np.sqrt(self.layer_dimensions[i][0]))
                    self.layers[i].update_parameters(w_temp[-1], b_temp[-1])
                else:
                    b_temp.append(np.array([[0]]))
                    self.layers[i].update_parameters(w_temp[-1])
                    
        return w_temp, b_temp
        

class NeuralNetwork():
    def __init__(self, sequential, learning_rate=1e-2):
        self.sequential = sequential
        self.learning_rate = learning_rate
        
        self.w, self.b = self.sequential.initialize_parameters()
        
        self.dw = [0]*len(self.w)
        self.db = [0]*len(self.b)
        
        self.caches = []
                
    def __str__(self):
        return "Neural Network with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.sequential)
    
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.sequential))
                                        
    def forward_step(self, prev_A, layer_pos):
        curr_layer_type = self.sequential[layer_pos].type()
        
        if curr_layer_type == "OrionML.Layer.Linear":
            curr_A, cache = self.sequential[layer_pos].forward(prev_A, training=True)
        elif curr_layer_type == "OrionML.Layer.Dropout":
            curr_A, cache = self.sequential[layer_pos].forward(prev_A, training=True)
        else:
            print("ERROR: Layer of unknown type in layers")
        
        return curr_A, cache
    
    def forward(self, x):
        caches = []
        A = x
        
        for i in range(self.sequential.num_layers):
            prev_A = A
            A, cache = self.forward_step(prev_A, i)
            caches.append(cache)
                
        return A, caches
    
    def Linear_backward_step(self, dA, layer_pos, cache):
        prev_A, curr_w, curr_b, curr_Z = cache
        d_activation = self.sequential[layer_pos].derivative(prev_A)
        curr_dA = np.einsum('ijk,ik->ij', d_activation, dA)
        curr_dw = 1/prev_A.shape[0] * np.matmul(prev_A.T, curr_dA)
        curr_db = 1/prev_A.shape[0] * np.sum(curr_dA, axis=0, keepdims=True)
        prev_dA = np.matmul(curr_dA, curr_w.T)
        return prev_dA, curr_dw, curr_db
    
    def Dropout_backward_step(self, dA, layer_pos, cache):
        prev_A, curr_mask = cache
        d_layer = self.sequential[layer_pos].derivative(curr_mask)
        prev_dA = d_layer * dA
        return prev_dA
    
    def backward(self, dA, caches):
        grads = []
        
        for i in reversed(range(self.sequential.num_layers)):
            curr_dA = dA
            curr_layer_type = self.sequential[i].type()
            curr_cache = caches[i]
            
            if curr_layer_type == "OrionML.Layer.Linear":
                dA, curr_dw, curr_db = self.Linear_backward_step(curr_dA, i, curr_cache)
                grads = [[dA, curr_dw, curr_db]] + grads
                
            elif curr_layer_type == "OrionML.Layer.Dropout":
                dA = self.Dropout_backward_step(curr_dA, i, curr_cache)
        
        return grads
    
    def update_parameters(self, grads):
        grad_pos = 0
        
        for layer in self.sequential:
            if layer.trainable==True:
                layer.w -= self.learning_rate * grads[grad_pos][1]
                layer.b -= self.learning_rate * grads[grad_pos][2]
                self.w[grad_pos] = layer.w
                self.b[grad_pos] = layer.b
                grad_pos += 1
                
        return
    
    def fit(self, x, y, epochs, batch_size = None):
        
        num_samples = x.shape[0]
        
        if batch_size==None:
            x_batches = [x]
            y_batches = [y]
        else:
            x_batches = [x[i:i+batch_size] for i in range(0, num_samples, batch_size)]
            y_batches = [y[i:i+batch_size] for i in range(0, num_samples, batch_size)]
        
        for i in range(epochs):
            for curr_x, curr_y in zip(x_batches, y_batches):
                A, caches = self.forward(curr_x)
                #AL = Loss.mse().value(curr_y, A)
                #dAL = Loss.mse().derivative(curr_y, A)
                AL = Loss.hinge().value(curr_y, A)
                dAL = Loss.hinge().derivative(curr_y, A)
                grads = self.backward(dAL, caches)
                self.update_parameters(grads)
            if i% math.ceil(epochs/10) == 0:
                print(f"Iteration {i:4}: Cost {AL:8.2f}, params: {self.w[0][0][0]:5.2f}, {self.b[0][0][0]:5.2f}")
        
        return
   
# %%

if __name__ == "__main__":

    np.random.seed(0)
    
    x = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], 
                  [2,0,0,0,0,0], [0,2,0,0,0,0], [0,0,2,0,0,0], [0,0,0,2,0,0], [0,0,0,0,2,0], [0,0,0,0,0,2],
                  [1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1],
                  [1,4,0,0,0,0], [0,0,1,3,0,0], [0,0,0,0,6,1]])
    y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1],
                  [1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1],
                  [1,0,0], [0,1,0], [0,0,1],
                  [1,0,0], [0,1,0], [0,0,1]])
    
    l = Sequential([Layer.Linear(6, 12, "linear"), Layer.Dropout(0.2), Layer.Linear(12, 3, "softmax")])
    n = NeuralNetwork(l, learning_rate=0.1)

    n.fit(x, y, epochs=1000, batch_size=None)
    
    w = n.w[0]
    b = n.b[0]
   
# %%

if __name__ == "__main__":

    np.random.seed(0)
    
    x = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], 
                  [2,0,0,0,0,0], [0,2,0,0,0,0], [0,0,2,0,0,0], [0,0,0,2,0,0], [0,0,0,0,2,0], [0,0,0,0,0,2],
                  [1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1],
                  [1,4,0,0,0,0], [0,0,1,3,0,0], [0,0,0,0,6,1]])
    y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1],
                  [2,0,0], [2,0,0], [0,2,0], [0,2,0], [0,0,2], [0,0,2],
                  [2,0,0], [0,2,0], [0,0,2],
                  [5,0,0], [0,4,0], [0,0,7]])
    
    l = [Layer.Linear(6, 3, "linear")]
    n = NeuralNetwork(l, learning_rate=1)
    n.fit(x, y, epochs=50, batch_size=6)
    
    w = n.w
    b = n.b
















































































