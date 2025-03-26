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
        '''
        Sequence of layers for the input to a Neural Network.

        Parameters
        ----------
        layers : list
            List containing the Layers for the Neural Network.

        '''
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
        '''

        Returns
        -------
        str
            Description of the Sequential and the layers it contains.

        '''
        return "Sequential with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.layers)
    
    def __repr__(self):
        '''
        
        Returns
        -------
        str
            Types of the layers in the Sequential.

        '''
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def __call__(self, x_in):
        '''
        Pass data through all the layers of the sequential.

        Parameters
        ----------
        x_in : ndarray, shape: (number of samples, self.feature_num)
            Input data.

        Returns
        -------
        x_next : ndarray, shape: (number of samples, self.output_dim)
            Result after the input is passed through all layers in the Sequential.

        '''
        assert x_in.shape[1] == self.feature_num, "Number of features in the input does not match with the model."
        
        x_next = x_in
        
        for layer in self.layers:
            x_curr = x_next
            x_next = layer.value(x_curr)
        
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
                
        return
                
    def initialize_parameters(self):
        '''
        Initialize the parameters in the trainable Layers of the Sequential.

        Returns
        -------
        w_temp : list
            List containing the weights of the trainable Layers in the sequential.
        b_temp : list
            List containing the bias of the trainable Layers in the sequential.

        '''
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
        '''
        Create a Neural Network with Layers defined in a Sequential.

        Parameters
        ----------
        sequential : OrionML.Sequential
            Sequential Object with the Layers of the Neural Network.
        learning_rate : float, optional
            Learning rate of the Neural Network. The default is 1e-2.

        '''
        self.sequential = sequential
        self.learning_rate = learning_rate
        
        self.w, self.b = self.sequential.initialize_parameters()
        
        self.dw = [0]*len(self.w)
        self.db = [0]*len(self.b)
        
        self.caches = []
                
    def __str__(self):
        '''

        Returns
        -------
        str
            Description of the Neural Network and the layers it contains.

        '''
        return "Neural Network with {self.num_layers} layers:\n" + "\n".join(val.description() for val in self.sequential)
    
    def __repr__(self):
        '''
        
        Returns
        -------
        str
            Types of the layers in the Sequential of the Neural Network.

        '''
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.sequential))
                                        
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
            
            if curr_layer_type == "OrionML.Layer.Linear":
                dA, curr_dw, curr_db = self.sequential[i].backward(curr_dA, curr_cache)
                grads = [[dA, curr_dw, curr_db]] + grads
                
            elif curr_layer_type == "OrionML.Layer.Dropout":
                dA = self.sequential[i].backward(curr_dA, curr_cache)
                        
        return grads
    
    def update_parameters(self, grads):
        '''
        Update the weights and bias of the trainable Layers.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.


        '''
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

        '''
        
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
                print(len(grads), len(grads[0]))
                print(grads[0][0].shape, grads[0][1].shape, grads[0][2].shape)
                print(grads[1][0].shape, grads[1][1].shape, grads[1][2].shape)
                print(grads[2][0].shape, grads[2][1].shape, grads[2][2].shape)
                #print(grads[0])
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
    
    l = Sequential([Layer.Linear(6, 12, "linear"), Layer.Dropout(0.2), Layer.Linear(12, 4, "linear"), Layer.Linear(4, 3, "softmax")])
    n = NeuralNetwork(l, learning_rate=0.1)

    n.fit(x, y, epochs=1, batch_size=None)
    
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
















































































