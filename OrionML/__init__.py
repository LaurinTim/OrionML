import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
from time import time

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
    
    def __len__(self):
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

        Returns
        -------
        x_next : ndarray, shape: (number of samples, self.output_dim)
            Result after the input is passed through all layers in the Sequential.

        '''
        assert x_in.shape[1] == self.feature_num, "Number of features in the input does not match with the model."
        
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
        w_temp = {}
        b_temp = {}
        
        for i in range(self.num_layers):
            if self.trainable_layers[i]==True:
                w_temp[f"w layer {i}"] = np.random.rand(self.layer_dimensions[i][0], self.layer_dimensions[i][1]) * 1e-2/np.sqrt(self.layer_dimensions[i][0])
                if self.bias_layers[i]==True:
                    b_temp[f"b layer {i}"] = np.random.rand(1, self.layer_dimensions[i][1]) * 1e-2/np.sqrt(self.layer_dimensions[i][0])
                    self.layers[i].update_parameters(w_temp[f"w layer {i}"], b_temp[f"b layer {i}"])
                else:
                    b_temp[f"b layer {i}"] = np.zeros((1,1))
                    self.layers[i].update_parameters(w_temp[f"w layer {i}"])
                    
        return w_temp, b_temp
        

class NeuralNetwork():
    def __init__(self, sequential, optimizer="gd", learning_rate=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        self.epsilon = epsilon
        self.optimizer_name = optimizer
        
        if optimizer in ["Adam", "adam"]:
            self.optimizer = self.Adam
            self.beta1 = beta1
            self.beta2 = beta2
            self.m_dw, self.v_dw = 0, 0
            self.m_db, self.v_db = 0, 0
            
        elif optimizer in ["gradient descent", "gd"]:
            self.optimizer = self.gradient_descent
        
        self.w, self.b = self.sequential.initialize_parameters()
        
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
        self.itf+=1
        curr_A, cache = self.sequential[layer_pos].forward(prev_A, training=True)
        
        if self.itf==54 and False:
            print(f"\nForward iteration {self.itf}:")
            print(self.sequential[layer_pos].description())
            print(curr_A)
            print([list(val) for val in prev_A])
            print()
            print([list(val) for val in self.sequential[layer_pos].w])
            print([list(val) for val in self.sequential[layer_pos].b])
        
        tst = np.isnan(curr_A).any()
        if tst and False:
            print(f"\nForward iteration: {self.itf}\n")
            
        if tst and True:
            print(f"\nForward iteration {self.itf}:")
            
            for i in range(curr_A.shape[0]):
                if np.isnan(curr_A[i]).any():
                    curr_A_nan = curr_A[i]
                    prev_A_nan = prev_A[i]
                    break
            
            print("curr_A: ", list(curr_A_nan))
            print("prev_A: ", list(prev_A_nan))
            print()
            print("W: ", list(self.sequential[layer_pos].w))
            print("b: ", list(self.sequential[layer_pos].b))
            print(self.sequential[layer_pos].description())
            print(prev_A_nan.shape)
            print(f"Forward iteration {self.itf}\n")
            
        assert not tst, "nan forward"
        
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
            self.itb+=1
            
            curr_dA = dA
            curr_layer_type = self.sequential[i].type()
            curr_cache = caches[i]
            
            if curr_layer_type == "OrionML.Layer.Linear":
                dA, curr_dw, curr_db = self.sequential[i].backward(curr_dA, curr_cache, training=True)
                grads = [[dA, curr_dw, curr_db]] + grads
                
            elif curr_layer_type == "OrionML.Layer.Dropout":
                dA = self.sequential[i].backward(curr_dA, curr_cache, training=True)
                
            tst = np.isnan(dA).any()
            if tst and False:
                print(f"\nBackward iteration: {self.itb}\n")
                
            assert not tst, "nan backward"
                                        
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
        grad_pos = 0
        
        for layer in self.sequential:
            if layer.trainable:
                layer.w -= self.learning_rate * grads[1][grad_pos]
                layer.b -= self.learning_rate * grads[2][grad_pos]
                self.w[f"w layer {grad_pos}"] = layer.w
                self.b[f"b layer {grad_pos}"] = layer.b
            
            grad_pos += 1
            
        return
    
    def Adam(self, grads) -> None:
        grad_pos = 0
        train_pos = 0
        
        for layer in self.sequential:
            if layer.trainable:
                self.m_dw[grad_pos] = (self.beta1*self.m_dw[grad_pos] + (1-self.beta1)*grads[1][train_pos])
                self.m_db[grad_pos] = (self.beta1*self.m_db[grad_pos] + (1-self.beta1)*grads[2][train_pos])
                self.v_dw[grad_pos] = (self.beta2*self.v_dw[grad_pos] + (1-self.beta2)*np.square(grads[1][train_pos]))
                self.v_db[grad_pos] = (self.beta2*self.v_db[grad_pos] + (1-self.beta2)*np.square(grads[2][train_pos]))
                                
                layer.w -= self.learning_rate * (self.m_dw[grad_pos] / (1-self.beta1**self.epoch))/(np.sqrt(self.v_dw[grad_pos] / (1-self.beta2**self.epoch)) + self.epsilon)
                layer.b -= self.learning_rate * (self.m_db[grad_pos] / (1-self.beta1**self.epoch))/(np.sqrt(self.v_db[grad_pos] / (1-self.beta2**self.epoch)) + self.epsilon)
                self.w[f"w layer {grad_pos}"] = layer.w
                self.b[f"b layer {grad_pos}"] = layer.b
                
                train_pos += 1
                                
            grad_pos += 1
        
        return
    
    def fit(self, x, y, epochs, batch_size=None, b1=0.9, b2=0.999):
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
        self.J_h = []
        self.epoch = 0
        self.times = []
        self.itf = 0
        self.itb = 0
        self.ita = 0
        
        num_samples = x.shape[0]
        
        if batch_size==None:
            x_batches = [x]
            y_batches = [y]
        else:
            x_batches = [x[i:i+batch_size] for i in range(0, num_samples, batch_size)]
            y_batches = [y[i:i+batch_size] for i in range(0, num_samples, batch_size)]
            
            
        if self.optimizer_name in ["Adam", "adam"]:
            self.m_dw, self.v_dw = list(np.zeros((len(self.sequential), 1))), list(np.zeros((len(self.sequential), 1)))
            self.m_db, self.v_db = list(np.zeros((len(self.sequential), 1))), list(np.zeros((len(self.sequential), 1)))
                
        for i in range(epochs):
            self.epoch += 1
            start_time = time()
            for curr_x, curr_y in zip(x_batches, y_batches):
                A, caches = self.forward(curr_x)
                
                #AL = Loss.mse().value(curr_y, A)
                #dAL = Loss.mse().derivative(curr_y, A)
                AL = Loss.hinge().value(curr_y, A)
                dAL = Loss.hinge().derivative(curr_y, A)
                
                grads = self.backward(dAL, caches)
                
                self.update_parameters(grads)
                
                tst = np.array([np.array([np.isnan(val).any() for val in bal]).any() for bal in grads]).any()
                assert not tst, "hello nan"
                
            self.times.append(time()-start_time)
                            
            self.J_h.append(AL)
            if (i+1)% math.ceil(epochs/10) == 0 or i==0:
                pred = np.array([np.random.multinomial(1,val) for val in self.sequential(x, training=True)])
                print(f"Iteration {i+1:4}: Cost {Loss.hinge().value(y, pred):8.4}")
                #print(f"Iteration {i+1:4}: Cost {AL:8.4}")
                        
        return

# %%

if __name__ == "__main__":

    np.random.seed(0)
    
    df1 = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\Examples\\example data\\MNIST\\mnist_train1.csv", delimiter=",", header=None)

    df2 = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\Examples\\example data\\MNIST\\mnist_train2.csv", delimiter=",", header=None)
    
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    
    #df = df.iloc[:10000]
    
    df_val = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\Examples\\example data\\MNIST\\mnist_test.csv", 
                       delimiter=",", header=None)
    
    train_X = np.array(df.iloc[:,1:])
    train_y_col = np.array(df.iloc[:,0]).reshape(-1,1)
    val_X = np.array(df_val.iloc[:,1:])
    val_y_col = np.array(df_val.iloc[:,0]).reshape(-1,1)
    
    train_y = np.zeros((len(train_y_col), 10))
    for i in range(len(train_y)):
        train_y[i][train_y_col[i]] = 1
        
    val_y = np.zeros((len(val_y_col), 10))
    for i in range(len(val_y)):
        val_y[i][val_y_col[i]] = 1
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="Adam", learning_rate=8e-4)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=1024)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 128, activation="relu"), Layer.Linear(128, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="gd", learning_rate=1)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=100)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="gradient descent", learning_rate=1)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=1000)

# %%

if __name__ == "__main__":
    y_pred = seq(val_X)
    pred = np.array([np.random.multinomial(1,val) for val in y_pred])
    same = np.array([np.array_equal(val_y[i], pred[i]) for i in range(len(val_y))])
    wrong = len(val_y)-np.sum(same)
    acc = np.sum(same)/len(val_y)
    loss = Loss.hinge().value(val_y, pred)
    print(f"Validation data tests: {wrong:4.0f}, {acc:0.4f}, {loss:0.4f}")
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    yt_pred = seq(train_X)
    predt = np.array([np.random.multinomial(1,val) for val in yt_pred])
    samet = np.array([np.array_equal(train_y[i], predt[i]) for i in range(len(train_y))])
    wrongt = len(train_y)-np.sum(samet)
    acct = np.sum(samet)/len(train_y)
    losst = Loss.hinge().value(train_y, predt)
    print(f"Training data tests: {wrongt:5}, {acct:0.4}, {losst:0.4}")

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
















































































