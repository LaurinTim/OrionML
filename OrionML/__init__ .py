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
        
        self.buffers = {}
        
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

        '''
        
        if self.initializer in ["glorot", "Xavier"]:
            params, derivs = initializer.glorot(self.layers)
            
        elif self.initializer=="he":
            params, derivs = initializer.he(self.layers)
            
        else:
            params, derivs = initializer.uniform(self.layers)
        
        return
    
    def initialize_layer_dimensions(self, input_shape) -> None:
        '''
        Define in and out dimensions for each layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of a single sample passed to the sequential.

        '''
        curr_shape = input_shape
        
        for i,layer in enumerate(self.layers):
            layer.in_dim = curr_shape
            
            if layer.type()=="OrionML.Layer.Linear":
                curr_shape = (layer.dim2,)
                
            elif layer.type() in ["OrionML.Layer.Conv", "OrionML.Layer.Pool"]:
                H, W, _ = curr_shape
                H_out = (H + 2*layer.padding - layer.kernel_size)//layer.stride + 1
                W_out = (W + 2*layer.padding - layer.kernel_size)//layer.stride + 1
                curr_shape = (H_out, W_out, layer.out_channels,)
                
            elif layer.type()=="OrionML.Layer.Reshape":
                curr_shape = (layer.output_shape[1:],)
                
            elif layer.type()=="OrionML.Layer.Flatten":
                curr_shape = (math.prod(curr_shape),)
                
            layer.out_dim = curr_shape
        
        return
    
    def init_buffers(self, batch_size, input_dim) -> None:
        '''
        Initialize the buffers for the value of the input for different layers and the 
        gradients of the parameters.

        Parameters
        ----------
        batch_size : int
            Batch size used in training.

        '''
        activations = [None] * (self.num_layers + 1)
        activations[0] = np.empty((batch_size, *input_dim))
        
        grad_a = [None] * (self.num_layers)
        grad_b = [None] * (self.num_layers)
        
        for i, layer in enumerate(self.layers):
            layer.init_buffers(batch_size)
            
            activations[i+1] = np.empty((batch_size, *layer.out_dim))
            
            if layer.type() in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                grad_a[i] = np.empty_like(layer.w)
                grad_b[i] = np.empty_like(layer.b)
                
            elif layer.type() in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                grad_a[i] = np.empty_like(layer.gamma)
                grad_b[i] = np.empty_like(layer.beta)
                
            else:
                grad_a[i] = np.empty(0)
                grad_b[i] = np.empty(0)
            
        self.buffers["activations"] = activations
        self.buffers["grads"] = (grad_a, grad_b)
        self.buffers["dLoss"] = np.empty((batch_size, *self.layers[-1].out_dim))


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
        
        self.sequential.initialize_parameters()
        
        self.input_dim = None
                
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
        acts = self.sequential.buffers["activations"]
        np.copyto(acts[0], x)
        
        for i, layer in enumerate(self.sequential):
            layer.forward(acts[i], out_buffer=acts[i+1], training=True)
                
        return acts[-1]
    
    def backward(self, dA):
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
        acts = self.sequential.buffers["activations"]
        grads_w, grads_b = self.sequential.buffers["grads"]
        np.copyto(acts[-1], dA)
        
        for i in reversed(range(len(self.sequential))):
            curr_dA = acts[i+1]
            curr_layer_type = self.sequential[i].type()
            
            if curr_layer_type in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                self.sequential[i].backward(curr_dA, dw_buffer=grads_w[i], db_buffer=grads_b[i], dout_buffer=acts[i], training=True)
                
            elif curr_layer_type in ["OrionML.Layer.Dropout", "OrionML.Layer.Reshape", "OrionML.Layer.Flatten", "OrionML.Layer.Pool"]:
                self.sequential[i].backward(curr_dA, dout_buffer=acts[i], training=True)
                
            elif curr_layer_type in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                self.sequential[i].backward(curr_dA, dgamma_buffer=grads_w[i], dbeta_buffer=grads_b[i], dout_buffer=acts[i], training=True)
                                        
        return
    
    def update_parameters(self) -> None:
        '''
        Update the weights and bias of the trainable Layers.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.


        '''
        self.optimizer()
                
        return
    
    def gradient_descent(self) -> None:
        '''
        Gradient descent optimizer.

        Parameters
        ----------
        grads : list, shape: (number of trainable Layers, 3)
            List containing dA after each trainable layer and dw and db for each trainable layer.

        '''        
        for layer, dw, db in zip(self.sequential, *self.sequential.buffers["grads"]):
            if layer.trainable:
                curr_layer_type = layer.type()
                if curr_layer_type in ["OrionML.Layer.Linear", "OrionML.Layer.Conv"]:
                    layer.w -= self.learning_rate * dw
                    layer.b -= self.learning_rate * db
                                    
                elif curr_layer_type=="OrionML.Layer.BatchNorm":
                    layer.gamma -= self.learning_rate * dw
                    layer.beta -= self.learning_rate * db
                    
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

                    
                elif curr_layer_type=="OrionML.Layer.BatchNorm":
                    layer.gamma -= self.learning_rate * grads[1][train_pos]
                    layer.beta -= self.learning_rate * grads[2][train_pos]
                
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
        
        if batch_size==None:
            batch_size = len(x)
        
        self.input_dim = x.shape[1:]
        self.sequential.initialize_layer_dimensions(self.input_dim)
        
        self.sequential.init_buffers(batch_size, self.input_dim)
        self.loss_function.init_buffers(batch_size, self.sequential.layers[-1].out_dim)
                
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
                A = self.forward(curr_x)
                curr_tfor += time()-stfor
                                                
                AL = self.loss_function.value_buffered(curr_y, A)
                                
                dAL = self.sequential.buffers["dLoss"]
                self.loss_function.derivative_buffered(curr_y, A, out_buffer=dAL)
                
                #print(curr_x[0][500:505])
                                
                stbak = time()
                self.backward(dAL)
                curr_tbak += time()-stbak
                
                #print(curr_x[0][500:505])
                
                #if i==1: print(self.sequential.buffers["grads"][0][1])
                                
                self.update_parameters()
                                                                                
            self.tfor.append(curr_tfor)
            self.tbak.append(curr_tbak)
                            
            self.J_h.append(AL)
            if self.verbose and ((i+1)% math.ceil(epochs/self.verbose_num) == 0 or i==0):
                if not validation is None:
                    pred_val = np.array([np.random.multinomial(1,val) for val in self.sequential(validation[0])])
                    same_arr_val = np.array([np.array_equal(validation[1][i], pred_val[i]) for i in range(len(validation[1]))])
                    acc_val = np.sum(same_arr_val)/len(validation[1])
                    
                    #pred_train = np.array([np.random.multinomial(1,val) for val in self.sequential(x)])
                    #same_arr_train = np.array([np.array_equal(y[i], pred_train[i]) for i in range(len(y))])
                    #acc_train = np.sum(same_arr_train)/len(y)
                    
                    print(f"Iteration {i+1:4}:")
                    #print(f"Training:   Loss {self.loss_function.value(y, pred_train):8.4} ({self.loss_function.value(y, self.sequential(x)):8.4}, {AL:8.4}), accuracy {100*acc_train:2.1f}%.\n")
                    #print(f"Training:   Loss {self.loss_function.value(y, pred_train):8.4} ({self.loss_function.value(y, self.sequential(x)):5.4}), accuracy {100*acc_train:2.1f}%.")
                    print(f"Validation: Loss {self.loss_function.value(validation[1], pred_val):8.4} ({self.loss_function.value(validation[1], self.sequential(validation[0])):5.10}), accuracy {100*acc_val:2.1f}%.\n")
                    
                else:
                    print(f"Iteration {i+1:4} training Loss: {AL:1.4}")
                    
            self.times.append(time()-start_time)
                        
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
    scaler = utils.StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.fit_transform(val_X)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    
    seq = Sequential([Layer.Linear(784, 512, activation="relu"), Layer.Linear(512, 256, activation="relu"), Layer.Linear(256, 128, activation="relu"), 
                      Layer.Linear(128, 64, activation="relu"), Layer.Linear(64, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="gd", loss="cross_entropy", learning_rate=5e-2, verbose=10)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=32, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.Linear(784, 512, activation="relu"), Layer.Linear(512, 256, activation="relu"), Layer.Linear(256, 128, activation="relu"), 
    #                  Layer.Linear(128, 64, activation="relu"), Layer.Linear(64, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])
    
    seq = Sequential([Layer.Linear(784, 128, activation="relu"), Layer.Linear(128, 64, activation="relu"), 
                      Layer.Linear(64, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="gd", loss="cross_entropy", learning_rate=1e-3, verbose=10)
    
    nn.fit(train_X, train_y, epochs=5, batch_size=32, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    val_X = val_X.reshape(val_X.shape[0], 28, 28, 1)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 144, activation="relu"), Layer.Reshape((12, 12, 1)), Layer.Conv(1, 6, 5, activation="relu"), 
                      Layer.Pool(2, 2, pool_mode="max"), Layer.Flatten(), Layer.Linear(96, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="adam", loss="cross_entropy", learning_rate=1e-4, verbose=10)
    
    nn.fit(train_X, train_y, epochs=100, batch_size=32, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 144, activation="relu"), Layer.Reshape((12, 12, 1)), Layer.Conv(1, 6, 5, activation="relu"), 
                      Layer.Pool(2, 2, pool_mode="max"), Layer.Flatten(), Layer.Linear(96, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="adam", loss="cross_entropy", learning_rate=1e-4, verbose=10)
    
    nn.fit(train_X, train_y, epochs=100, batch_size=32, validation=[val_X, val_y])
        
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 3, 4, "linear", stride=2), Layer.Flatten(), Layer.Linear(507, 10, "softmax")])
    
    nn = NeuralNetwork(seq, optimizer="adam", loss="cross_entropy", learning_rate=1e-3, verbose=True)
    
    nn.fit(train_X, train_y, epochs=2, batch_size=256)#, validation=[val_X, val_y])
    
    #print(np.mean(nn.times), np.median(nn.times))
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 3, 5, "relu"), Layer.BatchNorm2D(3), Layer.Pool(2, 2, pool_mode="max"), 
                      Layer.Dropout(0.25), Layer.Conv(3, 6, 5, "relu"), Layer.BatchNorm2D(6), 
                      Layer.Pool(2, 2, pool_mode="max"), Layer.Dropout(0.25), Layer.Flatten(), 
                      Layer.Linear(96, 10, activation="softmax")], initializer="he")
    
    nn = NeuralNetwork(seq, optimizer="adam", loss="cross_entropy", learning_rate=1e-2, verbose=10)
    
    nn.fit(train_X, train_y, epochs=5, batch_size=256)#, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 28, 5, "relu", padding="same"), Layer.BatchNorm2D(28), Layer.Conv(28, 28, 5, "relu"), Layer.Pool(2, 2, pool_mode="max"), 
                      Layer.Dropout(0.25), Layer.Conv(28, 32, 5, "relu", padding="same"), Layer.BatchNorm2D(32), Layer.Conv(32, 32, 5, "relu"), 
                      Layer.Pool(2, 2, pool_mode="max"), Layer.Dropout(0.25), Layer.Flatten(), Layer.Linear(512, 512, activation="relu"), Layer.Dropout(0.25), 
                      Layer.Linear(512, 10, activation="softmax")])
    
    nn = NeuralNetwork(seq, optimizer="adam", loss="hinge", learning_rate=1e-5, verbose=20)
    
    nn.fit(train_X, train_y, epochs=2, batch_size=256)#, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 3, 3, "relu", stride=1), Layer.Conv(3, 6, 4, "relu", stride=2), Layer.Pool(2, 2, pool_mode="max"), 
                      Layer.Conv(6, 12, 2, "relu", stride=1, flatten=True), Layer.Linear(300, 10, "softmax")])
    
    nn = NeuralNetwork(seq, optimizer="gd", loss="mse", learning_rate=1e-5, verbose=20)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=32, validation=[val_X, val_y])
    
    #print(np.mean(nn.times), np.median(nn.times))

# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 3, 4, "linear", stride=2), Layer.Conv(3, 5, 1, "linear", stride=1, flatten=True), Layer.Linear(845, 10, "softmax")])
    
    nn = NeuralNetwork(seq, optimizer="gd", loss="mse", learning_rate=1e-3, verbose=20)
    
    nn.fit(train_X, train_y, epochs=5, batch_size=16, validation=[val_X, val_y])
    
    #print(np.mean(nn.times), np.median(nn.times))

# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Conv(1, 3, 1, "linear", stride=1, flatten=True), Layer.Linear(2352, 10, "softmax")])
    
    nn = NeuralNetwork(seq, optimizer="gd", loss="mse", learning_rate=1e-2, verbose=20)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=16, validation=[val_X, val_y])
    
    #print(np.mean(nn.times), np.median(nn.times))

# %%

nn.fit(train_X, train_y, epochs=100, batch_size=1024, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    epoch_conv = 15
    epoch_lin = 10
    
    seq[0].w = nn.wl[2*(epoch_conv-1)]
    seq[0].b = nn.bl[2*(epoch_conv-1)]
    
    seq[1].w = nn.wl[2*(epoch_lin-1)+1]
    seq[1].b = nn.bl[2*(epoch_lin-1)+1]
    
# %%

if __name__ == "__main__":
    seq[0].w = nn.wl[-2]
    seq[0].b = nn.bl[-2]
    
    seq[1].w = nn.wl[-1]
    seq[1].b = nn.bl[-1]
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    pred_val = np.array([np.random.multinomial(1,val) for val in seq(val_X)])
    same_arr_val = np.array([np.array_equal(val_y[i], pred_val[i]) for i in range(len(val_y))])
    acc_val = np.sum(same_arr_val)/len(val_y)
    
    pred_train = np.array([np.random.multinomial(1,val) for val in seq(train_X)])
    same_arr_train = np.array([np.array_equal(train_y[i], pred_train[i]) for i in range(len(train_y))])
    acc_train = np.sum(same_arr_train)/len(train_y)
        
    print(f"Training:   Loss {Loss.mse().value(train_y, pred_train):8.4}, accuracy {100*acc_train:2.1f}%.")
    print(f"Validation: Loss {Loss.mse().value(val_y, pred_val):8.4}, accuracy {100*acc_val:2.1f}%.\n")
        
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="adam", loss="cross_entropy", learning_rate=6e-2, verbose=True)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=1024, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="Adam", loss="hinge", learning_rate=5e-5, verbose=True)
    
    nn.fit(train_X, train_y, epochs=100, batch_size=None, validation=[val_X, val_y])
        
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="gd", loss="hinge", learning_rate=1e-5, verbose=True)
    
    nn.fit(train_X, train_y, epochs=500, batch_size=None, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    plt.scatter(np.arange(len(nn.times)), nn.times)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    #seq = Sequential([Layer.Linear(784, 45, activation="relu"), Layer.Linear(45, 35, activation="relu"), Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])

    nn = NeuralNetwork(seq, optimizer="Adam", loss="hinge", learning_rate=1e-3, verbose=20)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=128, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 128, activation="relu"), Layer.Linear(128, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, loss="hinge", optimizer="gd", learning_rate=1e-2, verbose=True)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=32)
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 128, activation="relu"), Layer.Dropout(0.5), Layer.Linear(128, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, loss="mse", optimizer="gd", learning_rate=1e-1, verbose=True)
    
    nn.fit(train_X, train_y, epochs=100, batch_size=32, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    seq = Sequential([Layer.Linear(784, 128, activation="relu"), Layer.Dropout(0.5), Layer.Linear(128, 32, activation="relu"), Layer.Linear(32, 10, activation="softmax")])

    nn = NeuralNetwork(seq, loss="mse", optimizer="adam", learning_rate=1e-3, verbose=True)
    
    nn.fit(train_X, train_y, epochs=100, batch_size=1024, validation=[val_X, val_y])
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    #This one works well (around 97% accuracy for validation after 100 epochs, 1e-2 learning rate and batch size 128.)
    seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 128, activation="relu"), Layer.Dropout(0.3), Layer.Linear(128, 10, activation="softmax")])
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 45, activation="relu"), Layer.Dropout(0.3), Layer.Linear(45, 35, activation="relu"), 
    #                  Layer.Linear(35, 25, activation="relu"), Layer.Linear(25, 10, activation="softmax")])
    #This one does not work well at all. Does not get higher than around 70% accuracy for the validation and training data. (Why? A more complex nn should at least increase the accuracy for the training data.)
    #seq = Sequential([Layer.BatchNorm(784), Layer.Linear(784, 256, activation="relu"), Layer.Dropout(0.3), Layer.Linear(256, 128, activation="relu"), 
    #                  Layer.Dropout(0.3), Layer.Linear(128, 64, activation="relu"), Layer.Dropout(0.3), Layer.Linear(64, 32, activation="relu"), 
    #                  Layer.Dropout(0.3), Layer.Linear(32, 16, activation="relu"), Layer.Dropout(0.3), Layer.Linear(16, 10, activation="softmax")])

    nn = NeuralNetwork(seq, loss="hinge", optimizer="adam", learning_rate=1e-2, verbose=10)
    
    nn.fit(train_X, train_y, epochs=10, batch_size=128, validation=[val_X, val_y])
    
# %%

nn.fit(train_X, train_y, epochs=10, batch_size=128, validation=[val_X, val_y])
        
# %%

if __name__ == "__main__":
    y_pred = seq(val_X)
    pred = np.array([np.random.multinomial(1,val) for val in y_pred])
    same = np.array([np.array_equal(val_y[i], pred[i]) for i in range(len(val_y))])
    wrong = len(val_y)-np.sum(same)
    acc = np.sum(same)/len(val_y)
    loss = Loss.cross_entropy().value(val_y, pred)
    print(f"Validation data tests: {wrong:4.0f}, {loss:0.4f}, {acc:0.4f}")
    
# %%

if __name__ == "__main__":
    np.random.seed(0)
    yt_pred = seq(train_X)
    predt = np.array([np.random.multinomial(1,val) for val in yt_pred])
    samet = np.array([np.array_equal(train_y[i], predt[i]) for i in range(len(train_y))])
    wrongt = len(train_y)-np.sum(samet)
    acct = np.sum(samet)/len(train_y)
    losst = Loss.cross_entropy().value(train_y, predt)
    print(f"Training data tests: {wrongt:5}, {losst:0.4}, {acct:0.4}")

# %%

if __name__ == "__main__":
    plt.scatter(np.arange(nn.epoch)[:200], nn.J_h[:200])

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
















































































