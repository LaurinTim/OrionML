import numpy as np
import copy
import math

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import Loss
import activation
import regularizer

class GDRegressor():
    def __init__(self, loss_function="squared_error", learning_rate=1e-2, num_iters=1000, verbose=False, batch_size=None, epsilon=0.1, penalty=None, l=0.01, l0=0.5):
        '''
        Linear model fitted by minimizing a regularized empirical loss with GD.

        Parameters
        ----------
        loss_function : str, optional
            The loss function to be used. The available functions are: {squared_error, L1, L2}. 
            The default is "squared_error".
        learning_rate : float, optional
            Learning rate used in GD. The default is 1e-2.
        num_iters : int, optional
            Number of iterations over the training data. The default is 1000.
        verbose : bool, optional
            Whether the loss should be printed periodically during training. The default is False.
        batch_size : int/None, optional
            Batch size used for the training. If set to None, the whole training set is trained 
            on simultaniously. The default is None.
        epsilon : float, optional
            The value of epsilon for the loss functions "L1" and "L2". The default is 0.1.
        penalty : str/None, optional
            The regularizion technique that is used. The available regularizors are: 
            {L1, L2, Elastic}. If set to None, no regularization is applied. The default is None.
        l : float, optional
            Constant that is multiplied with the regularization term. The default is 0.01.
        l0 : float, optional
            Mixing parameter for the elastic regularizer. The elastic regularizer is a combination 
            of L1 and L2 regularization. L1 regularization is weighed by l0 and L2 regularization 
            by 1-l0. Values must be in the range [0,1). The default is 0.5.

        '''
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.verbose = verbose
        self.batch_size = batch_size
        
        if loss_function=="squared_error" or loss_function=="mse":
            self.cost_function = Loss.mse()
            
        elif loss_function=="L1":
            self.cost_function = Loss.L1loss(epsilon=epsilon)
            
        elif loss_function=="L2":
            self.cost_function = Loss.L2loss(epsilon=epsilon)
            
        else:
            print("Invalid Loss function. Please select one of {squared_error, L1, L2}.")
            
        if penalty==None:
            self.reg = regularizer.NoRegularizer()
            
        elif penalty=="L1":
            self.reg = regularizer.L1Regularizer(l=l)
            
        elif penalty=="L2":
            self.reg = regularizer.L2Regularizer(l=l)
            
        elif penalty=="Elastic":
            self.reg = regularizer.ElasticNetRegularizer(l=l, l0=l0)
            
        else:
            print("Invalid regularizer. Please select one of {L1, L2, Elastic} or None.")
        
    def fit(self, x, y) -> None:
        '''
        Finds weights and bias to fit data to the target.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Training data.
        y : ndarray, shape: (number of samples, 1)
            Traget values.

        '''
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
        
    def predict(self, x):
        '''
        Predict using the linear model.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.

        Returns
        -------
        y_pred : ndarray, shape: (number of samples, 1)
            Predicted target values of each element in x.

        '''
        y_pred = np.matmul(x, self.params[0]) + self.params[1]
        return y_pred
            
    def compute_gradients(self, x, y, w, b):
        '''
        Get the gradient of the loss function with respect to the weights and the bias.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.
        y : ndarray, shape: (number of samples, 1)
            Traget values.
        w : ndarray, shape: (number of features, 1)
            Weights at which the gradient is calculated.
        b : ndarray, shape: (1, 1)
            Bias at which the gradient is calculated.

        Returns
        -------
        dj_dw : ndarray, shape: (number of features, 1)
            Gradient of the loss function with respect to the weights.
        dj_db : ndarray, shape: (1, 1)
            Gradient of the loss function with respect to the bias.

        '''
        num_ex = x.shape[0]

        f_wb = (np.sum(np.matmul(x,w), axis=1) + b).reshape(num_ex, 1)
        dL_dy = self.cost_function.derivative(y, f_wb)
        dj_dw = (np.sum(x*(dL_dy), axis=0)).reshape(-1,1) + self.reg.derivative(w)
        dj_db = np.sum(dL_dy)

        return dj_dw, dj_db
    
    def gradient_descent(self, x, y):
        '''
        Get weights and bias to fit the input to the target using gradient descent.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.
        y : ndarray, shape: (number of samples, 1)
            Traget values.

        Returns
        -------
        w : ndarray, shape: (number of features, 1)
            Weights of the model.
        b : ndarray, shape: (1, 1)
            Bias of the model.
        J_history : list
            List containing the loss at each iteration.
        w_history : list
            List containing the weights at each iteration.
        b_history : list
            List containing the bias at each iteration.

        '''
        num_ex = x.shape[0]
        
        if len(x.shape)==1:
            x = copy.copy(x.reshape(num_ex, -1))
            
        if len(y.shape)==1:
            y = copy.copy(y.reshape(num_ex, -1))
        
        w_initial = np.random.rand(x.shape[1], 1)
        b_initial = np.random.rand(1, 1)
                
        J_history = []
        w_history = []
        b_history = []
        w = copy.deepcopy(w_initial)
        b = b_initial
        
        if self.batch_size==None:
            x_batches = [x]
            y_batches = [y]
        else:
            x_batches = [x[i:i+self.batch_size] for i in range(0, num_ex, self.batch_size)]
            y_batches = [y[i:i+self.batch_size] for i in range(0, num_ex, self.batch_size)]
        
        for i in range(self.num_iters):
            for curr_x, curr_y in zip(x_batches,y_batches):
                # Calculate the gradient and update the parameters
                dj_dw, dj_db = self.compute_gradients(curr_x, curr_y, w, b )  
    
                # Update Parameters using w, b, learning_rate and gradient
                w = w - self.learning_rate * dj_dw               
                b = b - self.learning_rate * dj_db      
            
            w_history.append(w)
            b_history.append(b)
    
            # Save cost J at each iteration
            y_pred = (np.sum(np.matmul(x,w), axis=1, keepdims=True) + b)
            cost =  self.cost_function.value(y, y_pred) + self.reg.value(w)
            J_history.append(cost)
    
            if self.verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(self.num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.4f}") #, params: {[round(float(val), 2) for val in w]}, {b[0]:0.2f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing
    
class GDClassifier():
    def __init__(self, loss_function="squared_error", learning_rate=1e-2, num_iters=1000, verbose=False, batch_size=None, epsilon=0.1, penalty=None, l=0.01, l0=0.5):
        '''
        Linear classifier with GD training.

        Parameters
        ----------
        loss_function : str, optional
            The loss function to be used. The available functions are: 
            {squared_error, hinge, squared_hinge, cross_entropy, L1, L2}. The default is "squared_error".
        learning_rate : float, optional
            Learning rate used in GD. The default is 1e-2.
        num_iters : int, optional
            Number of iterations over the training data. The default is 1000.
        verbose : bool, optional
            Whether the loss should be printed periodically during training. The default is False.
        batch_size : int/None, optional
            Batch size used for the training. If set to None, the whole training set is trained 
            on simultaniously. The default is None.
        epsilon : float, optional
            The value of epsilon for the loss functions "L1" and "L2". The default is 0.1.
        penalty : str/None, optional
            The regularizion technique that is used. The available regularizors are: 
            {L1, L2, Elastic}. If set to None, no regularization is applied. The default is None.
        l : float, optional
            Constant that is multiplied with the regularization term. The default is 0.01.
        l0 : float, optional
            Mixing parameter for the elastic regularizer. The elastic regularizer is a combination 
            of L1 and L2 regularization. L1 regularization is weighed by l0 and L2 regularization 
            by 1-l0. Values must be in the range [0,1). The default is 0.5.

        '''
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.verbose = verbose
        self.batch_size = batch_size
        self.activation = activation.softmax()
        
        if loss_function=="squared_error" or loss_function=="mse":
            self.cost_function = Loss.mse()
            
        elif loss_function=="hinge":
            self.cost_function = Loss.hinge()
            
        elif loss_function=="squared_hinge":
            self.cost_function = Loss.squared_hinge()
            
        elif loss_function=="cross_entropy" or loss_function=="logistic_regression" or loss_function=="log_loss":
            self.cost_function = Loss.cross_entropy()
            
        elif loss_function=="L1":
            self.cost_function = Loss.L1(epsilon=epsilon)
            
        elif loss_function=="L2":
            self.cost_function = Loss.L2(epsilon=epsilon)
            
        else:
            print("Invalid Loss function. Please select one of {squared_error, hinge, squared_hinge, cross_entropy, L1, L2}.")
            
        if penalty==None:
            self.reg = regularizer.NoRegularizer()
            
        elif penalty=="L1":
            self.reg = regularizer.L1Regularizer(l=l)
            
        elif penalty=="L2":
            self.reg = regularizer.L2Regularizer(l=l)
            
        elif penalty=="Elastic":
            self.reg = regularizer.ElasticNetRegularizer(l=l, l0=l0)
            
        else:
            print("Invalid regularizer. Please select one of {L1, L2, Elastic} or None.")
        
    def fit(self, x, y) -> None:
        '''
        Finds weights and bias assign the correct class to the training data.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Training data.
        y : ndarray, shape: (number of samples, number of classes)
            Traget classes.

        '''
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
        
    def predict(self, x):
        '''
        Predict the class of the elements in x.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.

        Returns
        -------
        y_pred : ndarray, shape: (number of samples, number of classes)
            Predicted target class of each element in x.

        '''
        y_pred = np.array([np.random.multinomial(1,val) for val in self.activation.value(np.matmul(x,self.params[0]) + self.params[1])])
        return y_pred
            
    def compute_gradients(self, x, y, w, b):
        '''
        Get the gradient of the loss function with respect to the weights and the bias.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.
        y : ndarray, shape: (number of samples, number of classes)
            Traget values.
        w : ndarray, shape: (number of features, number of classes)
            Weights at which the gradient is calculated.
        b : ndarray, shape: (1, number of classes)
            Bias at which the gradient is calculated.

        Returns
        -------
        dj_dw : ndarray, shape: (number of features, number of classes)
            Gradient of the loss function with respect to the weights.
        dj_db : ndarray, shape: (1, number of classes)
            Gradient of the loss function with respect to the bias.

        '''
        num_ex = x.shape[0]
        f_wb = self.activation.value(np.matmul(x,w) + b)
        #dL_dy = self.cost_function.derivative(y, np.array([np.random.multinomial(1,val) for val in f_wb]))
        dL_dy = self.cost_function.derivative(y, f_wb)
        dz = np.einsum('ijk,ik->ij', self.activation.derivative(np.matmul(x, w) + b), dL_dy)

        dj_dw = 1/num_ex * np.matmul(x.T, dz) + self.reg.derivative(w)
        dj_db = 1/num_ex * np.sum(dz, axis=0, keepdims=True)

        return dj_dw, dj_db
    
    def gradient_descent(self, x, y):
        '''
        Get weights and bias to fit the input to the target classes using gradient descent.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, number of features)
            Input data.
        y : ndarray, shape: (number of samples, number of classes)
            Traget values.

        Returns
        -------
        w : ndarray, shape: (number of features, number of classes)
            Weights of the model.
        b : ndarray, shape: (1, number of classes)
            Bias of the model.
        J_history : list
            List containing the loss at each iteration.
        w_history : list
            List containing the weights at each iteration.
        b_history : list
            List containing the bias at each iteration.

        '''
        num_ex = x.shape[0]
        num_classes = y.shape[1]
        
        if len(x.shape)==1:
            x = copy.copy(x.reshape(num_ex, -1))
            
        if len(y.shape)==1:
            y = copy.copy(y.reshape(num_ex, -1))
        
        w_initial = np.random.rand(x.shape[1], num_classes)*1e-3
        b_initial = np.random.rand(1, num_classes)*1e-3
                
        J_history = []
        w_history = []
        b_history = []
        w = copy.deepcopy(w_initial)
        b = b_initial
        
        if self.batch_size==None:
            x_batches = [x]
            y_batches = [y]
        else:
            x_batches = [x[i:i+self.batch_size] for i in range(0, num_ex, self.batch_size)]
            y_batches = [y[i:i+self.batch_size] for i in range(0, num_ex, self.batch_size)]
        
        for i in range(self.num_iters):
            for curr_x, curr_y in zip(x_batches,y_batches):
                # Calculate the gradient and update the parameters
                dj_dw, dj_db = self.compute_gradients(curr_x, curr_y, w, b )  
        
                # Update Parameters using w, b, learning_rate and gradient
                w = w - self.learning_rate * dj_dw               
                b = b - self.learning_rate * dj_db
            
            w_history.append(w)
            b_history.append(b)
    
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion
                y_pred = self.activation.value(np.matmul(x,w) + b)
                #print(y_pred)
                #cost =  self.cost_function.value(y, np.array([np.random.multinomial(1,val) for val in y_pred]))
                cost =  self.cost_function.value(y, y_pred) + self.reg.value(w)
                J_history.append(cost)
    
            if self.verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(self.num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.4f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing


















































































