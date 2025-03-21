import numpy as np
import copy
import math

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import Loss
import activation

class GDRegressor():
    def __init__(self, alpha=1e-2, num_iters=1000, verbose=False, batch_size=None):
        
        self.alpha = alpha
        self.num_iters = num_iters
        self.verbose = verbose
        self.batch_size = batch_size
        
    def fit(self, x, y) -> None:
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y, self.alpha, self.num_iters, self.verbose)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
        
    def predict(self, x):
        y_pred = np.matmul(x, self.params[0]) + self.params[1]
        return y_pred
            
    def compute_gradients(self, x, y, w, b):
        num_ex = x.shape[0]

        f_wb = (np.sum(np.matmul(x,w), axis=1) + b).reshape(num_ex, 1)
        dj_dw = 1/num_ex * np.sum(x*(f_wb - y), axis=0).reshape(-1,1)
        dj_db = 1/num_ex * np.sum(f_wb - y)

        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        num_ex = x.shape[0]
        
        if len(x.shape)==1:
            x = copy.copy(x.reshape(num_ex, -1))
            
        if len(y.shape)==1:
            y = copy.copy(y.reshape(num_ex, -1))
        
        w_initial = np.random.rand(x.shape[1], 1)
        b_initial = np.random.rand(1, 1)
        cost_function = Loss.mse
                
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
        
        for i in range(num_iters):
            for curr_x, curr_y in zip(x_batches,y_batches):
                # Calculate the gradient and update the parameters
                dj_dw, dj_db = self.compute_gradients(curr_x, curr_y, w, b )  
    
                # Update Parameters using w, b, alpha and gradient
                w = w - alpha * dj_dw               
                b = b - alpha * dj_db      
            
            w_history.append(w)
            b_history.append(b)
    
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion
                y_pred = (np.sum(np.matmul(x,w), axis=1, keepdims=True) + b)
                cost =  cost_function(y, y_pred)
                J_history.append(cost)
    
            if verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}") #, params: {[round(float(val), 2) for val in w]}, {b[0]:0.2f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing
    
    
class GDClassifier():
    def __init__(self, alpha=1e-2, num_iters=1000, verbose=False, batch_size=None):
        
        self.alpha = alpha
        self.num_iters = num_iters
        self.verbose = verbose
        self.batch_size = batch_size
        
    def fit(self, x, y) -> None:
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y, self.alpha, self.num_iters, self.verbose)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
        
    def predict(self, x):
        y_pred = np.array([np.random.multinomial(1,val) for val in activation.softmax(np.matmul(x,self.params[0]) + self.params[1])])
        return y_pred
            
    def compute_gradients(self, x, y, w, b):
        num_ex = x.shape[0]
        f_wb = activation.softmax(np.matmul(x,w) + b)
        dL_dy = Loss.dhinge(y, f_wb)
        dz = np.einsum('ijk,ik->ij', activation.dsoftmax(np.matmul(x, w) + b), dL_dy)

        dj_dw = 1/num_ex * np.matmul(x.T, dz)
        dj_db = 1/num_ex * np.sum(dz, axis=0, keepdims=True)

        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        num_ex = x.shape[0]
        num_classes = y.shape[1]
        print(num_ex, num_classes)
        
        if len(x.shape)==1:
            x = copy.copy(x.reshape(num_ex, -1))
            
        if len(y.shape)==1:
            y = copy.copy(y.reshape(num_ex, -1))
        
        w_initial = np.random.rand(x.shape[1], num_classes)*1e-3
        b_initial = np.random.rand(1, num_classes)*1e-3
        cost_function = Loss.hinge
                
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
        
        for i in range(num_iters):
            for curr_x, curr_y in zip(x_batches,y_batches):
                # Calculate the gradient and update the parameters
                dj_dw, dj_db = self.compute_gradients(curr_x, curr_y, w, b )  
        
                # Update Parameters using w, b, alpha and gradient
                w = w - alpha * dj_dw               
                b = b - alpha * dj_db
            
            w_history.append(w)
            b_history.append(b)
    
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion
                y_pred = activation.softmax(np.matmul(x,w) + b)
                #print(y_pred)
                cost =  cost_function(y, y_pred)
                J_history.append(cost)
    
            if verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.4f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing


















































































