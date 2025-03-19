import numpy as np
import copy
import math

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

from LossFile import Loss

class GDLinear():
    def __init__(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y, alpha, num_iters, verbose)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
            
    def compute_gradients(self, x, y, w, b):
        num_ex = x.shape[0]
            
        f_wb = (np.sum(w*x, axis=1) + b).reshape(num_ex, 1)
        dj_dw = 1/num_ex * np.sum(x*(f_wb - y), axis=0)
        dj_db = 1/num_ex * np.sum(f_wb - y)
        
        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        num_ex = x.shape[0]
        
        if len(x.shape)==1:
            x = copy.copy(x.reshape(num_ex, -1))
            
        if len(y.shape)==1:
            y = copy.copy(y.reshape(num_ex, -1))
        
        w_initial = np.random.rand(x.shape[1])
        b_initial = np.random.rand(1)
        cost_function = Loss.mse
                
        J_history = []
        w_history = []
        b_history = []
        w = copy.deepcopy(w_initial)
        b = b_initial
        
        for i in range(num_iters):
            # Calculate the gradient and update the parameters
            dj_dw, dj_db = self.compute_gradients(x, y, w, b )  
    
            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw               
            b = b - alpha * dj_db      
            
            w_history.append(w)
            b_history.append(b)
    
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion
                y_pred = (np.sum(w*x, axis=1) + b).reshape(num_ex, 1)
                cost =  cost_function(y, y_pred)
                J_history.append(cost)
    
            if verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}, params: {[round(float(val), 2) for val in w]}, {b[0]:0.2f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing
                