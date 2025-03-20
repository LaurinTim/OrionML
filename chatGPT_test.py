import math
import numpy as np
import copy

class Loss():
    def __init__(self):
        '''
        
        Loss functions: Mean Squared Error (mse), Mean Absolute Error (mae), 
        Mean Bias Error (mbe), Cross-Entropy Loss (cross_entropy), Hinge Loss (hinge)
        
        '''
        return
    
    def hinge(y, y_pred):
        '''

        Parameters
        ----------
        y : ndarray
            Correct labels.
        y_pred : ndarray
            Predicted labels.

        Returns
        -------
        float
            Hinge Loss of the correct and predicted labels.

        '''    
        size = y.shape[0]
        
        l = np.sum(np.clip(1 - np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf))
    
        return l / size
    
    def dhinge(y, y_pred):        
        l = np.array(np.clip(1-np.clip(np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf), a_min=0, a_max=np.inf), dtype=bool).reshape(-1,1)
        return -1*l*y

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def dsoftmax(z):
    sz = softmax(z)
    res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
    res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]))
    return res
    
class GDClassifier():
    def __init__(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        
        w, b, J_history, w_history, b_history = self.gradient_descent(x, y, alpha, num_iters, verbose)
        
        self.params = (w, b)
        self.history = (J_history, w_history, b_history)
            
    def compute_gradients(self, x, y, w, b):
        num_ex = x.shape[0]
        num_classes = y.shape[1]
        f_wb = softmax(np.matmul(x,w) + b)
        print((Loss.dhinge(y, f_wb).reshape(num_ex,1,num_classes)*dsoftmax(np.matmul(x,w) + b)).shape, x.shape)
        dj_dw = 1/num_ex * np.sum(Loss.dhinge(y, f_wb).reshape(num_ex,1,num_classes) * dsoftmax(np.matmul(x,w) + b) * x, axis=0)
        dj_db = 1/num_ex * np.sum(f_wb - y)
        
        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, alpha=1e-2, num_iters=1000, verbose=False):
        num_ex = x.shape[0]
        num_classes = y.shape[1]
        
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
                y_pred = softmax(np.sum(w*x, axis=1, keepdims=True) + b)
                cost =  cost_function(y, y_pred)
                J_history.append(cost)
    
            if verbose == True:
                # Print cost every at intervals 10 times or as many iterations if < 10
                if i% math.ceil(num_iters/10) == 0:
                    print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}, params: {[round(float(val), 2) for val in w]}, {b[0]:0.2f}")
                                        
        return w, b, J_history, w_history, b_history #return w and J,w history for graphing
                
    
x = np.array([[1,0,0,0,0,0], [0,0,1,0,0,0]])
y = np.array([[1,0,0], [0,1,0]])
a = GDClassifier(x, y, num_iters=10)