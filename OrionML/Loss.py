import numpy as np
import copy


class mse():
    def __init__(self):
        return
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Mean Squared Error of the correct and predicted labels.
    
        '''
        return np.sum( ( y - y_pred ) ** 2 ) / y.shape[0]
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Mean Squared Error of the correct and predicted labels.
    
        '''
        return 2/y.shape[0] * (y_pred-y)

class mae():
    def __init__(self):
        return
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Mean Absolute Error of the correct and predicted labels.
    
        '''
        return np.sum( np.abs( y - y_pred ) ) / y.shape[0]
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Mean Absolute Error of the correct and predicted labels.
    
        '''
        return 1/y.shape[0] * np.sign(y-y_pred)

class mbe():
    def __init__(self):
        return
    
    def value(self, y, y_pred ) :
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Mean Bias Error of the correct and predicted labels.
    
        '''
        return np.sum( y - y_pred ) / y.shape[0]
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Mean Bias Error of the correct and predicted labels.
    
        '''
        return -1/y.shape[0]

class cross_entropy():
    def __init__(self):
        return
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Cross Entropy Loss of the correct and predicted labels.
    
        '''
        #print(np.min(y * np.log(y_pred)), np.max(y * np.log(y_pred)))
        #print(np.min((1 - y) * np.log(1 - y_pred)), np.max((1 - y) * np.log(1 - y_pred)))
        r = - np.sum(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1-y_pred + 1e-8)) / y.shape[0]
        return r
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Cross Entropy Loss of the correct and predicted labels.
    
        '''
        return (y_pred-y) / (y.shape[0]*y_pred*(1-y_pred) + 1e-8)
    

class hinge():
    def __init__(self):
        return
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Hinge Loss of the correct and predicted labels.
    
        '''
        yc = copy.copy(y)
        yc[yc==0] = -1
        l = np.sum(np.clip(1 - yc*y_pred, a_min=0, a_max=np.inf))
        return l / np.size(yc)
    
        #l = np.sum(np.clip(1 - np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf))
        #return l / y.shape[0]
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Hinge Loss of the correct and predicted labels.
    
        '''
        yc = copy.copy(y)
        yc[yc==0] = -1
        l = np.array(np.clip(1-np.clip(yc*y_pred, a_min=0, a_max=np.inf), a_min=0, a_max=np.inf), dtype=bool)
        return -1*l*yc
    
        #l = np.array(np.clip(1-np.clip(np.sum(y*y_pred, axis=1), a_min=0, a_max=np.inf), a_min=0, a_max=np.inf), dtype=bool).reshape(-1,1)
        #return -1*l*y
    
class squared_hinge():
    def __init__(self):
        return
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Hinge Loss of the correct and predicted labels.
    
        '''
        yc = copy.copy(y)
        yc[yc==0] = -1
        l = np.sum(np.clip(1 - yc*y_pred, a_min=0, a_max=np.inf)**2)
        return l / np.size(yc)
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Hinge Loss of the correct and predicted labels.
    
        '''
        yc = copy.copy(y)
        yc[yc==0] = -1
        l = np.array(np.clip(1-np.clip(yc*y_pred, a_min=0, a_max=np.inf), a_min=0, a_max=np.inf), dtype=bool)
        return -2*yc*l*(1-yc*y_pred)

class L1loss():
    def __init__(self, epsilon=0.1):
        '''

        Parameters
        ----------
        epsilon : float
            Losses below epsilon are ignored and all others are scaled down by epsilon.
            The default is 0.1.

        '''
        self.epsilon = epsilon
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            L1 Loss of the correct and predicted labels.
    
        '''
        return np.sum(np.clip(np.abs(y-y_pred)-self.epsilon, a_min=0, a_max=np.inf))
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the L1 Loss of the correct and predicted labels.
    
        '''
        diff = y_pred-y
        return np.sign(diff) * np.array(np.abs(diff)>self.epsilon, dtype=float)

class L2loss():
    def __init__(self, epsilon=0.1):
        '''

        Parameters
        ----------
        epsilon : float
            Losses below epsilon are ignored and all others are scaled down by epsilon.
            The default is 0.1.

        '''
        self.epsilon=epsilon
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            L2 Loss of the correct and predicted labels.
    
        '''
        return np.sum(np.clip((y-y_pred)**2-self.epsilon, a_min=0, a_max=np.inf))
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the L2 Loss of the correct and predicted labels.
    
        '''
        diff = y-y_pred
        return -2*(diff) * np.array(np.abs(diff**2)>self.epsilon, dtype=float)
    
class huber():
    def __init__(self, delta=1):
        '''

        Parameters
        ----------
        delta : float
            Value of delta for the Huber loss.

        '''
        self.delta=delta
    
    def value(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Huber Loss of the correct and predicted labels.
    
        '''
        diff = y_pred-y
        L_small = 1/2 * (y_pred-y)**2
        L_large = self.delta * np.abs(y_pred - y) - 1/2 * self.delta**2
        mask = np.abs(diff)<=self.delta
        L = L_small*mask + L_large*(1-mask)
        return L
    
    def derivative(self, y, y_pred):
        '''
    
        Parameters
        ----------
        y : ndarray, shape: (number of samples, number of outputs)
            Correct labels.
        y_pred : ndarray, shape: (number of samples, number of outputs)
            Predicted labels.
    
        Returns
        -------
        float
            Derivative of the Huber Loss of the correct and predicted labels.
    
        '''
        diff = y_pred-y
        L_small = y_pred-y
        L_large = np.sign(y_pred-y) * self.delta
        mask = np.abs(diff)<=self.delta
        L = L_small*mask + L_large*(1-mask)
        return L











































































