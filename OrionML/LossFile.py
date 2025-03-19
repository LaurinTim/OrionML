import numpy as np

class Loss():
    def __init__(self):
        '''
        
        Loss functions: Mean Squared Error (mse), Mean Absolute Error (mae), 
        Mean Bias Error (mbe), Cross-Entropy Loss (cross_entropy), Hinge Loss (hinge)
        
        '''
        return
    
    def mse(y, y_pred):
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
            Mean Squared Error of the correct and predicted labels.

        '''
        return np.sum( ( y - y_pred ) ** 2 ) / np.size( y )
    
    def mae(y, y_pred):
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
            Mean Absolute Error of the correct and predicted labels.

        '''
        return np.sum( np.abs( y - y_pred ) ) / np.size( y )
    
    def mbe( y, y_pred ) :
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
            Mean Bias Error of the correct and predicted labels.

        '''
        return np.sum( y - y_pred ) / np.size( y )
    
    def cross_entropy(y, y_pred):
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
            Cross Entropy Loss of the correct and predicted labels.

        '''
        return - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / np.size(y)
    
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
        l = 0
    
        size = np.size(y)
    
        for i in range(size):
    
            l = l + max(0, 1 - y[i] * y_pred[i])
    
        return l / size