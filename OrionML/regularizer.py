import numpy as np

class L1Regularizer():
    def __init__(self, l=0.01):
        '''
        L1 regularizer.

        Parameters
        ----------
        l : float, optional
            Constant that is multiplied with the regularization term. The default is 0.01.

        '''
        self.l = l
        
    def value(self, w):
        '''
        Get the L1 regularization term for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the L1 regularization is calculated.

        Returns
        -------
        float
            L1 regularization term for w.

        '''
        return self.l*np.sum(abs(w))
    
    def derivative(self, w):
        '''
        Get the derivative of L1 regularization for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the L1 regularization is calculated.

        Returns
        -------
        ndarray, shape: (input size, output size)
            Partial derivative of L1 regularization with respect to each element in w.

        '''
        return self.l * np.sign(w)

class L2Regularizer():
    def __init__(self, l=0.01):
        '''
        L1 regularizer.

        Parameters
        ----------
        l : float, optional
            Constant that is multiplied with the regularization term. The default is 0.01.

        '''
        self.l = l
        
    def value(self, w):
        '''
        Get the L2 regularization term for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the L2 regularization is calculated.

        Returns
        -------
        float
            L2 regularization term for w.

        '''
        return self.l*np.sum(w**2)
    
    def derivative(self, w):
        '''
        Get the derivative of L2 regularization for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the L2 regularization is calculated.

        Returns
        -------
        ndarray, shape: (input size, output size)
            Partial derivative of L2 regularization with respect to each element in w.

        '''
        return 2*self.l*w

class ElasticNetRegularizer():
    def __init__(self, l=0.01, l0=0.5):
        '''
        Elastic regularizer using a combination of L1 and L2 regularization.

        Parameters
        ----------
        l : float, optional
            Constant that is multiplied with the regularization term. The default is 0.01.
        l0 : float, optional
            Mixing parameter for the elastic regularizer. The elastic regularizer is a combination 
            of L1 and L2 regularization. L1 regularization is weighed by l0 and L2 regularization 
            by 1-l0. Values must be in the range [0,1). The default is 0.5.

        '''
        self.l = l
        self.l0 = l0
        self.L1Reg = L1Regularizer(l=self.l0)
        self.L2Reg = L2Regularizer(l=1-self.l0)
        
    def value(self, w):
        '''
        Get the elastic regularization term for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the elastic regularization is calculated.

        Returns
        -------
        float
            Elastic regularization term for w.

        '''
        return self.l*(self.L1Reg.value(w) + self.L2Reg.value(w))
    
    def derivative(self, w):
        '''
        Get the derivative of elastic regularization for w.

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights for which the elastic regularization is calculated.

        Returns
        -------
        ndarray, shape: (input size, output size)
            Partial derivative of elastic regularization with respect to each element in w.

        '''
        return self.l*(self.L1Reg.derivative(w) + self.L2Reg.derivative(w))
    
class NoRegularizer():
    def __init__(self):
        '''
        Regularizer which is used if no regularization term is used. Returns 0 for both 
        the value and derivative, regardless of the input.

        Returns
        -------
        None.

        '''
        return
    
    def value(self, w):
        '''

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights.

        Returns
        -------
        0

        '''
        return 0
    
    def derivative(self, w):
        '''

        Parameters
        ----------
        w : ndarray, shape: (input size, output size)
            Weights.

        Returns
        -------
        0

        '''
        return 0
        