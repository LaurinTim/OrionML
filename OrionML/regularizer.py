import numpy as np

class L1Regularizer():
    def __init__(self, l=0.01):
        self.l = l
        
    def value(self, w):
        return self.l*np.sum(abs(w))
    
    def derivative(self, w):
        return self.l * np.sign(w)

class L2Regularizer():
    def __init__(self, l=0.01):
        self.l = l
        
    def value(self, w):
        return self.l*np.sum(w**2)
    
    def derivative(self, w):
        return 2*self.l*w

class ElasticNetRegularizer():
    def __init__(self, l=0.01, l0=0.5):
        self.l = l
        self.l0 = l0
        self.L1Reg = L1Regularizer(l=1-self.l0)
        self.L2Reg = L2Regularizer(l=self.l0)
        
    def value(self, w):
        return self.l*(self.L1Reg.value(w) + self.L2Reg.value(w))
    
    def derivative(self, w):
        return self.l*(self.L1Reg.derivative(w) + self.L2Reg.derivative(w))
    
class NoRegularizer():
    def __init__(self):
        return
    
    def value(self, w):
        return 0
    
    def derivative(self, w):
        return 0
        