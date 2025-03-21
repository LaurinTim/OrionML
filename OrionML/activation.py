import numpy as np
from time import time

def linear(z):
    return z

def dlinear(z):
    return np.ones(z.shape)

def relu(z):
    return np.clip(z, a_min=0, a_max=np.inf)

def drelu(z):
    return np.array(z>0, dtype=float)

def elu(z, alpha=1):
    return np.clip(z, a_min=0, a_max=np.inf) + (np.array(z<0, dtype=float))*alpha*(np.exp(z)-1)

def delu(z, alpha=1):
    return np.array(z>0, dtype=float) + (np.array(z<0, dtype=float))*alpha*np.exp(z)

def leakyrelu(z, alpha=0.1):
    return np.clip(z, a_min=0, a_max=np.inf) + alpha*np.clip(z, a_min=-np.inf, a_max=0)

def dleakyrelu(z, alpha=0.1):
    return np.array(z>0, dtype=float) + alpha*np.array(z<0, dtype=float)

def softplus(z):
    return np.log(1+np.exp(z))

def dsoftplus(z):
    return 1/(1+np.exp(z))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return sig*(1-sig)

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def dtanh(z):
    res = tanh(z)
    return 1 - res**2

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def dsoftmax(z):
    sz = softmax(z)
    res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
    res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]))
    return res
 