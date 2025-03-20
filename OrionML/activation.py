import numpy as np

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def dsoftmax(z):
    sz = softmax(z)
    res = -sz.reshape(sz.shape[0],-1,1) * sz.reshape(sz.shape[0],1,sz.shape[1])
    res = res + np.einsum("ij,jk->ijk", sz, np.eye(sz.shape[1]))
    return res