import numpy as np
import math
import copy

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML\\OrionML")

import activation as activ
import utils

class Linear():
    
    def __init__(self, dim1, dim2, activation, bias=True, alpha=0.1):
        '''
        Linear Layer.

        Parameters
        ----------
        dim1 : int
            Size of the input sample.
        dim2 : int
            Size of the output sample.
        activation : str
            Activation to use for this Layer. The available activations are: 
            {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.
        bias : TYPE, optional
            Whether or not the Layer contains a bias. The default is True.
        alpha : float
            Only used if the activation is "leakyrelu". Slope of the leaky ReLU when z<0. 
            The default is 0.1.

        '''
        self.dim1 = dim1
        self.dim2 = dim2
        self.bias = bias
        self.alpha = alpha
        self.w = np.zeros((dim1, dim2))
        self.b = np.zeros((1, dim2))
        
        self.activation = activation
        self.activation_function = self.get_activation_function()
        
        self.trainable = True
        self.dimension = np.array([dim1, dim2])
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to linear Layers.

        '''
        return "OrionML.Layer.Linear"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the linear Layer with information about the input and output 
            dimension and the activation.

        '''
        return f"OrionML.Layer.Linear  (shape:({self.dim1, self.dim2}), activation: {self.activation})"
    
    def get_activation_function(self):
        '''
        Get the correct activation function from the string input activation.

        Returns
        -------
        Correct activation function class from OrionMl.activation.

        '''
        if self.activation == "linear": return activ.linear()
        elif self.activation == "relu": return activ.relu()
        elif self.activation == "elu": return activ.elu()
        elif self.activation == "leakyrelu": return activ.leakyrelu(alpha=self.alpha)
        elif self.activation == "softplus": return activ.softplus()
        elif self.activation == "sigmoid": return activ.sigmoid()
        elif self.activation == "tanh": return activ.tanh()
        elif self.activation == "softmax": return activ.softmax()
        else:
            assert False, "ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}."
            
    def update_parameters(self, w_new, b_new=None):
        '''
        Updade the weights and bias of the current Layer.

        Parameters
        ----------
        w_new : ndarray, shape: (self.dim1, self.dim2)
            New weights to replace the current ones of the linear Layer.
        b_new : ndarray/None, optional, shape: (1, self.dim2)
            New bias to replace the current one of the linear Layer. If self.bias is set to False 
            this must be None, otherwise a ndarray. The default is None.

        '''
        assert not (self.bias==True and b_new is None), "ERROR: Expected a bias when updating the parameters."
        assert not (self.bias == False and not b_new is None), "ERROR: This Layer has no bias but a new bias was passed along."
        
        self.w = w_new
        if self.bias==True: self.b = b_new
        
        return
        
    def value(self, x, training=None):
        '''
        Pass an input to the linear Layer to get the output after the weights, bias and 
        activation function is applied.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, self.dim1)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        ndarray, shape: (number of samples, self.dim2)
            Array after the weights, bias and activation function of the linear Layer are 
            applied to the input data.
        z : ndarray, shape: (number of samples, self.dim2)
            Array after the weights and bias of the linear Layer are applied to the input data.

        '''
        z = np.matmul(x, self.w) + self.b
        out = self.activation_function.value(z)
        if np.isnan(out).any():
            print("\nERROR IN LAYER VALUE: NAN FOUND")
            #print(out[:5])
            print(np.isnan(self.activation_function.value(z[350:400])).any())
            
            print(self.activation_function.value(np.array([z[393]])))
            print(z[393])

            for i in range(50):
                temp = self.activation_function.value(z[350+i:350+i+2])
                if np.isnan(temp).any():
                    print("-"*50)
                    print(i+350)
                    print()
                    print(list(z[350+i:350+i+2]))
                    print(np.max(z[350+i:350+i+2]))
                    print(np.max(z[350+i:350+i+2], axis=1, keepdims=True))
                    print("-"*50)
                    
        return out, z
    
    def derivative(self, x, training=None):
        '''
        Get the derivative of the activation function for the values after applying the 
        weights and bias of the linear Layer to the input data.

        Parameters
        ----------
        x : ndarray, shape: (number of samples, self.dim1)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        d_activ : ndarray, shape: (input size, output size, output size)
            Derivative of the activation function for the values after applying the 
            weights and bias of the linear Layer to the input data.

        '''
        z = np.matmul(x, self.w) + self.b
        d_activ = self.activation_function.derivative(z)
        return d_activ
    
    def forward(self, prev_A, training=None):
        '''
        Forward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Data before the current linear Layer is applied.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim2)
            Data after the current linear Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
                    Input for the current forward step.
                self.w : ndarray, shape: (self.dim1, self.dim2)
                    Weights of the current linear Layer.
                self.b : ndarray, shape: (1, self.dim2)
                    Bias of the current linear Layer. If the current Layer has no bias, an array 
                    with shape contianing 0's is returned.
                Z : ndarray, shape: (number of samples, self.dim2)
                    Array after the weights and bias of the linear Layer are applied to the input data.

        '''
        curr_A, Z = self.value(prev_A)
        cache = (prev_A, self.w, self.b, Z)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=None):
        '''
        Backward step of a linear Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples passed to the Neural Network, self.dim2)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current linear Layer. 
            For its contens, refer to the return of self.forward.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        prev_dA : ndarray, shape: (number of samples passed to the Neural Network, self.dim1)
            Derivative of all Layers in the Neural Network starting from the current Layer.
        curr_dw : ndarray, shape: (self.dim1, self.dim2)
            Derivative of the weights of the current Layer given dA and the values in the cache.
        curr_db : ndarray, shape: (1, self.dim2)
            Derivative of the bias of the current Layer given dA and the values in the cache.

        '''
        prev_A, curr_w, curr_b, curr_Z = cache
        d_activation = self.derivative(prev_A)
        if self.activation != "softmax":
            curr_dA = dA * d_activation
            
        else:
            curr_dA = np.einsum('ijk,ik->ij', d_activation, dA)
            
        curr_dw = 1/prev_A.shape[0] * np.matmul(prev_A.T, curr_dA)
        curr_db = 1/prev_A.shape[0] * np.sum(curr_dA, axis=0, keepdims=True)
        prev_dA = np.matmul(curr_dA, curr_w.T)
        
        return prev_dA, curr_dw, curr_db
    
       
class Dropout():
    def __init__(self, dropout_probability=0.3, scale=True):
        '''
        Dropout Layer.

        Parameters
        ----------
        dropout_probability : TYPE, optional
            Probability for each node to be set to 0. The default is 0.3.
        scale : bool, optional
            Whether the remaining weights should be scaled by 1/dropout_probability.
            The default is True.


        '''
        self.dropout_probability = dropout_probability   
        self.scale = scale
        
        self.trainable = False
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to dropout Layers.

        '''
        return "OrionML.Layer.Dropout"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the dropout Layer with information about dropout probability.

        '''
        return f"OrionML.Layer.Dropout (dropout probability: {self.dropout_probability})"
        
    def value(self, activation_output, training=False):
        '''
        Set each element in activation_output with probability dropout_probability to 0 if training is True.
        
        Parameters
        ----------
        activation_output : ndarray, shape: (input size, output size)
            Output after an activation function to pass through the dropout Layer.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Copy of activation_output but each element set to 0 with probability dropout_probability.
        mask : ndarray, shape: (input size, output size)
            Is only returned if training is set to True. An array that is 0 at every element in 
            activation_output that was set to 0, otherwise 1.

        '''
        if training==False:
            return activation_output, np.zeros(1)
        
        mask = np.random.rand(activation_output.shape[0], activation_output.shape[1]) > self.dropout_probability
        res = mask*activation_output
        if self.scale==True:
            res = res * 1/(1-self.dropout_probability)
            
        return res, mask
    
    def derivative(self, mask, training=False):
        '''
        Get the derivative of the dropout Layer.

        Parameters
        ----------
        mask : ndarray, shape: (input size, output size)
            Mask from the dropout Layer when it was applied
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied and the derivative is the same as for linear activation. The default is False.

        Returns
        -------
        ndarray, shape: (input size, output size)
            If training is False, return an array filled with ones. Otherwise return mask.

        '''
        if training==False: return np.ones(mask.shape)
        return mask
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a dropout Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (input size, output size)
            Data before the current dropout Layer is applied.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        curr_A : ndarray, shape: (input size, output size)
            Data after the current dropout Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (input size, output size)
                    Input for the current forward step.
                curr_mask : ndarray, shape: (input size, output size)
                    Mask used in the dropout Layer.
                
        '''
        curr_A, curr_mask = self.value(prev_A, training=training)
        cache = (prev_A, curr_mask)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step of a dropout Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current dropout Layer. 
            For its contens, refer to the return of self.forward.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting from the current Layer.

        '''
        prev_A, curr_mask = cache
        d_Layer = self.derivative(curr_mask, training=training)
        prev_dA = d_Layer * dA
        return prev_dA
    
    
class BatchNorm():
    def __init__(self, sample_dim, momentum=0.9, epsilon=1e-8):
        '''
        Batch normalization Layer.

        Parameters
        ----------
        


        '''
        self.epsilon = epsilon
        self.momentum = momentum
        self.sample_dim = sample_dim
        self.dimension = np.array([self.sample_dim])
        
        self.gamma = np.random.randn(1, self.sample_dim)
        self.beta = np.zeros((1, self.sample_dim))
        self.running_mean = np.zeros((1, self.sample_dim))
        self.running_variance = np.zeros((1, self.sample_dim))
        
        self.trainable = True
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to Batch normalization Layers.

        '''
        return "OrionML.Layer.BatchNorm"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the Batch normalization Layer with information about dropout probability.

        '''
        return "OrionML.Layer.BatchNorm"
        
    def value(self, x, training=False):
        '''
        
        Parameters
        ----------
        x : ndarray, shape: (input size, output size)
            Input for the batch normalization.
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied. The default is False.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Output of the batch normalization layer.

        '''
        if training==True:
            mean = np.mean(x, axis=0)
            variance = np.var(x, axis=0)
            x_normalized = (x-mean)/np.sqrt(variance + self.epsilon)
            
            out = self.gamma*x_normalized + self.beta
                    
            return out, x_normalized, mean, variance
        
        elif training==False:
            x_normalized = (x-self.running_mean)/np.sqrt(self.running_variance + self.epsilon)
            out = self.gamma*x_normalized + self.beta
            
            return out, ()
    
    def forward(self, prev_A, training=False):
        '''
        Forward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (input size, output size)
            Data before the current dropout Layer is applied.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        curr_A : ndarray, shape: (input size, output size)
            Data after the Batch normalization Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                DESCRIPTION.
                
        '''
        if training==True:
            curr_A, x_normalized, batch_mean, batch_variance = self.value(prev_A, training=training)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*batch_mean
            self.running_variance = self.momentum*self.running_variance + (1-self.momentum)*batch_variance
            
            cache = (prev_A, x_normalized, batch_mean, batch_variance)
            
        elif training==False:
            curr_A = self.value(prev_A, training=training)
        
            cache = (prev_A)
        
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        '''
        Backward step of a Batch normalization Layer in a Neural Network.

        Parameters
        ----------
        dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting after the current Layer.
        cache : tuple
            cache containing information from the forward propagation of the current dropout Layer. 
            For its contens, refer to the return of self.forward.
        training : bool, optional
            Whether the Layer is currently in training or not. The default is False.

        Returns
        -------
        prev_dA : ndarray, shape: (input size, output size)
            Derivative of all Layers in the Neural Network starting from the current Layer.

        '''
        assert training, "Training should be True for backward propagation."
        
        prev_A, x_normalized, batch_mean, batch_variance = cache
        
        dgamma = np.sum(dA*x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(dA, axis=0, keepdims=True)
        
        m = prev_A.shape[0]
        t = 1/np.sqrt(batch_variance + self.epsilon)
        curr_dA = (self.gamma * t/m) * (m*dA - np.sum(dA, axis=0) - t**2 * (prev_A-batch_mean) * np.sum(dA * (prev_A - batch_mean), axis=0))
        
        return curr_dA, dgamma, dbeta


class Conv():
    def __init__(self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.activation = activation
        self.activation_function = self.get_activation_function()
        
        self.trainable = True
        self.dimensions = np.array([self.kernel_size, self.kernel_size, self.in_channels, self.out_channels])
        
        self.w = np.ones((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
        self.b = np.zeros((1, self.out_channels))
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to convolutional Layers.

        '''
        return "OrionML.Layer.Conv"
        
    def description(self):
        '''

        Returns
        -------
        str
            Description of the convolutional Layer with information about the input and output 
            channels, the kernel size, the stride and the padding.

        '''
        return f"OrionML.Layer.Conv  (shape:({self.kernel_size, self.kernel_size, self.in_channels, self.out_channels}), stride: {self.stride}, padding: {self.padding})"
    
    def get_activation_function(self):
        '''
        Get the correct activation function from the string input activation.

        Returns
        -------
        Correct activation function class from OrionMl.activation.

        '''
        if self.activation == "linear": return activ.linear()
        elif self.activation == "relu": return activ.relu()
        elif self.activation == "elu": return activ.elu()
        elif self.activation == "leakyrelu": return activ.leakyrelu(alpha=self.alpha)
        elif self.activation == "softplus": return activ.softplus()
        elif self.activation == "sigmoid": return activ.sigmoid()
        elif self.activation == "tanh": return activ.tanh()
        else:
            assert False, "ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh}."
            
            
    def update_parameters(self, w_new, b_new=None):
        '''
        Updade the weights and bias of the current Layer.

        Parameters
        ----------
        w_new : ndarray, shape: (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            New weights to replace the current ones of the linear Layer.
        b_new : ndarray/None, optional, shape: (self.out_channels, self.kernel_size, self.kernel_size)
            New bias to replace the current one of the linear Layer. If self.bias is set to False 
            this must be None, otherwise a ndarray. The default is None.

        '''
        assert not (self.bias==True and b_new is None), "ERROR: Expected a bias when updating the parameters."
        assert not (self.bias == False and not b_new is None), "ERROR: This Layer has no bias but a new bias was passed along."
        
        self.w = w_new
        if self.bias==True: self.b = b_new
        
        return
        
    def value(self, A):
        '''
        Pass an input to the convolutional Layer to get the output after the weights and bias  
        are applied.

        Parameters
        ----------
        A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.

        Returns
        -------
        prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.
        prev_A_strided : ndarray, shape: (number of samples, output height, output width, kernel size, kernel size, input channels)
            Array containing the sliding windows of prev_A used for the convolution.

        '''
        # =============================================================================
        #  As a reminder: ndarray.strides gives the number of bytes to step until the next element is reached in each dimension. Each number in the arrays is of type np.float64, the last 
        #  Dimension of A.strides will be 64/8=4. 
        #  For the array np.array([[0,1,2],[3,4,5]]) b.strides is (12, 4) since each number is a 32 bit integer and thus there are 4 bytes for each number.
        #  The first dimension is filled with three 32 bit integers and thus the stride for the first dimension is 3*4=12.
        #  For the array np.array([[0,1,2],[3,4,5]], dtype=float) b.strides is (24, 8) since each number is a 64 bit float and thus there are 8 bytes for each number.
        #  The first dimension is filled with three 64 bit floats and thus the stride for the first dimension is 3*8=24. 
        # =============================================================================
            
        h_out = (A.shape[1] + 2*self.padding - self.kernel_size)//self.stride + 1
        w_out = (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1
        
        A_padded = np.copy(A)
        
        if self.padding > 0:
            A_padded = np.pad(A_padded, pad_width=((0,), (self.padding,), (self.padding,), (0,)), mode="constant", constant_values=(0.,))
        
        batch_stride, kern_h_stride, kern_w_stride, channel_stride = A_padded.strides
        strides = (batch_stride, self.stride*kern_h_stride, self.stride*kern_w_stride, kern_h_stride, kern_w_stride, channel_stride)
        
        A_strided = np.lib.stride_tricks.as_strided(A_padded, (A.shape[0], h_out, w_out, self.w.shape[0], self.w.shape[1], A.shape[3]), strides)
        
        #A_convoluted = np.einsum('abcijk,ijkd', A_strided, self.w) + self.b
        A_convoluted = np.tensordot(A_strided, self.w, axes=3) + self.b
        
        output = self.activation_function.value(A_convoluted)
        
        return output, (A_strided, A_convoluted)
    
    def forward(self, prev_A, training=None):
        '''
        Forward step of a convolutional Layer in a Neural Network.

        Parameters
        ----------
        prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
            Input data.
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for convolutional 
            Layers. The default is None.

        Returns
        -------
        curr_A : ndarray, shape: (number of samples, output height, output width, output channels)
            Data after the convolutional Layer is applied.
        cache : tuple
            Cache containing information needed in the backwards propagation. Its contents are:
                prev_A : ndarray, shape: (number of samples, input height, input width, input channels)
                    Input data.
                prev_A_strided : ndarray, shape: (number of samples, output height, output width, kernel size, kernel size, input channels)
                    Array containing the sliding windows of prev_A used for the convolution.

        '''
        curr_A, (prev_A_strided, A_convoluted) = self.value(prev_A)
        cache = (prev_A, prev_A_strided, A_convoluted)
        
        return curr_A, cache
    
    def backward(self, dA, cache):
        '''
        Backward step for a convolutional Layer.

        Parameters
        ----------
        dA : ndarray, shape: (number of samples, output height, output width, output channels)
            Upstream gradient.
        cache : tuple
            Cache containing information from the forwards propagation used in the backwards propagation.

        Returns
        -------
        curr_dA : ndarray, shape: (number of samples, input height, input width, input channels)
            Gradient to be passed as upstream gradient to the next layer in backwards propagation.
        db : ndarray, shape: (1, input channels)
            Gradient of the bias.
        dw : ndarray, shape: (kernel size, kernel size, input channels, output channels)
            Gradient of the weights.

        '''
        A, A_strided, A_convoluted = cache
        
        d_activ = self.activation_function.derivative(A_convoluted)
        dA = d_activ * dA
        
        dA_padded = np.copy(dA)
        back_padding = self.kernel_size - 1 if self.padding==0 else self.padding
                
        if self.stride > 1:
            dA_padded = np.insert(dA, range(1, dA.shape[1]), 0, axis=1)
            dA_padded = np.insert(dA_padded, range(1, dA.shape[2]), 0, axis=2)
                
        if back_padding > 0:
            dA_padded = np.pad(dA_padded, pad_width=((0,), (back_padding,), (back_padding,), (0,)), mode="constant", constant_values=(0.,))
        
        batch_stride, kern_h_stride, kern_w_stride, channel_stride = dA_padded.strides
        strides = (batch_stride, 1*kern_h_stride, 1*kern_w_stride, kern_h_stride, kern_w_stride, channel_stride)
        
        dA_strided = np.lib.stride_tricks.as_strided(dA_padded, (dA.shape[0], A.shape[1], A.shape[2], self.w.shape[0], self.w.shape[1], dA.shape[3]), strides)
        
        db = np.array([np.sum(dA, axis=(0, 1, 2))])
        
        #dw = np.einsum('bhwkli,bhwo->klio', A_strided, dA)
        dw = np.tensordot(A_strided, dA, axes=([0,1,2], [0,1,2]))
        
        #dx = np.einsum('bhwklo,klio->bhwi', dA_strided, np.rot90(self.w, 2, axes=(0,1)))
        curr_dA = np.tensordot(dA_strided, np.rot90(self.w, 2, axes=(0,1)), axes=([3,4,5], [0,1,3]))
        
        return curr_dA, db, dw

# %%

if __name__ == "__main__":
    in_channels = 3
    out_channels = 128
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = (4, in_channels, 12, 10)  # expected input size
    dout_size = (4, out_channels, 6, 5)   # expected to match the size of the layer outputs

    np.random.seed(42)  # for reproducibility

    x = np.random.random(batch_size)  # create data for forward pass
    dout = np.random.random(dout_size)  # create random data for backward
    
    xl = np.transpose(x, axes=(0,2,3,1))
    doutl = np.transpose(dout, axes=(0,2,3,1))
    '''
    print('x: ', x.shape)
    print('d_out: ', dout.shape)'''
    
    conv = Conv(in_channels, out_channels, kernel_size, stride, padding)
    
    conv_outl, c = conv.forward(xl)
    
# %%

if __name__ == "__main__":
    dxl, dbl, dwl = conv.derivative(doutl, c)
    '''
    print('conv_out: ', conv_outl.shape)
    print('db: ', dbl.shape)
    print('dw: ', dwl.shape)
    print('dx: ', dxl.shape)'''
    
# %%

if __name__ == "__main__":
    al = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    dal = np.ones((2, 2, 2, 3))
    
    l = Conv(2, 3, 2)
    l.w = np.ones((2,2,2,3))
    rl, c = l.forward(al)
    drl = l.backward(dal, c)
    
# %%

if __name__ == "__main__":
    al = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    dal = np.ones((2, 2, 2, 3))
    
    l = Conv(2, 3, 2, "linear")
    l.w = np.ones((2,2,2,3))
    arl, c = l.forward(al)
    darl = l.backward(dal, c)
    
# %%

if __name__ == "__main__":
    #2 samples, 2 channels, height=width=3
    a = np.array([[[[2,0,2],
                    [3,3,1],
                    [4,1,3]],
                   [[5,2,0],
                    [0,1,4],
                    [4,0,4]]],
                  
                  [[[5,2,3],
                    [0,0,2],
                    [4,5,2]],
                   [[1,4,4],
                    [3,5,0],
                    [0,2,1]]]])
    
    a = np.array([[[[2,5],[0,2],[2,0]],
                   [[3,0],[3,1],[1,4]],
                   [[4,4],[1,0],[3,4]]],
                  
                  [[[5,1],[2,4],[3,4]],
                   [[0,3],[0,5],[2,0]],
                   [[4,0],[5,2],[2,1]]]])
    
    dal = np.ones((2, 2, 2, 3))
    
    ns = 2
    oc = 2
    nc = 3
    ks = 2
    st = 1
    p = 0
    nh = (a.shape[2] + 2*p - ks)/st + 1
    nw = (a.shape[3] + 2*p - ks)/st + 1
    
    w = np.ones((ks, ks, oc, nc))
    #w[0][0][0][0] = 0
    ww = w*np.array([[[[1,10,100],[1,10,100]],[[1,10,100],[1,10,100]]],[[[1,10,100],[1,10,100]],[[1,10,100],[1,10,100]]]])
    '''w[:,0,0,1]=0
    w[:,0,1,0]=0
    w[:,1,0,0]=0
    w[:,1,1,1]=0'''
    
    b = np.zeros((nc, int(nh), int(nw)))
    
# %%

if __name__ == "__main__":
    Hout = a.shape[1] - w.shape[0] + 1
    Wout = a.shape[2] - w.shape[1] + 1
    
    aa = np.lib.stride_tricks.as_strided(a, (a.shape[0], Hout, Wout, w.shape[0], w.shape[1], a.shape[3]), a.strides[:3] + a.strides[1:])
    
    r = np.einsum('abcijk,ijkd', aa, w)
    rr = np.tensordot(aa, w, axes=3)
    
# %%

if __name__ == "__main__":
    l = Conv(2, 3, 2)
    l.w = w
    rl, c = l.value(a)
    drl = l.derivative(dal, c)
    
# %%

def value1(self, A, training=False):
         '''
         Apply pooling of type self.pool_mode to the input.
         
         Parameters
         ----------
         A : ndarray, shape: (input size, number of filters, dim1, dim2)
             array to apply the pooling to the second dimension
         training : bool/None, optional
             Whether the Layer is currently in training or not. This has no effect for linear 
             Layers. The default is None.
 
         Returns
         -------
         res : ndarray, shape: (input size, output size)
             Copy of activation_output but each element set to 0 with probability dropout_probability.
 
         '''
 # =============================================================================
 #       As a reminder: ndarray.strides gives the number of bytes to step until the next element is reached in each dimension. Each number in the arrays is of type np.float64, the last 
 #       Dimension of A.strides will be 64/8=4. 
 #       For the array np.array([[0,1,2],[3,4,5]]) b.strides is (12, 4) since each number is a 32 bit integer and thus there are 4 bytes for each number.
 #       The first dimension is filled with three 32 bit integers and thus the stride for the first dimension is 3*4=12.
 #       For the array np.array([[0,1,2],[3,4,5]], dtype=float) b.strides is (24, 8) since each number is a 64 bit float and thus there are 8 bytes for each number.
 #       The first dimension is filled with three 64 bit floats and thus the stride for the first dimension is 3*8=24. 
 # =============================================================================
         A_num_dims = A.ndim
         
         assert A_num_dims == 4, "Input to pooling layer has wrong number of dimensions. A needs 4 dimensions."
         
         A = np.pad(A, self.padding, mode="constant")
         
         output_shape = (A.shape[0], A.shape[1], (A.shape[2] + 2*self.padding - self.kernel_size) // self.stride + 1, (A.shape[3] + 2*self.padding - self.kernel_size) // self.stride + 1)
         w_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3], self.kernel_size, self.kernel_size)
         w_strides = (A.strides[0], A.strides[1], self.stride*A.strides[2], self.stride*A.strides[3], A.strides[2], A.strides[3])
         
         A_w = np.lib.stride_tricks.as_strided(A, w_shape, w_strides)
 
         if self.pool_mode=="max":
             A_w = A_w.max(axis=(4, 5))
         elif self.pool_mode=="avg":
             A_w = A_w.mean(axis=(4, 5))
         
         return A_w

# %%

class Pool():
    def __init__(self, kernel_size, stride, padding=0, pool_mode="max"):
        '''
        Pooling Layer.

        Parameters
        ----------
        kernel_size : int
            Size of the window over which the pooling is applied.
        stride : int
            Stride of the window.
        padding : int, optional
            Zeros padding on the sides of the input. The default is 0.
        pool_mode : str, optional
            What type of pooling to apply. Has to be one of {max, avg}. The default is "max".

        '''
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_mode = pool_mode
        
        self.trainable = False
        
    def type(self):
        '''

        Returns
        -------
        str
            String unique to pooling Layers.

        '''
        return "OrionML.Layer.Pool"
    
    def description(self):
        '''

        Returns
        -------
        str
            Description of the pooling Layer with information about the kernel size, stride, 
            padding and what type of pooling is applied.

        '''
        return f"OrionML.Layer.Pool (kernel size: {self.kernel_size}, stride: {self.stride}, padding: {self.padding}, pooling mode: {self.pool_mode})"
    
    
    def value(self, A, training=False):
        '''
        Apply pooling of type self.pool_mode to the input.
        
        Parameters
        ----------
        A : ndarray, shape: (input size, number of filters, dim1, dim2)
            array to apply the pooling to the second dimension
        training : bool/None, optional
            Whether the Layer is currently in training or not. This has no effect for linear 
            Layers. The default is None.

        Returns
        -------
        res : ndarray, shape: (input size, output size)
            Copy of activation_output but each element set to 0 with probability dropout_probability.

        '''
        A_w_cols = utils.im2col(A, self.kernel_size, self.kernel_size, self.stride)
        max_idx = np.argmax(A_w_cols.reshape(A.shape[1], self.kernel_size**2, -1), axis=1)
        
        #A_w = np.reshape(np.array([A_w_cols[:, 0::2], A_w_cols[:, 1::2]]), (A.shape[0], A.shape[1], self.kernel_size**2, ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1) * ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1)))
        A_w = np.reshape(np.array([A_w_cols[:, val::A.shape[0]] for val in range(A.shape[0])]), (A.shape[0], A.shape[1], self.kernel_size**2, ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1) * ((A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1)))
        
        if self.pool_mode=="max":
            A_w = np.max(A_w, axis=2)
            
        if self.pool_mode=="avg":
            A_w = np.mean(A_w, axis=2)
            
        cache = (A, A_w_cols.reshape(A.shape[1], self.kernel_size**2, -1), max_idx)
        
        return A_w.reshape(A.shape[0], A.shape[1], (A.shape[2] + 2*self.padding - self.kernel_size)//self.stride + 1, (A.shape[3] + 2*self.padding - self.kernel_size)//self.stride + 1), cache
    
    def derivative(self, A_prev, dA, x_cols, max_idx, training=False):
        '''
        Get the derivative of the pooling Layer.

        Parameters
        ----------
        mask : ndarray, shape: (input size, output size)
            Mask from the dropout Layer when it was applied
        training : bool, optional
            Whether the Layer is currently in training or not. If training is False, no dropout 
            is applied and the derivative is the same as for linear activation. The default is False.

        Returns
        -------
        ndarray, shape: (input size, output size)
            If training is False, return an array filled with ones. Otherwise return mask.

        '''        
        N, C, H, W = A_prev.shape

        # Reshape dout to match the dimensions of x_cols:
        # dout: (N, C, out_height, out_width) -> (C, 1, N*out_height*out_width)
        dA_reshaped = dA.transpose(1, 2, 3, 0).reshape(C, -1)
        
        if self.pool_mode=="max":
            # Initialize gradient for x_cols as zeros.
            dmax = np.zeros_like(x_cols)
            
            # Scatter the upstream gradients to the positions of the max indices.
            # For each channel and each pooling window, place the corresponding gradient at the max index.
            dmax[np.arange(C)[:, None], max_idx, np.arange(x_cols.shape[2])] = dA_reshaped
            
            # Reshape dmax back to the 2D column shape expected by col2im.
            dmax = dmax.reshape(C * self.kernel_size**2, -1)
            
            # Convert the columns back to the original image shape.
            dx = utils.col2im(dmax, A_prev.shape, self.kernel_size, self.kernel_size, self.stride)
            
        if self.pool_mode=="avg":
            dcols = np.repeat(dA_reshaped, self.kernel_size**2, axis=0) / (self.kernel_size**2)
            dx = utils.col2im(dcols, A_prev.shape, self.kernel_size, self.kernel_size, self.stride)

        return dx
    
    def forward(self, prev_A, training=False):
        curr_A, cache = self.value(prev_A, training=training)
        return curr_A, cache
    
    def backward(self, dA, cache, training=False):
        A_prev, A_w_cols, max_idx = cache
        dx = self.derivative(A_prev, dA, A_w_cols, max_idx, training=training)
        return dx
























































































