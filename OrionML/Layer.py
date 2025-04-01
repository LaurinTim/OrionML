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
        Correct activation function class from OrionMl.activation. If the value of self.activation 
        is not valid, an error is printed and a linear activation is used.

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
            print("ERROR: Invalid activation function. Please set activation to one of the following: {linear, relu, elu, leakyrelu, softplus, sigmoid, tanh, softmax}.")
            return activ.linear()
            
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
        return self.activation_function.value(z), z
    
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
        if self.activation == "softmax":
            curr_dA = np.einsum('ijk,ik->ij', d_activation, dA)
            
        else:
            curr_dA = dA * d_activation
            
        curr_dw = 1/prev_A.shape[0] * np.matmul(prev_A.T, curr_dA)
        curr_db = 1/prev_A.shape[0] * np.sum(curr_dA, axis=0, keepdims=True)
        prev_dA = np.matmul(curr_dA, curr_w.T)
        
        return prev_dA, curr_dw, curr_db
    
# %%

if __name__ == "__main__":
    a = np.array([[ 0.        ,  0.        ,  0.        , 14.15997885, 14.78391227, 14.05610726, 12.17075247,  0.        ,  0.        , 10.86832194, 
                    0.        ,  8.37061739,  0.        ,  0.        , 13.02588479, 0.         , 14.72502661,  0.        ,  0.        ,  0.        ,
                    0.        , 15.06996079, 14.00639234,  0.        ,  0.        ]])
    
    w = np.array([[-5.68375131, -5.68377928, -5.68370317, -5.6837172 , -5.68378457,
            5.70160855, -5.68385013, -5.68384838, -5.68368429, -5.68381163], [-5.68689634, -5.6869508 , -5.68685039, -5.68684987, -5.6868769 ,
            5.70189185, -5.68697452, -5.68696895, -5.68685626, -5.68688733], [-5.68707789, -5.68723437, -5.68708575, -5.68705492, -5.6871425 ,
            5.7020226 , -5.68709351, -5.68708093, -5.68719735, -5.68715048], [-5.66684685, -5.66680694, -5.66680001, -5.66680472, -5.66677756,
            5.69981565, -5.66676356, -5.6668834 , -5.66693847, -5.66693028], [-5.61692991, -5.61691561, -5.61693741, -5.61683701, -5.61696408,
            5.69416981, -5.6170143 , -5.61691205, -5.61695837, -5.61699297], [-5.20954866, -5.21003236, -5.20999737, -5.2099682 , -5.20992483,
            5.64443237, -5.21003584, -5.20990888, -5.20999099, -5.20985959], [-5.67318843, -5.67306553, -5.67310402, -5.67302997, -5.67317253,
            5.70044909, -5.67314431, -5.67320055, -5.67309736, -5.67319175], [-5.61282307, -5.61286941, -5.61290526, -5.61280544, -5.61294419,
            5.693555  , -5.61290238, -5.61295061, -5.61287688, -5.61285024], [-5.67937333, -5.67949172, -5.67941509, -5.67951043, -5.67934818,
            5.70113897, -5.67943263, -5.67954039, -5.67945862, -5.67949485], [-5.66008373, -5.66006471, -5.66007684, -5.66007087, -5.66003546,
            5.69894946, -5.66005612, -5.6600887 , -5.66010884, -5.66008088], [-5.61988639, -5.61984739, -5.6198559 , -5.61992763, -5.61990724,
            5.6943751 , -5.61985841, -5.61980959, -5.61978941, -5.61981713], [-5.2150457 , -5.21548494, -5.21540248, -5.21531662, -5.21536353,
            5.64515553, -5.21541304, -5.21530181, -5.21531126, -5.21532412], [-5.67930491, -5.67934808, -5.67942906, -5.67943534, -5.67930699,
            5.70110758, -5.67933585, -5.67946225, -5.67929013, -5.67928593], [-5.66771673, -5.66777304, -5.66774969, -5.66778602, -5.66775492,
            5.69982719, -5.66769564, -5.66781804, -5.66780013, -5.66776854], [-5.66837323, -5.66826799, -5.6683847 , -5.66828118, -5.66838551,
            5.69993775, -5.66828378, -5.66833155, -5.6683592 , -5.66823386], [-5.67266169, -5.67270299, -5.67278195, -5.67263703, -5.67278452,
            5.70030027, -5.6726423 , -5.67279027, -5.67264737, -5.67268562], [-5.68161328, -5.68160401, -5.68148782, -5.68157968, -5.68154726,
            5.70147152, -5.68161253, -5.68151088, -5.68152765, -5.68160191], [-5.67822086, -5.67829385, -5.67823676, -5.67835648, -5.67827999,
            5.70093442, -5.67828598, -5.67828071, -5.67829241, -5.67830894], [-5.65545267, -5.65537373, -5.65537409, -5.65535576, -5.65534742,
            5.69836656, -5.65541647, -5.65544887, -5.6553105 , -5.6553492 ], [-5.68167171, -5.68166728, -5.6817315 , -5.68171807, -5.68174459,
            5.70139665, -5.68166641, -5.68164699, -5.68174287, -5.68166426], [-5.67411914, -5.6741579 , -5.67412754, -5.67417745, -5.67423549,
            5.70046129, -5.6741992 , -5.6740936 , -5.67408174, -5.67418607], [-5.68008061, -5.6801098 , -5.6801599 , -5.68004999, -5.6800796 ,
            5.70131683, -5.6800535 , -5.68014458, -5.68011105, -5.68007382], [-5.54212052, -5.54214121, -5.54222935, -5.54222891, -5.54216887,
            5.68549144, -5.54228761, -5.54225231, -5.54209644, -5.54220421], [-5.65786365, -5.65797644, -5.65794686, -5.65784009, -5.65790878,
            5.69867937, -5.65793771, -5.65793504, -5.65781038, -5.65780982], [-5.68215914, -5.68211717, -5.68215121, -5.68216317, -5.68215247,
            5.70137081, -5.68211094, -5.68207857, -5.68217549, -5.68206562]])
                                                                              
    b = np.array([[-5.70372126, -5.70364477, -5.70360135, -5.70365413, -5.70364012,
            5.70387884, -5.70363339, -5.7037012 , -5.70365785, -5.70364209]])
    
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.540610991889495e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.314123602650649])
    
    w = np.array([[ 0.01423119, -0.08238678,  0.01677191,  0.01746933, -0.09073655,
            0.01058983, -0.09050491, -0.08344004,  0.03179502, -0.09277976], [ 0.01542936, -0.08300413,  0.01643495,  0.01760986, -0.08782761,
            0.00456018, -0.08933017, -0.08194094,  0.01672772, -0.09119366], [ 0.02214347, -0.0821694 ,  0.00981532,  0.01864001, -0.08284981,
           -0.02320888, -0.08287267, -0.07471278, -0.05009705, -0.08644869], [ 0.00875386, -0.00660787,  0.00953017,  0.00944169, -0.00620299,
           -0.00623215, -0.00613697,  0.00865401, -0.00791366, -0.00783506], [ 0.00029083, -0.09614467,  0.03046559,  0.01686955, -0.11213527,
            0.04032944, -0.108478  , -0.10384695,  0.07049555, -0.11325687], [ 0.02256142, -0.08219169,  0.00768964,  0.01787886, -0.08401486,
           -0.02518802, -0.08386992, -0.07501061, -0.05310749, -0.08529057], [ 0.00803486, -0.00656401,  0.00912219,  0.00973924, -0.00782366,
           -0.00695427, -0.00738894,  0.0083422 , -0.00690453, -0.0078259 ], [ 0.02295757, -0.08095604,  0.00794287,  0.0190425 , -0.08467362,
           -0.02542593, -0.08302001, -0.0756633 , -0.05138184, -0.08566723], [ 0.02384806, -0.08072419,  0.00779002,  0.01736418, -0.08154393,
           -0.0292375 , -0.08099265, -0.07468436, -0.05740035, -0.0850349 ], [ 0.00599297, -0.00443139,  0.00501135, -0.00356348,  0.00559916,
           -0.00911502,  0.00288501,  0.01137872, -0.01089896, -0.00404935], [ 0.00026651, -0.09664596,  0.03142027,  0.01548274, -0.1128415 ,
            0.03886466, -0.10689991, -0.10396423,  0.07089851, -0.11266599], [ 0.01795049, -0.08388348,  0.01365827,  0.01844045, -0.0881891 ,
           -0.00356591, -0.08748985, -0.07881672, -0.01383193, -0.08952631], [ 0.00711085, -0.09168441,  0.02363344,  0.01571806, -0.10482323,
            0.02942287, -0.09993433, -0.09762047,  0.05886071, -0.10558839], [ 0.02281386, -0.08119687,  0.009031  ,  0.01799021, -0.08385371,
           -0.02571681, -0.08185695, -0.07544795, -0.0527465 , -0.08595647], [ 0.00590884, -0.00435616,  0.00372886, -0.00556075,  0.01199568,
           -0.01070724, -0.00944949,  0.01552467, -0.01677853,  0.00566684], [ 0.02321712, -0.08173621,  0.00646649,  0.01911559, -0.085701  ,
           -0.02592707, -0.08270897, -0.07637931, -0.05103604, -0.08662833], [-0.00113849, -0.0978196 ,  0.0319546 ,  0.01454975, -0.11196062,
            0.04105913, -0.10844681, -0.10377828,  0.06974649, -0.11448731], [ 0.02588652, -0.07757429,  0.00591451,  0.01763733, -0.07930493,
           -0.03390051, -0.07837006, -0.07082974, -0.06150592, -0.08170548], [ 0.00850154, -0.0067287 ,  0.00934049,  0.00947616, -0.00637971,
           -0.00786482, -0.00711085,  0.00853728, -0.00608468, -0.00647635], [ 0.00899321, -0.00539505,  0.01116197,  0.00972615, -0.00858115,
           -0.00923048, -0.00859316, -0.00136597, -0.00055276, -0.00780531], [ 0.02256511, -0.08014957,  0.00920264,  0.0177896 , -0.08382291,
           -0.0266176 , -0.08235303, -0.07348032, -0.05116498, -0.08533668], [ 0.00838731, -0.00693434,  0.00761027,  0.00464626, -0.01004931,
           -0.00634517, -0.00624927,  0.01087422, -0.00906623, -0.00712131], [ 0.00848219, -0.00649125,  0.00874874,  0.00865461, -0.00670836,
           -0.0064694 , -0.00788055,  0.00834955, -0.00603862, -0.00711118], [ 0.02261649, -0.08229773,  0.00774705,  0.01895408, -0.08480402,
           -0.02437272, -0.08380255, -0.07583493, -0.04925583, -0.08572036], [-0.30725476,  0.30250286, -0.27807743, -0.32725684,  0.30814015,
           -0.30385317,  0.28011996,  0.31519118, -0.28410316,  0.30227038]])
    
    b = np.array([[-0.23947243,  0.25006803, -0.00070004,  0.13581317,  0.24314911,
           -0.01081268,  0.24520788,  0.24411459,  0.17961772,  0.23303293]])
                                                                              
    l = Linear(25, 10, activation="softmax")
    l.w = w
    l.b = b
    
    r, c = l.forward(a)

# %%    
       
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

# %%    

if __name__ == "__main__":
    a = np.array([[[[1, 1, 2, 4],
                    [5, 6, 7, 8],
                    [3, 2, 1, 0],
                    [1, 2, 3, 4]], 
                  
                   [[3, 2, 7, 4],
                    [8, 1, 4, 2],
                    [3, 1, 1, 2],
                    [5, 6, 2, 3]]]])
    
    l = Pool(3, 2, 0, pool_mode="max")
    aw, c = l.value(a)
    daw = l.backward(np.ones_like(aw), c)
    
# %%

if __name__ == "__main__":
    a = np.array([[[[1, 1, 2, 4],
                    [5, 6, 7, 8],
                    [3, 2, 1, 0],
                    [1, 2, 3, 4]], 
                  
                   [[3, 2, 7, 4],
                    [8, 1, 4, 2],
                    [3, 1, 1, 2],
                    [5, 6, 2, 3]]],
                  
                  [[[1, 1, 2, 4],
                    [5, 2, 7, 1],
                    [3, 2, 1, 0],
                    [1, 2, 5, 4]], 
                 
                   [[3, 2, 1, 4],
                    [1, 1, 4, 2],
                    [4, 1, 1, 2],
                    [5, 3, 2, 3]]]])
    
    l = Pool(2, 2, 0, pool_mode="max")
    aw, c = l.value(a)
    daw = l.backward(np.ones_like(aw), c)
























































































