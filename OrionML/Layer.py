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
    
# %%

if __name__ == "__main__":
    
    a = np.array([[5.704978663144137, 5.6725052661433235, 5.441349806741868, 0.0, 0.0, 5.447274267945885, 0.0, 5.337731717958393, 5.304539129759923, 0.0, 0.0, 5.627774854167119, 0.0, 5.404477135633962, 0.0, 5.317173515716561, 0.0, 5.243865710439962, 0.0, 0.0, 5.437848537307963, 0.0, 0.0, 5.398077522551578, 0.0]])
    
    w = np.array([[ 0.0176298 , -0.09600292,  0.01791177,  0.01759975, -0.1014484,  0.00623568, -0.10340144, -0.08959691,  0.00311199, -0.10301925], 
          [ 0.0178451 , -0.09555231,  0.01801215,  0.01786708, -0.0984683,  0.00554143, -0.10257039, -0.08870446,  0.00311527, -0.10161922], 
          [ 0.0183559 , -0.09405028,  0.01773849,  0.01790214, -0.0966749,  0.00578105, -0.09930253, -0.08563243,  0.00101286, -0.09983591], 
          [ 0.01076109, -0.00861482,  0.01153643,  0.01144243, -0.0081826, -0.00823844, -0.00813569,  0.01065884, -0.00991703, -0.00984071], 
          [ 0.01039345, -0.00882974,  0.01105837,  0.01200073, -0.0092506, -0.00811234, -0.0097542 ,  0.0112151 , -0.0092461 , -0.00959542], 
          [ 0.01770807, -0.09474378,  0.01705367,  0.01699269, -0.0991171,  0.00666818, -0.10118577, -0.08680445,  0.00121563, -0.09956204], 
          [ 0.01004713, -0.00855991,  0.01113328,  0.01182283, -0.0094490, -0.00894403, -0.00930715,  0.01013194, -0.0088655 , -0.00981463], 
          [ 0.01829557, -0.09299499,  0.01724751,  0.01795049, -0.0991419,  0.0052628 , -0.09970061, -0.08690403,  0.00156926, -0.09933553], 
          [ 0.01815642, -0.09410143,  0.01785061,  0.01684889, -0.0980485,  0.00602151, -0.09942808, -0.08763173,  0.00126278, -0.10068859], 
          [-0.31057145,  0.32015315, -0.32552139, -0.33171226,  0.3166842, -0.33601308,  0.27049601,  0.33701218, -0.31632559,  0.31463527], 
          [ 0.00907825, -0.0066739 ,  0.00825959, -0.00182697, -0.0002418, -0.01137223,  0.00331355,  0.01527324, -0.01226915, -0.0060197 ], 
          [ 0.01808766, -0.0954443 ,  0.01760052,  0.01805486, -0.0997927,  0.00696104, -0.10152043, -0.08684483,  0.00315238, -0.10045347], 
          [ 0.01171074, -0.00878651,  0.01048995,  0.01021481, -0.0090961, -0.00943181, -0.00875178,  0.01051114, -0.00824989, -0.00816469], 
          [ 0.01817908, -0.09393514,  0.01789736,  0.01739167, -0.0992196,  0.00612223, -0.09934711, -0.08742991,  0.00154475, -0.10048707], 
          [ 0.01013429, -0.00809733,  0.01021982,  0.01059921, -0.0110676, -0.00873715, -0.00827329,  0.01116659, -0.00985555, -0.00834807], 
          [ 0.01798062, -0.09396025,  0.01670088,  0.01780988, -0.1001718,  0.00517511, -0.09948576, -0.08780806,  0.00207816, -0.10032192], 
          [ 0.00988505, -0.00695874,  0.01086016,  0.00559781, -0.0147819, -0.00840829, -0.00858335,  0.01236246, -0.01021851, -0.01029805], 
          [ 0.01854431, -0.09138085,  0.01812776,  0.01700263, -0.0965811,  0.00448162, -0.09749243, -0.08450287,  0.00127507, -0.09810267], 
          [ 0.0105087 , -0.00873558,  0.01134659,  0.01147669, -0.0083656, -0.00987092, -0.00910778,  0.0105415 , -0.00808708, -0.00848179], 
          [ 0.00710186, -0.00134638,  0.00248874, -0.00714306,  0.0088162, -0.01237526,  0.0071054 ,  0.01544465, -0.01317186,  0.00051594], 
          [ 0.01812275, -0.09290051,  0.01778887,  0.01726438, -0.0991372,  0.00483826, -0.09978262, -0.08541523,  0.00250114, -0.09980818], 
          [ 0.01071178, -0.00947083,  0.01011887,  0.0109893 , -0.0100255, -0.00809682, -0.00899316,  0.01062075, -0.00952626, -0.00911173], 
          [ 0.01048941, -0.00849828,  0.01075499,  0.01065694, -0.0087023, -0.00847548, -0.00987339,  0.0103526 , -0.0080393 , -0.00911676], 
          [ 0.01809798, -0.09437516,  0.01698723,  0.01778928, -0.0991244,  0.00546793, -0.10041516, -0.086998  ,  0.00253734, -0.09924498], 
          [-0.31021636,  0.32246735, -0.32660415, -0.33309632,  0.3144542, -0.33452494,  0.27608805,  0.33464105, -0.3131086 ,  0.31446892]])
    
    b = np.array([[-0.2670652 ,  0.26470863,  0.09646136,  0.15517183,  0.25665678,
           -0.04831743,  0.25673403,  0.25377899,  0.1602995 ,  0.23992642]])
                                                                              
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
    
# %%

if __name__ == "__main__":
    l = BatchNorm(3)
    
    a = np.array([[0,1,2],[-1,4,3],[2,0,-1], [5,1,1]])
    da = np.array([[-3,-2,5], [1,1,2], [0,-2,3], [0,1,0]])
    
    aw, c = l.forward(a, training=True)
    dx, dg, db = l.backward(da, c, training=True)

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
























































































