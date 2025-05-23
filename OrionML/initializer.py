import numpy as np
        
def glorot(layers):
    '''
    Initialize parameters with glorot/Xavier initialization.

    Parameters
    ----------
    layers : list
        List containing the layers for which the trainanble parameters should 
        be initialized.

    Returns
    -------
    parameters : dict
        Dictionary containing the parameters of the layers.
    derivatives : dict
        Dictionary containing arrays for the derivatines of the parameters.

    '''
    parameters = {}
    derivatives = {}
    
    for i, layer in enumerate(layers):
        if layer.trainable:
            curr_layer_type = layer.type()
            if curr_layer_type=="OrionML.Layer.Linear":
                limit = np.sqrt(6 / (layer.dim1 + layer.dim2))
                parameters[f"w layer {i}"] = np.random.uniform(-limit, limit, size=tuple(layer.dimension))
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dim2))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dim2))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1, 1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                    
            elif curr_layer_type=="OrionML.Layer.Conv":
                limit = np.sqrt(6 / (layer.dimension[0] * layer.dimension[1] * (layer.dimension[2] + layer.dimension[3])))
                parameters[f"w layer {i}"] = np.random.uniform(-limit, limit, tuple(layer.dimension))
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dimension[3]))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dimension[3]))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1,1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                    
            elif curr_layer_type in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                limit = np.sqrt(6 / layer.sample_dim)
                parameters[f"gamma layer {i}"] = np.random.uniform(-limit, limit, size=(1, layer.sample_dim)) + 1
                parameters[f"beta layer {i}"] = np.zeros((1, layer.sample_dim))
                derivatives[f"dgamma layer {i}"] = np.zeros((1, layer.sample_dim))
                derivatives[f"dbeta layer {i}"] = np.zeros((1, layer.sample_dim))
                layer.gamma = parameters[f"gamma layer {i}"]
                layer.beta = parameters[f"beta layer {i}"]
                
    return parameters, derivatives


def he(layers):
    '''
    Initialize parameters with He initialization.

    Parameters
    ----------
    layers : list
        List containing the layers for which the trainanble parameters should 
        be initialized.

    Returns
    -------
    parameters : dict
        Dictionary containing the parameters of the layers.
    derivatives : dict
        Dictionary containing arrays for the derivatines of the parameters.

    '''
    parameters = {}
    derivatives = {}
    
    for i, layer in enumerate(layers):
        if layer.trainable:
            curr_layer_type = layer.type()
            if curr_layer_type=="OrionML.Layer.Linear":
                limit = np.sqrt(6 / layer.dim1)
                parameters[f"w layer {i}"] = np.random.uniform(-limit, limit, size=tuple(layer.dimension))
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dim2))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dim2))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1, 1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                    
            elif curr_layer_type=="OrionML.Layer.Conv":
                limit = np.sqrt(6 / (layer.dimension[0] * layer.dimension[1] * layer.dimension[2]))
                parameters[f"w layer {i}"] = np.random.uniform(-limit, limit, tuple(layer.dimension))
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dimension[3]))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dimension[3]))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1,1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                
            elif curr_layer_type in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                limit = np.sqrt(6 / layer.sample_dim)
                parameters[f"gamma layer {i}"] = np.random.uniform(-limit, limit, size=(1, layer.sample_dim)) + 1
                parameters[f"beta layer {i}"] = np.zeros((1, layer.sample_dim))
                derivatives[f"dgamma layer {i}"] = np.zeros((1, layer.sample_dim))
                derivatives[f"dbeta layer {i}"] = np.zeros((1, layer.sample_dim))
                layer.gamma = parameters[f"gamma layer {i}"]
                layer.beta = parameters[f"beta layer {i}"]
                
    return parameters, derivatives


def uniform(layers):
    '''
    Initialize parameters in a uniform distribution.

    Parameters
    ----------
    layers : list
        List containing the layers for which the trainanble parameters should 
        be initialized.

    Returns
    -------
    parameters : dict
        Dictionary containing the parameters of the layers.
    derivatives : dict
        Dictionary containing arrays for the derivatines of the parameters.

    '''
    parameters = {}
    derivatives = {}
    
    for i, layer in enumerate(layers):
        if layer.trainable:
            curr_layer_type = layer.type()
            if curr_layer_type=="OrionML.Layer.Linear":
                limit = np.sqrt(6 / layer.dim1)
                parameters[f"w layer {i}"] = np.random.rand(*tuple(layer.dimension)) * 1e-1/np.sqrt(layer.dim1)
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dim2))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dim2))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1, 1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                    
            elif curr_layer_type=="OrionML.Layer.Conv":
                limit = np.sqrt(6 / (layer.dimension[0] * layer.dimension[1] * layer.dimension[2]))  * 1/np.sqrt(layer.dimension[2])
                parameters[f"w layer {i}"] = np.random.uniform(-limit, limit, tuple(layer.dimension))
                derivatives[f"dw layer {i}"] = np.zeros(tuple(layer.dimension))
                if layer.bias:
                    parameters[f"b layer {i}"] = np.zeros((1, layer.dimension[3]))
                    derivatives[f"db layer {i}"] = np.zeros((1, layer.dimension[3]))
                    layer.update_parameters(parameters[f"w layer {i}"], parameters[f"b layer {i}"])
                else:
                    parameters[f"b layer {i}"] = np.zeros((1,1))
                    derivatives[f"db layer {i}"] = np.zeros((1,1))
                    layer.update_parameters(parameters[f"w layer {i}"])
                    
            elif curr_layer_type in ["OrionML.Layer.BatchNorm", "OrionML.Layer.BatchNorm2D"]:
                parameters[f"gamma layer {i}"] = np.random.rand(1, layer.sample_dim) * 1e-2/np.sqrt(layer.sample_dim) + 1
                parameters[f"beta layer {i}"] = np.random.rand(1, layer.sample_dim) * 1e-2/np.sqrt(layer.sample_dim)
                derivatives[f"dgamma layer {i}"] = np.zeros((1, layer.sample_dim))
                derivatives[f"dbeta layer {i}"] = np.zeros((1, layer.sample_dim))
                layer.gamma = parameters[f"gamma layer {i}"]
                layer.beta = parameters[f"beta layer {i}"]
                
    return parameters, derivatives