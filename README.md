# Tools for Machine Learning built using numpy arrays.

In this repository, there are tools for different machine learning tasks, programmed wholly in python, mainly using numpy.
As a first step, I created simple Regression and Classification methods. Both of these methods use gradient descent. The regression method is similar to the SGDRegressor from sklearn, if a batch size of 1 is chosen. Similarly, the classification method is similar to SGDClassifier from sklearn, if a batch size of 1 is chosen.
As a next step, I built tools for neural networks. With these tools, any neural network architecture, using the supported functions, can be built. All tools for neural networks, that are implemented currently, are listed below.

## Simple Regression and Classification methods

Regression and classification models with gradient descent can be trained. 
For regression the following loss functions can be used:

  * mean squared error
  * L1 (epsilon insentive) 
  * L2 (squared epsilon insentive)

For classification, in addition to the three above, the following loss functions can also be used:

  * hinge
  * squared hinge
  * cross entropy (logistics function) 

Regularization can be performed with the L1 or L2 norm, or a mix of both with the "Elastic" regularizer. 

The most common activation functions are also available: 

  * linear
  * ReLU
  * eLU
  * leaky ReLU
  * softplus
  * sigmoid
  * tanh
  * softmax

## Neural Networks

There is a class for neural networks (OrionML.NeuralNetwork) that implements the following layers:

 * Linear
 * Dropout
 * BatchNorm
 * Convolutional
 * Pool
 * BatchNorm2D
 * Reshape
 * Flatten

The Dropout layer works for both outputs from a linear and a convolutional layer. BatchNorm is for the output of a linear layer, BatchNorm2D for the output of a convolutional layer. For pooling layers, max and average pooling are available. Additionally, all activation functions mentioned before can be chosen to apply to the output of Linear and Convolutional layers. The following loss functions are available:

 * mean squared error
 * mean average error
 * mean bias error
 * cross entropy (logistics)
 * hinge
 * squared hinge
 * L1 (epsilon insentive) 
 * L2 (squared epsilon insentive)
 * Huber

Lastly, the initialization of the parameters for the layers with trainable parameters can be one of the following:

 * Xavier/glorot
 * He
 * Uniform

## Other Tools

There are some other tools that can be used to prepare data in OrionML.utils. A function to split data into training and test sets and a function to plot a confusion matrix is available. Additionally, a class for a standard scaler and a min max scaler can be used. Lastly, there are functions for im2col and col2im algorithms, which are already used for convolutional layers.

## Examples

There are some examples available in the Examples directory. 

For regression, the GDRegressor_example_Boston_housing notebook contains an example predicting house prices in Boston based on 13 features.

For classification, animal_faces_GDCLassifier_example.py contains an example where pictures are classified into 12 categories, based on which animal can be seen on the picture. To minimize computational power, a histogram of oriented gradients (hog) is created for each image. An accuracy of around 90% is reached in the validation set. The same data is also evaluated with the SGDClassifier from sklearn, with which again an accuracy around 90% is reached.

For Neural Networks, MNIST_NN_example contains an example for a Neural Netwok. In this example, this MNIST dataset consisting of images of handwritten digits between 0 and 9. The goal is to use image classification to figure out which number is written on the images.
