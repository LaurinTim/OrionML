## Tools for Machine Learning built using numpy arrays.

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

## Examples

There are some examples available in the examples directory. 

For regression, the GDRegressor_example_Boston_housing notebook contains an example predicting house prices in Boston based on 13 features.

For classification, animal_faces_GDCLassifier_example.py contains an example where pictures are classified into 12 categories, based on which animal can be seen on the picture. To minimize computational power, a histogram of oriented gradients (hog) is created for each image. An accuracy of around 90% is reached in the validation set. The same data is also evaluated with the SGDClassifier from sklearn, with which again an accuracy around 90% is reached.

## Neural Networks

There is a class for neural networks (OrionML.NeuralNetwork) that implements Linear (Dense) layers and Dropout layers. All activations mentioned above can be chosen for the Linear layers. In addition, the following loss functions can be chosen:

 * mean squared error
 * mean average error
 * mean bias error
 * cross entropy (logistics)
 * hinge
 * squared hinge
 * L1 (epsilon insentive) 
 * L2 (squared epsilon insentive)
