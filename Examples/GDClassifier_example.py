import numpy as np

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML")

import OrionML as orn

#example for the gradient descent classifier
#there are 6 features which are assigned to 1 of three classes, the higher the first two features the more like a sample is part of the first class,
#the higher the next two features are the more likely it is that a sample is in the second class and the higher the last two features are the more 
#likely it is that a sample is in the third class

x = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]])
a = orn.method.GDClassifier(x, y, alpha=1e-1, num_iters=1000, verbose=True)

w, b = a.params

jh, wh, bh = a.history

wc = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
bc = np.array([[0,0,0]])
