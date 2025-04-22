import numpy as np
from pathlib import Path
import sys

import os
sys.path.insert(0, str(Path(os.path.abspath('')).resolve().parent))
os.chdir(Path(os.path.abspath('')).resolve().parent)

import OrionML as orn

#example for the gradient descent classifier
#there are 6 features which are assigned to 1 of three classes, the higher the first two features the more like a sample is part of the first class,
#the higher the next two features are the more likely it is that a sample is in the second class and the higher the last two features are the more 
#likely it is that a sample is in the third class

x = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]])
a = orn.method.GDClassifier(num_iters=1000, verbose=True, batch_size=1)
a.fit(x, y)

w, b = a.params

jh, wh, bh = a.history

y_pred = a.predict(x)
