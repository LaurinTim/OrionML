import numpy as np
import matplotlib.pyplot as plt


import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML")

import OrionML as orn

# %%

#example where y depends only on 1 variable

x = np.random.rand(100)*10
y = x*4.1 + 2 + ((np.random.rand(100)-0.5)*2)

res = orn.method.GDRegressor(x, y, alpha=1e-2, num_iters=1000, verbose=True)

w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = w_pred*x + b_pred

plt.figure()
plt.scatter(x, y, c='r')
plt.plot([0,10], [w_pred*0+b_pred, w_pred*10+b_pred])

# %%

#example where y depends on multiple variable, in this case 2

x0 = np.random.rand(1000).reshape(-1,1)*10
x1 = np.random.rand(1000).reshape(-1,1)*10
x = np.concatenate((x0, x1), axis=1)
y = np.sum(np.array([[1.1, -0.3]])*x, axis=1) + 1.2 + ((np.random.rand(1000)-0.5))

res = orn.method.GDRegressor(x, y, alpha=0.01, num_iters=10000, verbose=True)
w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = np.matmul(x, w_pred) + b_pred

