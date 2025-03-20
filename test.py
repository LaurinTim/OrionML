import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML")

import OrionML as orn

# %%

x = np.random.rand(100)*10
y = x*4.1 + 2 + (np.random.rand(100)*2)**2

# %%

res = orn.method.GDRegressor(x, y, alpha=1e-2, num_iters=1000, verbose=True)

w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = w_pred*x + b_pred

# %%

plt.figure()
plt.scatter(x, y, c='r')
plt.plot([0,10], [w_pred*0+b_pred, w_pred*10+b_pred])

# %%

x0 = np.random.rand(100).reshape(-1,1)*10
x1 = np.random.rand(100).reshape(-1,1)*10
x = np.concatenate((x0, x1), axis=1)
y = np.sum(np.array([[1.1, -0.3]])*x, axis=1) + 1.2 + ((np.random.rand(100)-0.5))
# %%

plt.figure()
plt.scatter(x0, y)
plt.figure()
plt.scatter(x1, y)

# %%

res = orn.method.GDRegressor(x, y, alpha=0.01, num_iters=1000, verbose=True)
w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = w_pred*x + b_pred

# %%

y_pred = np.sum(np.array([[1.1585027, -0.22851711]])*x, axis=1) + 0.41479037

# %%

err = np.sum((y-y_pred)**2)/100