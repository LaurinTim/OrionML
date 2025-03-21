import numpy as np
import matplotlib.pyplot as plt


import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML")

import OrionML as orn

# %%

#example where y depends only on 1 variable

np.random.seed(0)

x = np.random.rand(100, 1)*10
y = x*4.1 + 2 + ((np.random.rand(100, 1)-0.5))

res = orn.method.GDRegressor(loss_function="squared_error", learning_rate=0.01, num_iters=100, verbose=True, batch_size=8, penalty="L1", l=0.01, l0=0.5)
res.fit(x, y)

w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = res.predict(x)

plt.figure()
plt.scatter(x, y, c='r')
plt.plot([0,10], [(w_pred*0+b_pred)[0], (w_pred*10+b_pred)[0]])

# %%

#example where y depends on multiple variable, in this case 2

np.random.seed(1)

x0 = np.random.rand(1000).reshape(-1,1)*10
x1 = np.random.rand(1000).reshape(-1,1)*10
x = np.concatenate((x0, x1), axis=1)
y = np.sum(np.array([[1.1, -0.3]])*x, axis=1) + 1.2 + ((np.random.rand(1000)-0.5))

res = orn.method.GDRegressor(loss_function="squared_error", learning_rate=0.001, num_iters=200, verbose=True, batch_size=4, penalty="L2", l=0.01, l0=0.5)
res.fit(x, y)
w_pred, b_pred = res.params
J_history, w_history, b_history = res.history
y_pred = res.predict(x)

