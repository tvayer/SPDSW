# %%
import numpy as np
import matplotlib.pyplot as plt


def f(x, lamda=0.5, thresh=1e-3):
    return lamda/(1+np.exp(-lamda*x))+thresh


m = 1000
x = np.linspace(-10, 10, m)
y = f(x, lamda=10)
plt.plot(x, y)
# %%
f(-1000, 10)
# %%
