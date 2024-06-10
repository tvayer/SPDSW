# quad_prod_acceleration

# %%
import matplotlib.pyplot as plt
import torch
import time
from torch import einsum
import numpy as np


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    st = time.time()
    tmp = einsum("bi, bij, bj->b", vec, H, vec)
    ed = time.time()
    return ed - st, tmp


def _quad_prod_2(H, vec):
    st = time.time()
    tmp = torch.sum((vec[:, None, :] @ H) * vec[:, None, :], dim=(-1, 1))
    ed = time.time()
    return ed - st, tmp


def _quad_vmap(H, vec):

    def quad_measurements(X, vec):
        # vec is size p
        # X is size (p, p)
        # return vec[None, :] @ X @ vec[:, None]
        # return torch.sum(X * (vec[:, None] @ vec[None, :]))
        return torch.dot(vec, torch.mv(X, vec))
    # vec is size (nb, p)
    # H is size (nb, p, p)
    batch_quad = torch.vmap(quad_measurements, in_dims=0)
    st = time.time()
    tmp = batch_quad(H, vec).ravel()
    ed = time.time()
    return ed - st, tmp

# %%


p = 50
nbatch = 5
H = torch.zeros((nbatch, p, p))
vec = torch.zeros((nbatch, p))

for i in range(nbatch):
    A = torch.randn(p, p)
    B = A @ A.T + torch.eye(p)
    H[i, :, :] = B
    vec[i, :] = torch.rand(p)
t_vmap, res_vmap = _quad_vmap(H, vec)
# %%
t_vmap, res_vmap = _quad_vmap(H, vec)
t_quad, res_quad = _quad_prod(H, vec)
t_quad_2, res_quad_2 = _quad_prod_2(H, vec)
print('res quad = {}, \n res quad 2 = {}'.format(t_quad, res_quad_2))
print('t quad = {}, \n res quad 2 = {}'.format(t_quad, res_quad_2))
print('t vmap = {}, \n res vmap = {}'.format(t_vmap, res_vmap))
# %%
no = 10
times_quad = []
times_quad_2 = []
times_vmap = []
all_p = [int(m) for m in np.logspace(1, 3.5, no)]
nbatch = 300
print(all_p)
# %%
for p in all_p:
    H = torch.zeros((nbatch, p, p))
    vec = torch.zeros((nbatch, p))
    for i in range(nbatch):
        A = torch.randn(p, p)
        B = A @ A.T + torch.eye(p)
        H[i, :, :] = B
        vec[i, :] = torch.rand(p)
    t_quad, res_quad = _quad_prod(H, vec)
    times_quad.append(t_quad)

    t_quad_2, res_quad_2 = _quad_prod_2(H, vec)
    times_quad_2.append(t_quad_2)

    t_vmap, res_vmap = _quad_vmap(H, vec)
    times_vmap.append(t_vmap)

    print('p = {} done '.format(p))
# %%
plt.loglog(all_p, times_quad, label='time einsum', lw=2, marker='o')
plt.loglog(all_p, times_quad_2, label='time no einsum', lw=2, marker='o')
plt.loglog(all_p, times_vmap, label='time vmap', lw=2, marker='o')
plt.xlabel('dim $p$')
plt.ylabel('time (in sec.)')
plt.grid()
plt.legend()
# %%
