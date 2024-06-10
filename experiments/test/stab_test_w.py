# %% Stability test
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from spdsw.spodnet import _a_outer
from torch import einsum


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)


class NanError(Exception):
    pass


class SpodNet(nn.Module):

    def __init__(self, p, K=1, gy=1.0):
        super().__init__()

        self.p = p
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.K = K
        self.gy = gy

    def one_pass(self, Theta):

        indices = torch.arange(self.p)

        for col in range(self.p):
            if torch.isnan(Theta).any():
                raise NanError('Theta has nan')

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of Theta
            Theta_11 = Theta[_11]
            theta_22 = Theta[_22]
            theta_12 = Theta[_12]

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]
            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)

            # Inv block W_11
            theta_12_next = torch.randn(
                self.p-1).type(torch.float64).unsqueeze(0)

            theta_22_next = self.gy + _quad_prod(W_11, theta_12_next)
            # Update Theta
            Delta = torch.zeros_like(Theta)
            Delta[_22] = theta_22_next - theta_22
            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12
            Theta = Theta + Delta
            self.log['lambda_min_Theta'].append(
                torch.linalg.eigvalsh(Theta.squeeze())[0].item())

            # update W
            w_22_next = 1.0 / \
                (theta_22_next - _quad_prod(inv_Theta_11, theta_12_next))
            w_12_next = einsum('b, bij, bj -> bi', -w_22_next,
                               inv_Theta_11, theta_12_next)
            self.W[_11] = (
                inv_Theta_11 + _a_outer(1.0 / w_22_next, w_12_next, w_12_next)).detach()
            self.W[_12] = w_12_next.detach()
            self.W[_21] = w_12_next.detach()
            self.W[_22] = w_22_next.detach()

        return Theta

    def forward(self, Theta):
        """ Forward pass. """

        self.W = torch.linalg.inv(Theta).detach()

        for k in range(0, self.K):
            Theta = self.one_pass(Theta)

        return Theta


test = 50
# for _ in range(test):
p = 50
A = torch.randn(p, p)
Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
Theta_true = torch.linalg.inv(Cov_true).type(torch.float64)

#%%
K = 50
spodnet = SpodNet(K=K,
                  p=p,
                  gy=5)

Theta_new = spodnet.forward(Theta_true[None, :, :])

print(spodnet.W @ Theta_new)
print(torch.allclose(spodnet.W @ Theta_new,
                     torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3))

fs = 13
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.plot(spodnet.log['schur'], lw=2, label='schur')
ax.plot(spodnet.log['lambda_min_Theta'], lw=2,
        label='$\\lambda_{min}(\\Theta)$')
ax.hlines(spodnet.gy, xmin=0, xmax=p*spodnet.K,
          linestyle='--', color='k', label='gy', lw=2)
for f in range(spodnet.K):
    if f == 0:
        ax.vlines(p*(f+1), ymin=0, ymax=spodnet.gy,
                  linestyle='--', color='grey', label='Full pass', lw=2)
    else:
        ax.vlines(p*(f+1), ymin=0, ymax=spodnet.gy,
                  linestyle='--', color='grey', lw=2)
ax.set_yscale('log')
ax.set_xlabel('iter', fontsize=fs)
ax.legend(fontsize=fs)
ax.grid()
fig.suptitle('Dimension p = {}'.format(p), fontsize=fs+2)

# Theta_new

# %%
p = 10
A = torch.randn(p, p)
Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
torch.linalg.eigvalsh(Cov_true)
# %%
spodnet.log['lambda_min_Theta']
# %%
