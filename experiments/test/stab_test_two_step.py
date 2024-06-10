# %% Stability test
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from spdsw.spodnet import _a_outer
from torch import einsum


class NanError(Exception):
    pass


def prox_l1(x, alpha=1):
    return torch.sign(x) * torch.maximum(torch.abs(x) - alpha,
                                         torch.Tensor([[0.]]).type(
                                             torch.float64)
                                         )


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)


class SpodNet(nn.Module):

    def __init__(self, p, K=1, gy=1.0):
        super().__init__()

        self.p = p
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.log['norm_diff'] = []
        self.K = K
        self.gy = gy

    def one_pass(self, Theta):
        indices = torch.arange(self.p)
        for col in range(self.p):  # update col

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of Theta
            theta_12 = Theta[_12]
            w_12 = self.W[_12]
            theta_12_next = torch.randn(
                self.p-1).type(torch.float64).unsqueeze(0)
            Delta = torch.zeros_like(Theta)
            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12
            Theta = Theta + Delta

            # self.log['lambda_min_Theta'].append(
            #     torch.linalg.eigvalsh(Theta.squeeze())[0].item())
            # self.log['schur'].append(schur.squeeze().item())
        for col in range(self.p):  # update diag

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of Theta
            theta_12 = Theta[_12]
            theta_22 = Theta[_22]

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
            # Compute schur comp
            schur = _quad_prod(inv_Theta_11, theta_12)

            gy = torch.Tensor([[self.gy]]).type(torch.float64)
            theta_22_next = gy + schur

            # Update W
            Delta = torch.zeros_like(Theta)
            Delta[_22] = theta_22_next - theta_22
            Theta = Theta + Delta

            # update W
            w_22_next = 1.0 / gy
            w_12_next = einsum('b, bij, bj ->bi', -w_22_next.squeeze(-1),
                               inv_Theta_11, theta_12)
            self.W[_11] = (
                inv_Theta_11 + _a_outer(gy.squeeze(-1), w_12_next, w_12_next)).detach()
            self.W[_12] = w_12_next.detach()
            self.W[_21] = w_12_next.detach()
            self.W[_22] = w_22_next.detach()

        self.log['lambda_min_Theta'].append(
            torch.linalg.eigvalsh(Theta.squeeze())[0].item())
        self.log['schur'].append(schur.squeeze().item())
        norm_ = torch.linalg.norm(
            self.W @ Theta - torch.eye(self.p).type(torch.float64))
        self.log['norm_diff'].append(norm_)

        return Theta

    def forward(self, Theta):
        """ Forward pass. """

        self.W = torch.linalg.inv(Theta).detach()

        # W = S + self.diag_init * torch.eye(S.shape[-1]).expand_as(S).type_as(S)
        for k in range(0, self.K):
            Theta = self.one_pass(Theta)

        return Theta


# %%
test = 50
fs = 13
# for _ in range(test):
p = 100
A = torch.randn(p, p)
Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
Theta_true = torch.linalg.inv(Cov_true).type(torch.float64)

spodnet = SpodNet(K=70,
                  p=p,
                  gy=8.0)

Theta_new = spodnet.forward(Theta_true[None, :, :])

print(spodnet.W @ Theta_new)
print(torch.allclose(spodnet.W @ Theta_new,
                     torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3))
# %%
cmap = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
to_plot = [spodnet.log['schur'],
           spodnet.log['lambda_min_Theta'],
           spodnet.log['norm_diff']]
label = ['schur', '$\\lambda_{\\min}(\\Theta)$', '$\|\\Theta W -I\|$']
for i, (theplot, thelabel) in enumerate(zip(to_plot, label)):
    ax.plot(theplot, lw=2, label=thelabel, color=cmap(i),
            marker='o', markersize=4)
ax.hlines(spodnet.gy, xmin=0, xmax=len(spodnet.log['lambda_min_Theta']),
          linestyle='--', color='k', label='gy', lw=2)
ax.set_yscale('log')
ax.set_xlabel('iter')
ax.legend(fontsize=fs)
ax.grid(alpha=0.9)
# %%
