# %% Stability test
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from spdsw.spodnet import _a_outer
from torch import einsum
# %%


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

    def __init__(self, p, init_inv, K=1, gy=1.0, lamb=1e-3):
        super().__init__()

        self.p = p
        self.W = init_inv.clone().detach()
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.K = K
        self.gy = gy
        self.lamb = lamb

    def one_pass(self, Theta):

        indices = torch.arange(self.p)

        for col in range(self.p):

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of Theta
            Theta_11 = Theta[_11]
            theta_22 = Theta[_22]
            if torch.isnan(theta_22).any():
                raise NanError('theta_22 has nan')

            theta_12 = Theta[_12]
            if torch.isnan(theta_12).any():
                raise NanError('theta_12 has nan')

            # Blocks of W
            W_11 = self.W[_11]
            if torch.isnan(W_11).any():
                raise NanError('W_11 has nan')

            w_22 = self.W[_22]
            if torch.isnan(w_22).any():
                raise NanError('w_22 has nan')

            w_12 = self.W[_12]
            if torch.isnan(w_12).any():
                raise NanError('w_12 has nan')

            # Inv block W_11
            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
            # inv_Theta_11 = W_11
            if torch.isnan(inv_Theta_11).any():
                print('inv_Theta_11 = {}'.format(inv_Theta_11))
                print(f'{theta_22=}')
                print(f'{theta_12=}')
                print(f'{W_11=}')
                print(f'{w_22=}')
                print(f'{w_12=}')
                print(f'{col=}')
                raise NanError('inv_Theta_11 has nan')

            # theta_12_next = torch.randn(
            #     self.p-1).type(torch.float64).unsqueeze(0)
            theta_12_next = torch.randn(self.p-1).type(torch.float64)[None, :]
            # theta_12_next = einsum(
            #    'bij, bj -> bi', Theta_11, direction)
            # print(theta_12_next)
            # theta_12_next = prox_l1(
            #    theta_12_next, alpha=self.lamb) / math.sqrt(self.p)

            if torch.isnan(theta_12_next).any():
                print('theta_12 = {}'.format(theta_12))
                print('theta_12_next = {}'.format(theta_12_next))
                raise NanError('theta_12_next has nan')

            # Compute schur comp
            # schur = _quad_prod(inv_Theta_11, theta_12_next - theta_12)
            schur = _quad_prod(W_11, theta_12_next)
            # schur = _quad_prod(Theta_11, direction)
            if torch.isnan(schur).any():
                raise NanError('schur has nan')

            gy = torch.Tensor([[self.gy]]).type(torch.float64)
            # gy = theta_22[None, :]
            # theta_22_next = self.lamb + gy + schur
            theta_22_next = gy + schur

            if torch.isnan(theta_22_next).any():
                raise NanError('theta_22_next has nan')
            # Update W
            Delta = torch.zeros_like(Theta)
            # Delta = self.lamb*torch.eye(self.p)[None, :, :].type(torch.float64)
            Delta[_22] = theta_22_next - theta_22
            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12
            Theta = Theta + Delta
            # self.log['lambda_min_Theta'].append(
            #     torch.linalg.eigvalsh(Theta.squeeze())[0].item())
            # self.log['schur'].append(schur.squeeze().item())

            # update W
            w_22_next = 1.0 / \
                (theta_22_next - _quad_prod(inv_Theta_11, theta_12_next))
           # print(gy[None, :].shape)
            if torch.isnan(w_22_next).any():
                raise NanError('w_22_next has nan')
            w_12_next = einsum('b, bij, bj ->bi', -w_22_next,
                               inv_Theta_11, theta_12_next)
            # w_12_next = einsum('b, bij, bj ->bi', -w_22_next.squeeze(-1),
            #                    Theta_11, direction)
            if torch.isnan(w_12_next).any():
                print('inv_Theta_11 = {}'.format(inv_Theta_11))
                print('w_22_next = {}'.format(w_22_next))
                print('theta_12_next = {}'.format(theta_12_next))
                print('theta_22_next = {}'.format(theta_22_next))
                print('kron = {}'.format(schur))
                print('gy = {}'.format(gy))
                print(f'{col=}')
                raise NanError('w_12_next has nan')
            self.W[_11] = (
                inv_Theta_11 + _a_outer(w_22_next, w_12_next, w_12_next)).detach()
            self.W[_12] = w_12_next.detach()
            self.W[_21] = w_12_next.detach()
            self.W[_22] = w_22_next.detach()

        # Theta = Theta + self.lamb * \
        #    torch.eye(self.p)[None, :, :].type(torch.float64)
        # self.W = torch.linalg.inv(Theta).detach()
        return Theta

    def forward(self, Theta):
        """ Forward pass. """

        self.W = torch.linalg.inv(Theta).detach()

        # W = S + self.diag_init * torch.eye(S.shape[-1]).expand_as(S).type_as(S)
        for k in range(0, self.K):
            Theta = self.one_pass(Theta)

        return Theta


test = 50
# for _ in range(test):
p = 5
A = torch.randn(p, p)
Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
Theta_true = torch.linalg.inv(Cov_true).type(torch.float64)

# %%
spodnet = SpodNet(K=1,
                  p=p,
                  init_inv=Cov_true[None, :, :],
                  gy=1,
                  lamb=0)

Theta_new = spodnet.forward(Theta_true[None, :, :])

print(spodnet.W @ Theta_new)
print(torch.allclose(spodnet.W @ Theta_new,
                     torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(spodnet.log['schur'], lw=2, label='schur')
ax.plot(spodnet.log['lambda_min_Theta'], lw=2, label='lambda_min')
ax.hlines(spodnet.gy, xmin=0, xmax=p*spodnet.K,
          linestyle='--', color='k', label='gy', lw=2)
ax.set_yscale('log')
ax.set_xlabel('iter')
ax.legend()
ax.grid()

# Theta_new

# %%
p = 10
A = torch.randn(p, p)
Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
torch.linalg.eigvalsh(Cov_true)
# %%
spodnet.log['lambda_min_Theta']
# %%
