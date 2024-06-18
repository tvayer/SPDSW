# %% Test cov and da
from spdsw.spodnet import SpodNet as SpodNet2
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import einsum
from spdsw.spodnet import _a_outer, _quad_prod
# , Exp
from torch import nn
import time
import math
# import torch.jit as jit

OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam,
              'LBFGS': torch.optim.LBFGS}


class NanError(Exception):
    pass


def return_indices(col, indices):
    indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
    _11 = slice(
        None), indices_minus_col[:, None], indices_minus_col[None]
    _12 = slice(None), indices_minus_col, col
    _21 = slice(None), col, indices_minus_col
    _22 = slice(None), col, col
    return _11, _12, _21, _22


class SpodNet(nn.Module):

    def __init__(self, p, K=1, two_steps=False, one_step_diff=False):
        super().__init__()

        self.p = p
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.K = K
        self.one_step_diff = one_step_diff
        self.two_steps = two_steps

        self.col_learner = nn.Sequential(
            nn.Linear(self.p - 1, self.p - 1, bias=False, dtype=torch.float64),
            # nn.Identity()
        )

    def update_W_from_Theta(self, theta_22_next, theta_12_next, inv_Theta_11, _11, _12, _21, _22):
        # update W
        w_22_next = 1.0 / \
            (theta_22_next - _quad_prod(inv_Theta_11, theta_12_next))

        w_12_next = einsum('b, bij, bj ->bi', -w_22_next,
                           inv_Theta_11, theta_12_next)

        self.W[_11] = (
            inv_Theta_11 + _a_outer(1.0 / w_22_next, w_12_next, w_12_next)).detach()
        self.W[_12] = w_12_next.detach()
        self.W[_21] = w_12_next.detach()
        self.W[_22] = w_22_next.detach()

    def one_pass(self, Theta):

        indices = torch.arange(self.p)

        for col in range(self.p):
            _11, _12, _21, _22 = return_indices(col, indices)

            # Blocks of Theta
            theta_22 = Theta[_22]
            theta_12 = Theta[_12]

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            # Inv block W_11
            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
            theta_12_next = self.col_learner(theta_12)

            # Compute schur comp
            schur = einsum("bi, bij, bj->b", theta_12_next,
                           inv_Theta_11, theta_12_next)
            gy = theta_22 - schur
            # - einsum("bi, bij, bj->b",
            #                       theta_12, inv_Theta_11, theta_12)
            # print(f'{gy=}')
            # print(f'{torch.min(torch.linalg.eigvalsh(Theta.squeeze())[0])=}')

            theta_22_next = gy + schur

            # print(f'{theta_22_next=}')

            # Update Theta
            Delta = torch.zeros_like(Theta)
            Delta[_22] = theta_22_next - theta_22
            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12
            Theta = Theta + Delta

            self.update_W_from_Theta(
                theta_22_next, theta_12_next, inv_Theta_11, _11, _12, _21, _22)

        return Theta

    def forward(self, Theta):
        """ Forward pass. """
        # if self.W is None:
        self.W = torch.linalg.inv(Theta).detach()
        if self.one_step_diff:
            with torch.no_grad():
                for k in range(0, self.K):
                    Theta = self.one_pass(Theta)
            Theta = self.one_pass(Theta)
        else:
            for k in range(0, self.K):
                Theta = self.one_pass(Theta)
        return Theta


def generate_ds(n, p):
    D = torch.zeros((n, p, p)).type(torch.float64)
    for i in range(n):
        A = torch.randn(p, p).type(torch.float64)
        Theta = A @ A.T + torch.eye(p).type(torch.float64)
        D[i] = Theta
    return D


# %%
n = 512
p = 22
Theta_true = generate_ds(n, p)
# Theta_true = torch.load('test_tensor.pt')[None, :, :]
p = 22
# %%
spodnet = SpodNet(K=1, p=p)

Theta_new = spodnet.forward(Theta_true)
# print(f'{spodnet.W @ Theta_new}=')
print(spodnet.W @ Theta_new)
# print(torch.allclose(spodnet.W @ Theta_new,
#                      torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3))
# print(f'{Theta_new - Theta_true}')


# %%
argmin_eigh = np.argmin(np.array(
    [min(torch.linalg.eigvalsh(Theta_new[l].detach())) for l in range(n)]))
min_eigh = min(torch.linalg.eigvalsh(Theta_new[argmin_eigh].detach()))
print(f'{min_eigh}')
print(f'{argmin_eigh}')

# %%
# print(f'{torch.linalg.eigvalsh(Theta_new[l])}')
# print(f'{torch.linalg.eigvalsh(spodnet.W[l])}')


# %%
# %%
Theta_init = Theta_true

print('Theta_init shape = {}'.format(Theta_init.shape))
print('Theta_true eigval = {}'.format(torch.linalg.eigvalsh(Theta_true)))


K = 1
spodnet = SpodNet(K=K,
                  p=p)
pytorch_total_params = sum(p.numel() for p in spodnet.parameters())
print('SpodNet has {} parameters'.format(pytorch_total_params))


def loss(x):
    return ((Theta_true - x)**2).mean()


loss_init = loss(Theta_init).item()
print('Distance of Theta_init to Theta_true = {0:.5f}'.format(loss_init))

losses = []
times_forward = []
times_backward = []

trainloader = torch.utils.data.DataLoader(
    Theta_init,
    batch_size=n,
    shuffle=True)

nb_epochs = 5000
lr = 1e-2
thresh = 1e-5
optimizer_name = 'Adam'
optimizer = OPTIMIZERS[optimizer_name](spodnet.parameters(), lr=lr)
tot = 0
for epoch in range(nb_epochs):
    t0 = time.time()
    for _, data in enumerate(trainloader):
        # print(i)
        optimizer.zero_grad()
        st = time.time()
        Theta_new = spodnet(Theta_init)
        ed = time.time()
        times_forward.append(ed - st)
        output = loss(Theta_new)
        # +  reg * torch.linalg.norm(spodnet.col_learner[0].weight)**2
        st = time.time()
        output.backward()
        ed = time.time()
        times_backward.append(ed - st)
        optimizer.step()
        losses.append(output.item())
        if torch.isnan(output):
            print('Nan during iterations')
            break
    t1 = time.time()
    tot += t1 - t0
    if epoch % 50 == 0:
        print(
            '--- epoch = {0}, loss = {1:.7f}, time = {2:.5f} ---'.format(epoch, output.item(), tot))
        tot = 0
    if len(losses) > 2:
        if losses[-2] < losses[-1] and abs(losses[-1] - losses[-2]) <= thresh:
            print('Break iterations')
            break
iter_attained = epoch
# %%
fs = 15
plt.plot(losses, lw=2, label='loss')
# plt.hlines(loss_init, xmin=0, xmax=iter_attained, label='SE between Theta_init and Theta_true',
#            lw=2, color='grey', linestyle='--')
plt.ylabel('Mean squared error', fontsize=fs)
plt.xlabel('iter', fontsize=fs)
plt.title('Loss during iterations', fontsize=fs+2)
plt.yscale('log')
plt.legend()
plt.grid()

# %%
# from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

k = 0
true_precision = Theta_true.numpy()[k]
Theta_esti = spodnet(Theta_init)[k].detach().numpy()


cmap = plt.cm.bwr

vmax = max([np.max(true_precision), np.max(Theta_esti)])/2
vmin = min([np.min(true_precision), np.min(Theta_esti)])/2
norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

ax[0].imshow(true_precision, cmap=cmap, norm=norm)
ax[0].set_title('True precision', fontsize=fs)
ax[0].set_xticks([])
ax[0].set_yticks([])

im = ax[1].imshow(Theta_esti, cmap=cmap, norm=norm)
ax[1].set_title('Esti precision', fontsize=fs)
ax[1].set_xticks([])
ax[1].set_yticks([])

cbar = fig.colorbar(im, ax=ax[1])
cbar.ax.tick_params(labelsize=10)
cbar.ax.locator_params(nbins=5)

# %%

weights_linear = spodnet.col_learner[0].weight.detach().numpy()
# bias = spodnet.col_learner[0].bias.detach().numpy()
# print(f'{bias=}')
vmax = max(weights_linear.ravel())
vmin = min(weights_linear.ravel())
norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
cmap = plt.cm.bwr
fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
im = ax.imshow(weights_linear, cmap=cmap, norm=norm)
ax.set_title('NN weights')
cbar = fig.colorbar(im, ax=ax)
# %%
fs = 15
plt.loglog(times_forward, lw=2, label='time forward')
plt.loglog(times_backward, lw=2, label='time backward')
plt.ylabel('time (in sec.)', fontsize=fs)
plt.xlabel('iter', fontsize=fs)
plt.legend()
# plt.grid()
# # %%
# to_plot = 'mean'
# cmap = plt.cm.get_cmap('tab10')
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# loc = 0
# for k, v in spodnet.updater.log.items():
#     print(k)
#     # time_ = np.mean(v)
#     if to_plot == 'sum':
#         time_ = np.sum(v)
#     elif to_plot == 'mean':
#         time_ = np.mean(v)
#     # std = np.std(v)
#     ax.bar(loc, time_,
#            # yerr=std,
#            label=k, color=cmap(loc), alpha=0.9, ecolor='black', capsize=10)
#     loc += 1
# ax.legend()
# ax.grid()
# d = {'sum': 'total', 'mean': 'average'}
# ax.set_ylabel('{} time (in sec)'.format(d[to_plot]))
# plt.tight_layout()
# spodnet.updater.log['invW11']
# A = torch.randn(p, p)
# B = (A @ A.T + torch.eye(p)).type(torch.float64)
# W_init = B[None, :, :]
# W_esti = spodnet(W_init)[0].detach().numpy()

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(W_esti)
# ax[0].set_title('W_esti. err = {0:5f}'.format(((B - W_esti)**2).sum()))
# ax[1].imshow(B)
# ax[1].set_title('W_true')
# plt.tight_layout()


# p = 10
# A = torch.randn(p, p)
# B = A @ A.T + torch.eye(p)
# Binv = torch.linalg.inv(B)

# # %%
# Theta = Binv[None, :, :].type(torch.float64)
# W = B[None, :, :].type(torch.float64)
# # %%
# updater = Update_W(p=p, device=torch.device('cpu'))
# # %%
# W_new, Theta_new = updater.forward(W, Theta)
# # %%
# torch.allclose(W_new @ Theta_new,
#                torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3)

# %%
# class Update_Theta(nn.Module):
#     """ Layer class : Updates every column/row/diagonal element of the input matrix. """

#     def __init__(self, p, init_inv, device=None, thresh=1e-5, lamb=10, zeta=1.0):
#         super().__init__()

#         self.p = p
#         self.device = device
#         self.log = {}
#         self.log['inv_Theta_11'] = []
#         self.log['compute_theta_12'] = []
#         self.log['update_Theta'] = []
#         # self.log['update_Theta'] = []
#         self.log['compute_kron'] = []
#         self.log['compute_gy'] = []
#         self.log['update_W_11'] = []
#         self.log['update_w_12'] = []

#         self.alpha_learner = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.p+(self.p-1)**2, self.p, dtype=torch.float64),
#             nn.ReLU(),
#             nn.Linear(self.p, self.p, dtype=torch.float64),
#             nn.ReLU(),
#             nn.Linear(self.p, 1, dtype=torch.float64),
#             CustomSigmoid(thresh=thresh, lamb=lamb)
#         )

#         self.col_learner = nn.Sequential(
#             # nn.Flatten(),
#             nn.Linear(self.p - 1, self.p - 1, dtype=torch.float64),
#             # nn.Identity()
#             nn.ReLU(),
#             # nn.Linear(self.p, self.p, dtype=torch.float64), nn.Tanh(),
#             # nn.Linear(self.p-1, self.p - 1, dtype=torch.float64)
#         )
#         self.W = init_inv.clone().detach()
#         self.zeta = zeta

#     def forward(self, Theta):
#         """ A single layer update: update column/row/diagonal of all indices. """

#         indices = torch.arange(self.p)

#         for col in range(self.p):

#             indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
#             _11 = slice(
#                 None), indices_minus_col[:, None], indices_minus_col[None]
#             _12 = slice(None), indices_minus_col, col
#             _21 = slice(None), col, indices_minus_col
#             _22 = slice(None), col, col

#             # Blocks of Theta
#             # Theta_11 = Theta[_11]
#             theta_22 = Theta[_22]
#             if torch.isnan(theta_22).any():
#                 raise NanError('theta_22 has nan')

#             theta_12 = Theta[_12]
#             if torch.isnan(theta_12).any():
#                 raise NanError('theta_12 has nan')

#             # Blocks of W
#             W_11 = self.W[_11]
#             if torch.isnan(W_11).any():
#                 raise NanError('W_11 has nan')

#             w_22 = self.W[_22]
#             if torch.isnan(w_22).any():
#                 raise NanError('w_22 has nan')

#             w_12 = self.W[_12]
#             if torch.isnan(w_12).any():
#                 raise NanError('w_12 has nan')

#             # Inv block W_11
#             st = time.time()
#             inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
#             ed = time.time()
#             self.log['inv_Theta_11'].append(ed-st)
#             if torch.isnan(inv_Theta_11).any():
#                 print('inv_Theta_11 = {}'.format(inv_Theta_11))
#                 print(f'{theta_22=}')
#                 print(f'{theta_12=}')
#                 print(f'{W_11=}')
#                 print(f'{w_22=}')
#                 print(f'{w_12=}')
#                 print(f'{col=}')
#                 raise NanError('inv_Theta_11 has nan')

#             # Compute theta_12_next
#             st = time.time()
#             # direction = self.col_learner(theta_12)
#             # in the span of Theta_11
#             # theta_12_next = einsum('bij, bj -> bi', Theta_11, direction)
#             theta_12_next = self.col_learner(theta_12)
#             # Normalize
#             schur = _quad_prod(inv_Theta_11, theta_12_next)
#             theta_12_next = (torch.sqrt(self.zeta / schur))*theta_12_next
#             ed = time.time()
#             self.log['compute_theta_12'].append(ed-st)
#             if torch.isnan(theta_12_next).any():
#                 print('theta_12 = {}'.format(theta_12))
#                 print('theta_12_next = {}'.format(theta_12_next))
#                 print(f'{schur=}')
#                 raise NanError('theta_12_next has nan')

#             # Compute schur comp
#             st = time.time()
#             # schur = _quad_prod(inv_Theta_11, theta_12_next)
#             schur = torch.Tensor([[self.zeta]]).type(torch.float64)
#             # kron = _quad_prod(Theta_11, direction)
#             ed = time.time()
#             self.log['compute_kron'].append(ed-st)
#             if torch.isnan(schur).any():
#                 raise NanError('kron has nan')

#             # alpha_learner_features = W
#             # print(f'{theta_12.shape=}')
#             # print(f'{theta_22[None, :].shape=}')
#             # print(f'{W_11.ravel()[None, :].shape=}')
#             # alpha_learner_features = theta_12
#             alpha_learner_features = torch.cat(
#                 (theta_22[None, :], theta_12, W_11.ravel()[None, :]), -1)
#             # print(f'{alpha_learner_features.shape=}')
#             # alpha_learner_features = W

#             st = time.time()
#             gy = self.alpha_learner(
#                 alpha_learner_features)  # of size (batch, 1)
#             # gy = (1.0 / (self.S[_22].unsqueeze(-1) + 1e-3))
#             # gy = torch.Tensor([[1.0]]).type(torch.float64)
#             # print(gy.shape)
#             ed = time.time()
#             self.log['compute_gy'].append(ed-st)
#             theta_22_next = gy + schur

#             if torch.isnan(theta_22_next).any():
#                 raise NanError('theta_22_next has nan')
#             # Update W
#             st = time.time()
#             Delta = torch.zeros_like(Theta)
#             Delta[_22] = theta_22_next - theta_22
#             Delta[_12] = theta_12_next - theta_12
#             Delta[_21] = theta_12_next - theta_12
#             Theta = Theta + Delta
#             ed = time.time()
#             self.log['update_Theta'].append(ed-st)

#             # Update W
#             # print(f'{kron=}')
#             # print(f'{gy[None]=}')
#             w_22_next = 1.0 / gy
#             if torch.isnan(w_22_next).any():
#                 raise NanError('w_22_next has nan')
#             # print('w_22_next = {}'.format(w_22_next.item()))
#             st = time.time()
#             w_12_next = einsum('b, bij, bj ->bi', -w_22_next.squeeze(-1),
#                                inv_Theta_11, theta_12_next)
#             ed = time.time()
#             self.log['update_w_12'].append(ed - st)
#             if torch.isnan(w_12_next).any():
#                 print('inv_Theta_11 = {}'.format(inv_Theta_11))
#                 print('w_22_next = {}'.format(w_22_next))
#                 print('theta_12_next = {}'.format(theta_12_next))
#                 print('theta_22_next = {}'.format(theta_22_next))
#                 print('kron = {}'.format(schur))
#                 print('gy = {}'.format(gy))
#                 print(f'{col=}')
#                 # print(f'{gy=}')
#                 raise NanError('w_12_next has nan')
#             st = time.time()
#             self.W[_11] = (
#                 inv_Theta_11 + _a_outer(gy.squeeze(-1), w_12_next, w_12_next)).detach()
#             ed = time.time()
#             self.log['update_W_11'].append(ed-st)
#             self.W[_12] = w_12_next.detach()
#             self.W[_21] = w_12_next.detach()
#             self.W[_22] = w_22_next.detach()

#         return Theta
# test = 50
# for _ in range(test):
#     p = 50
#     A = torch.randn(p, p)
#     Cov_true = (A @ A.T + torch.eye(p)).type(torch.float64)
#     Theta_true = torch.linalg.inv(Cov_true).type(torch.float64)

#     spodnet = SpodNet(K=1,
#                       p=p,
#                       device=torch.device('cpu'),
#                       init_inv=Cov_true[None, :, :],
#                       thresh=1e-3,
#                       lamb=1e3)

#     Theta_new = spodnet.forward(Theta_true[None, :, :])

#     # print(spodnet.updater.W @ Theta_new)
#     print(torch.allclose(spodnet.updater.W @ Theta_new,
#                          torch.eye(p).type(torch.float64), atol=1e-3, rtol=1e-3))

# else:

#     for col in range(self.p):
#         _11, _12, _21, _22 = return_indices(col, indices)
#         theta_12 = Theta[_12]
#         theta_12_next = self.col_learner(theta_12)
#         Delta = torch.zeros_like(Theta)
#         Delta[_12] = theta_12_next - theta_12
#         Delta[_21] = theta_12_next - theta_12
#         Theta = Theta + Delta

#     for col in range(self.p):
#         _11, _12, _21, _22 = return_indices(col, indices)
#         # Blocks of Theta
#         theta_12 = Theta[_12]
#         theta_22 = Theta[_22]

#         # Blocks of W
#         W_11 = self.W[_11]
#         w_22 = self.W[_22]
#         w_12 = self.W[_12]

#         inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
#         # Compute schur comp
#         schur = _quad_prod(inv_Theta_11, theta_12)

#         gy = theta_22 - einsum("bi, bij, bj->b",
#                                theta_12, W_11, theta_12)
#         theta_22_next = gy + schur

#         # Update W
#         Delta = torch.zeros_like(Theta)
#         Delta[_22] = theta_22_next - theta_22
#         Theta = Theta + Delta

#         self.update_W_from_Theta(
#             theta_22_next, theta_12, inv_Theta_11, _11, _12, _21, _22)
