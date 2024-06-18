# %%
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

# %%


class NanError(Exception):
    pass


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp = torch.exp

    def forward(self, x):
        x = self.exp(x)
        return x


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
            nn.Linear(self.p+(self.p-1)**2, self.p - 1,
                      bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(self.p - 1, self.p - 1, bias=True, dtype=torch.float64),
        )

        self.diag_learner = nn.Sequential(
            nn.Linear(self.p+(self.p-1)**2, self.p - 1, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(self.p - 1, 1, dtype=torch.float64),
            Exp()
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

        Delta = torch.zeros_like(Theta)

        for col in range(self.p):
            _11, _12, _21, _22 = return_indices(col, indices)

            # Blocks of Theta
            Theta_11 = Theta[_11]
            theta_22 = Theta[_22]
            theta_12 = Theta[_12]

            features = torch.cat(
                (Theta_11.flatten(1, 2), theta_12, theta_22[:, None]), -1)

            # Inv block W_11
            theta_12_next = self.col_learner(features)

            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12

        Theta = Theta + Delta

        for col in range(self.p):
            _11, _12, _21, _22 = return_indices(col, indices)

            Theta_11 = Theta[_11]
            theta_12 = Theta[_12]
            theta_22 = Theta[_22]
            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            features = torch.cat(
                (Theta_11.flatten(1, 2), theta_12, theta_22[:, None]), -1)

            # Compute schur comp
            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)

            schur = einsum("bi, bij, bj->b", theta_12,
                           inv_Theta_11, theta_12)
            gy = self.diag_learner(features).squeeze()
            theta_22_next = gy + schur

            # Update Theta
            Delta_diag = torch.zeros_like(Theta)
            Delta_diag[_22] = theta_22_next - theta_22

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
    W = torch.diag(torch.randn(p)).type(torch.float64)
    D = torch.zeros((n, p, p)).type(torch.float64)
    D_transfo = torch.zeros((n, p, p)).type(torch.float64)
    for i in range(n):
        A = torch.randn(p, p).type(torch.float64)
        Theta = A @ A.T + torch.eye(p).type(torch.float64)
        D[i] = Theta
        D_transfo[i] = W @ Theta @ W
    return D, D_transfo


# %%
n = 256
p = 5
Theta_true, Theta_transfo = generate_ds(n, p)
# %%
K = 1
spodnet = SpodNet(K=K, p=p)
Theta_init = Theta_true
# Theta_new = spodnet.forward(Theta_true)

print('Theta_init shape = {}'.format(Theta_init.shape))
pytorch_total_params = sum(p.numel() for p in spodnet.parameters())
print('SpodNet has {} parameters'.format(pytorch_total_params))


def loss(x):
    return ((Theta_transfo - x)**2).mean()


loss_init = loss(Theta_init).item()
print('Distance of Theta_init to Theta_true = {0:.5f}'.format(loss_init))

losses = []
times_forward = []
times_backward = []
batch_size = 128
trainloader = torch.utils.data.DataLoader(
    Theta_init,
    batch_size=batch_size,
    shuffle=True)

nb_epochs = 4000
lr = 1e-2
thresh = 1e-6
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
        st = time.time()
        output.backward()
        ed = time.time()
        times_backward.append(ed - st)
        optimizer.step()
        losses.append(output.item())
        if torch.isnan(output):
            raise NanError('Nan during iterations')
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
k = 1
true_precision = Theta_transfo.numpy()[k]
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
