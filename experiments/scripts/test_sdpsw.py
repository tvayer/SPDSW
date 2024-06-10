# Test with Spdsw
# %%
import warnings
import torch
import argparse
import time
import ot
import geoopt
import numpy as np
import pandas as pd
import itertools
import os
from torch import nn
from torch import einsum

from pathlib import Path
from joblib import Memory
from tqdm import trange

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from utils.download_bci import download_bci
from utils.get_data import get_data, get_cov, get_cov2
from utils.models import Transformations, FeaturesKernel, get_svc
# from spdsw.spodnet import TwoStepSpodNet


def _a_outer(a, vec_1, vec_2):
    """batch version of ``a * outer(vec_1, vec_2)``."""
    return einsum("b,bi,bj->bij", a, vec_1, vec_2)


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)
# %%


N_JOBS = 1
SEED = 2022
NTRY = 1
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
RESULTS = os.path.join(EXPERIMENTS, "results/da.csv")
DEVICE = "cpu"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_da/"),
    verbose=0
)

# Set to True to download the data in experiments/data_bci
DOWNLOAD = False

if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)

# %%
# hyperparams = {
#     "distance": ["les", "lew", "spdsw", "logsw"],
#     "n_proj": [500],
#     "n_epochs": [500],
#     "seed": RNG.choice(10000, NTRY, replace=False),
#     "subject": [1, 3, 7, 8, 9],
#     "target_subject": [1, 3, 7, 8, 9],
#     #         "cross_subject": [False],
#     "multifreq": [False],
#     "reg": [10.],
# }

hyperparams = {
    "distance": ["spdsw"],
    "n_proj": [500],
    "n_epochs": [500],
    "seed": RNG.choice(10000, NTRY, replace=False),
    "subject": [1],
    "target_subject": [3],
    #         "cross_subject": [False],
    "multifreq": [False],
    "reg": [10.],
}

task = "subject"
if task == "session":
    hyperparams["cross_subject"] = [False]
    RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_session.csv")
elif task == "subject":
    hyperparams["cross_subject"] = [True]
    RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_subject.csv")

keys, values = zip(*hyperparams.items())
permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

dico_results = {
    "align": [],
    "no_align": [],
    "time": []
}
# %% Test if one param
params = permuts_params[0]
print(params)
# %%
distance = params["distance"]
n_proj = params["n_proj"]
n_epochs = params["n_epochs"]
seed = params["seed"]
subject = params["subject"]
multifreq = params["multifreq"]

cross_subject = params["cross_subject"]
target_subject = params["target_subject"]
reg = params["reg"]

if multifreq:
    get_cov_function = get_cov2
else:
    get_cov_function = get_cov

if cross_subject:
    if target_subject == subject:
        s_noalign, s_align, runtime = 1., 1., 0

    Xs, ys = get_data(subject, True, PATH_DATA)
    cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
    ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

    Xt, yt = get_data(target_subject, True, PATH_DATA)
    cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
    yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

else:

    Xs, ys = get_data(subject, True, PATH_DATA)
    cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
    ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

    Xt, yt = get_data(subject, False, PATH_DATA)
    cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
    yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1
# %%


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
            # nn.Flatten(),
            nn.Linear(self.p-1, self.p - 1, dtype=torch.float64),
            # nn.ReLU(),
            # nn.Linear(10, self.p - 1, dtype=torch.float64), nn.ReLU(),
            # nn.Linear(10, self.p - 1, dtype=torch.float64)
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
            gy = theta_22 - einsum("bi, bij, bj->b",
                                   theta_12, inv_Theta_11, theta_12)

            theta_22_next = gy + schur

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


class TwoStepSpodNet(nn.Module):

    def __init__(self, p, K=1):
        super().__init__()

        self.p = p
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.log['norm_diff'] = []
        self.K = K

        self.col_learner = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p**2, self.p - 1, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(self.p - 1, self.p - 1, dtype=torch.float64)
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
        for col in range(self.p):  # update col

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of Theta
            theta_12 = Theta[_12]
            theta_12_next = self.col_learner(Theta)

            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = theta_12_next - theta_12

        Theta = Theta + Delta

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
            diff_12 = Delta[_12]

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
            # Compute schur comp
            schur = _quad_prod(inv_Theta_11, diff_12)

            gy = theta_22
            theta_22_next = gy + schur

            # Update W
            Delta_diag = torch.zeros_like(Theta)
            Delta_diag[_22] = theta_22_next - theta_22
            Theta = Theta + Delta_diag
            # update W
            self.update_W_from_Theta(
                theta_22_next, theta_12, inv_Theta_11, _11, _12, _21, _22)

        return Theta

    def forward(self, Theta):
        """ Forward pass. """

        self.W = torch.linalg.inv(Theta).detach()

        # W = S + self.diag_init * torch.eye(S.shape[-1]).expand_as(S).type_as(S)
        for k in range(0, self.K):
            Theta = self.one_pass(Theta)

        return Theta


# %%
# cov_Xs.shape
# p = Xs.shape[1]
# model = TwoStepSpodNet(p=p, K=1)

# zs = model(cov_Xs.squeeze())


d = 22
n_freq = cov_Xs.shape[2]

n_samples_s = len(cov_Xs)
n_samples_t = len(cov_Xt)

model = TwoStepSpodNet(p=d, K=1)

# model = Transformations(d, n_freq, DEVICE, DTYPE, seed=seed)

start = time.time()

if distance in ["lew", "les"]:
    lr = 1e-2
    a = torch.ones((n_samples_s,), device=DEVICE, dtype=DTYPE) / n_samples_s
    b = torch.ones((n_samples_t,), device=DEVICE, dtype=DTYPE) / n_samples_t
    manifold = geoopt.SymmetricPositiveDefinite("LEM")

elif distance in ["spdsw", "logsw", "sw"]:
    if cross_subject:
        lr = 5e-1
    else:
        lr = 1e-1

    spdsw = SPDSW(
        d,
        n_proj,
        device=DEVICE,
        dtype=DTYPE,
        random_state=seed,
        sampling=distance
    )

# optimizer = RiemannianSGD(model.parameters(), lr=lr)

OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam,
              'LBFGS': torch.optim.LBFGS}
optimizer_name = 'Adam'
lr = 1e-5
optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr=lr)


# %%
pbar = trange(n_epochs)

for e in pbar:
    zs = model(cov_Xs.squeeze()).reshape(cov_Xs.shape)
    # zs = model(cov_Xs)
    # print(zs.reshape(cov_Xs.shape).shape)
    loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
    for f in range(n_freq):
        if distance == "lew":
            M = manifold.dist(zs[:, 0, f][:, None], cov_Xt[:, 0, f][None]) ** 2
            loss += 0.1 * ot.emd2(a, b, M)

        elif distance == "les":
            M = manifold.dist(zs[:, 0, f][:, None], cov_Xt[:, 0, f][None]) ** 2
            loss += 0.1 * ot.sinkhorn2(a, b, M, reg)

        elif distance in ["spdsw", "logsw", "sw"]:
            loss += spdsw.spdsw(zs[:, 0, f], cov_Xt[:, 0, f], p=2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    pbar.set_postfix_str(f"loss = {loss.item():.3f}")

stop = time.time()
# %%
s_noalign = get_svc(cov_Xs[:, 0], cov_Xt[:, 0], ys,
                    yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)
# %%
s_align = get_svc(
    model(cov_Xs.squeeze()).reshape(cov_Xs.shape)[:, 0],  ys,
    # model(cov_Xs)[:, 0], cov_Xt[:, 0], ys,
    yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)

# %%
print(f'{s_noalign}, {s_align}')
# %%
