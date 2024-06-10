# %%
import random
import torch
import matplotlib.pyplot as plt

# %%


def create_inverse_sdp(Theta, W0=None, gy_min=1e-5, gy_max=5):
    # A partir d'une matrice symmetrique forme deux matrices symmetriques
    # inverses l'une de l'autre dont la premiere a les memes elements hors diagonaux que la matrice d'origine
    # si W0 est PD les deux le sont

    p = Theta.shape[0]
    indices = torch.arange(p)
    log = {}
    log['lambda_min_Theta'] = []
    log['schur'] = []
    log['norm_diff'] = []
    log['inv_W_W'] = []
    log['lambda_min_W'] = []
    if W0 is None:
        W = torch.eye(p).type(torch.float64)
    else:
        W = W0
    Theta_new = Theta.clone()
    list_col = list(range(p))
    random.shuffle(list_col)
    gy = (gy_min - gy_max) * torch.rand(p) + gy_max
    log['lambda_min_W'].append(torch.linalg.eigvalsh(W)[0])
    for col in list_col:  # update diag
        # print(col)
        gy_ = gy[col].item()
        # print(gy_)
        indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
        _11 = indices_minus_col[:, None], indices_minus_col[None]
        _12 = indices_minus_col, col
        _21 = col, indices_minus_col
        _22 = col, col

        # Blocks of W
        W_11 = W[_11]
        w_22 = W[_22]
        w_12 = W[_12]

        A = W_11 - (1.0/w_22) * torch.outer(w_12, w_12)
        # Compute schur comp
        theta_22_next = gy_ + Theta[_12].T @ A @ Theta[_12]

        Theta_new[_22] = theta_22_next

        # update W
        W_new = W.clone()
        w_12_next = - (1.0 / gy_) * A @ Theta[_12]
        W_new[_11] = A + gy_ * torch.outer(w_12_next, w_12_next)
        W_new[_12] = w_12_next
        W_new[_21] = w_12_next
        W_new[_22] = 1.0 / gy_
        log['lambda_min_W'].append(torch.linalg.eigvalsh(W_new)[0])
        # log['inv_W_W'].append(torch.linalg.pinv(W) @ W_new)
        # print(W @ Theta_new)
    return Theta_new, W, log


# %%
p = 100
A = torch.randn(p, 2)
Theta = (A @ A.T).type(torch.float64)
torch.linalg.eigvalsh(Theta)
# %%
A = torch.randn(p, p).type(torch.float64)
B0 = torch.randn(p, p)
# W0 = (B0 @ B0.T - 2*torch.eye(p)).type(torch.float64)
W0 = (B0 @ B0.T + 2*torch.eye(p)).type(torch.float64)
print(torch.linalg.eigvalsh(W0))
Theta_new, W, log = create_inverse_sdp(Theta, W0=W0)
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(log['lambda_min_W'])
ax.set_yscale('log')
# %%
# for inv_ in log['inv_W_W']:
#    plt.imshow(inv_)
#    plt.show()
# %%
torch.linalg.eigvalsh(Theta_new)
# %%
torch.linalg.eigvalsh(W)

# %%
plt.imshow(Theta_new @ W)
plt.colorbar()
# %%
d = list(range(p))
random.shuffle(d)
d
# %%
