# Spodnet for covariances
import torch
from torch import nn
from torch import einsum
import time


def _a_outer(a, vec_1, vec_2):
    """batch version of ``a * outer(vec_1, vec_2)``."""
    return einsum("b,bi,bj->bij", a, vec_1, vec_2)


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp = torch.exp

    def forward(self, x):
        x = self.exp(x)
        return x


class TwoStepSpodNet(nn.Module):

    def __init__(self, p, K=1):
        super().__init__()

        self.p = p
        self.log = {}
        self.log['lambda_min_Theta'] = []
        self.log['schur'] = []
        self.log['norm_diff'] = []
        self.K = K

        self.alpha_learner = nn.Sequential(
            nn.Linear(2, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 1, dtype=torch.float64), Exp()
        )

        self.col_learner = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p**2, 10, dtype=torch.float64), nn.ReLU(),
            # nn.Linear(10, self.p - 1, dtype=torch.float64), nn.ReLU(),
            nn.Linear(10, self.p - 1, dtype=torch.float64)
        )

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
            # theta_12_next = torch.randn(
            #     self.p-1).type(torch.float64).unsqueeze(0)
            theta_12_next = self.col_learner(Theta)
            Delta = torch.zeros_like(Theta)
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

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            inv_Theta_11 = W_11 - _a_outer(1.0/w_22, w_12, w_12)
            # Compute schur comp
            schur = _quad_prod(inv_Theta_11, theta_12)

            alpha_learner_features = torch.cat(
                (w_22[:, None], theta_22[:, None]), -1)

            gy = self.alpha_learner(
                alpha_learner_features).squeeze()  # of size (batch, 1)
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

        # self.log['lambda_min_Theta'].append(
        #     torch.linalg.eigvalsh(Theta.squeeze())[0].item())
        # self.log['schur'].append(schur.squeeze().item())
        # norm_ = torch.linalg.norm(
        #     self.W @ Theta - torch.eye(self.p).type(torch.float64))
        # self.log['norm_diff'].append(norm_)

        return Theta

    def forward(self, Theta):
        """ Forward pass. """

        self.W = torch.linalg.inv(Theta).detach()

        # W = S + self.diag_init * torch.eye(S.shape[-1]).expand_as(S).type_as(S)
        for k in range(0, self.K):
            Theta = self.one_pass(Theta)

        return Theta


class SpodNet(nn.Module):
    """ Main class : unrolled algorithm. """

    def __init__(self, K, p, device, diag_init=0.1):
        super().__init__()

        self.K = K
        self.p = p
        self.device = device
        self.diag_init = diag_init
        self.forward_stack = Update_W(self.p, device=device)

    def forward(self, S):
        """ Forward pass. """

        W = S + self.diag_init * torch.eye(S.shape[-1]).expand_as(S).type_as(S)

        Theta = torch.linalg.pinv(W, hermitian=True)

        for k in range(0, self.K):

            W, Theta = self.forward_stack(W, Theta)

        return W  # , Theta_list


class Update_W(nn.Module):
    """ Layer class : Updates every column/row/diagonal element of the input matrix. """

    def __init__(self, p, device=None):
        super().__init__()

        self.p = p
        self.device = device

        self.alpha_learner = nn.Sequential(  # could be anything
            nn.Linear(1, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 1, dtype=torch.float64), Exp()
        )

        self.col_learner = nn.Sequential(
            nn.Linear(self.p - 1, self.p, dtype=torch.float64), nn.ReLU(),
            nn.Linear(self.p, self.p, dtype=torch.float64), nn.ReLU(),
            nn.Linear(self.p, self.p - 1, dtype=torch.float64)
        )

    def forward(self, W, Theta):
        """ A single layer update: update column/row/diagonal of all indices. """

        indices = torch.arange(self.p)

        for col in range(self.p):

            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of W
            # W_11 = W[_11]
            w_22 = W[_22]
            w_12 = W[_12]

            # Blocks of Theta
            Theta_11 = Theta[_11]
            theta_22 = Theta[_22]
            theta_12 = Theta[_12]

            # Compute w_12_next and mask j-th entry
            w_12_next = self.col_learner(w_12)

            # Diagonal element update
            alpha_learner_features = w_22  # only take w_22 for simplicity
            inv_W_11 = Theta_11 - _a_outer(1/theta_22, theta_12, theta_12)
            zeta = _quad_prod(inv_W_11, w_12_next)
            w_22_next = self.alpha_learner(
                alpha_learner_features).squeeze() + zeta

            # Update W
            Delta = torch.zeros_like(W)
            Delta[_22] = w_22_next - w_22
            Delta[_12] = w_12_next - w_12
            Delta[_21] = w_12_next - w_12
            W = W + Delta

            # Update Theta
            theta_22_next = 1.0 / (w_22_next - zeta)

            theta_12_next = einsum('b, bij, bj ->bi', -theta_22_next,
                                   inv_W_11, w_12_next)
            Theta[_11] = inv_W_11 + \
                _a_outer(1/theta_22_next, theta_12_next, theta_12_next)
            Theta[_12] = theta_12_next
            Theta[_21] = theta_12_next
            Theta[_22] = theta_22_next

        return W, Theta
