import torch
from torch import nn
from torch import einsum


def _a_outer(a, vec_1, vec_2):
    """batch version of ``a * outer(vec_1, vec_2)``."""
    return einsum("b,bi,bj->bij", a, vec_1, vec_2)


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)


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
        )

    def update_W_from_Theta(self, theta_22_next, theta_12_next,
                            inv_Theta_11, _11, _12, _21, _22):
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
            inv_Theta_11 = W_11 - _a_outer(1.0 / w_22, w_12, w_12)
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
            nn.Linear(self.p - 1, self.p - 1, bias=False, dtype=torch.float64),
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
            theta_12_next = self.col_learner(theta_12)

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
