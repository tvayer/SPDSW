# Test spodnet clean
import torch
from spdsw.clean_spodnet import SpodNet, TwoStepSpodNet


def generate_ds(n, p, reg=1e-3):
    D = torch.zeros((n, p, p)).type(torch.float64)
    for i in range(n):
        A = torch.randn(p, p).type(torch.float64)
        Theta = A @ A.T + reg*torch.eye(p).type(torch.float64)
        D[i] = Theta
    return D
