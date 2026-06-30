"""Test fixtures: deterministic linear PDEs with known closed-form solutions.

These validate the solver plumbing only -- they are NOT shipped examples.
"""

import torch

from spde_st.equations.base import Equation


class LinearHeat(Equation):
    """Deterministic linear PDE solved by the BSDE machinery.

    Forward diffusion dX = dW (b = 0, sigma = I), zero driver (f = 0), terminal
    g(x) = |x|^2. Then u(t, x) = E[g(X_T) | X_t = x] = |x|^2 + dim*(T - t),
    a closed form to benchmark the solver against. No randomness in the problem
    itself; the Brownian paths are only the Feynman-Kac tool.
    """

    def __init__(self, x0=1.0, T=1.0, dim=1):
        super().__init__(x_0=[x0] * dim, T=T, dim_x=dim, dim_y=1, dim_d=dim)

    def b(self, t, x):
        return torch.zeros_like(x)

    def sigma(self, t, x):
        bs = x.size(0)
        return torch.eye(self.dim_x).unsqueeze(0).expand(bs, self.dim_x, self.dim_d)

    def f(self, t, x, y, z):
        return torch.zeros(x.size(0), self.dim_y)

    def g(self, x):
        return (x**2).sum(dim=1, keepdim=True)

    def oracle(self, x, t):
        """Exact u(t, x) = |x|^2 + dim*(T - t)."""
        return (x**2).sum(dim=1, keepdim=True) + self.dim_x * (self.T - t)
