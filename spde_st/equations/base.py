"""Abstract forward-backward SDE / parabolic PDE data.

A concrete Equation defines the forward diffusion (b, sigma) and the backward
data (driver f, terminal g). The deep backward dynamic programming solver
(see spde_st.bsde.solver) consumes this interface and is otherwise
equation-agnostic.

Discrete backward relation used by the solver (Euler):
    Y_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) * dt + Z_n . dW_n,
with terminal Y_N = g(X_N). Solving it yields Y_n = u(t_n, X_n), the value
function of the associated parabolic PDE.

Shape conventions (bs = batch size):
    x : [bs, dim_x]      y : [bs, dim_y]      z : [bs, dim_y, dim_d]
    b -> [bs, dim_x]     sigma -> [bs, dim_x, dim_d]
    f -> [bs, dim_y]     g -> [bs, dim_y]
"""

from abc import ABC, abstractmethod

import torch


class Equation(ABC):
    def __init__(self, x_0, T, dim_x, dim_y, dim_d):
        self.x_0 = torch.as_tensor(x_0, dtype=torch.float32).reshape(-1)
        if self.x_0.numel() != dim_x:
            raise ValueError(f"x_0 has {self.x_0.numel()} entries, expected dim_x={dim_x}")
        self.T = float(T)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d

    @abstractmethod
    def b(self, t, x):
        """Drift, returns [bs, dim_x]."""

    @abstractmethod
    def sigma(self, t, x):
        """Diffusion matrix, returns [bs, dim_x, dim_d]."""

    @abstractmethod
    def f(self, t, x, y, z):
        """BSDE driver, returns [bs, dim_y]."""

    @abstractmethod
    def g(self, x):
        """Terminal condition u(T, .), returns [bs, dim_y]."""
