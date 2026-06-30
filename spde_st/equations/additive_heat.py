"""Additive-noise stochastic heat equation, S-transformed and time-reversed.

Physical SPDE on the torus D = [0,1]:  d_t U = Lap U + Wdot,  U(0,.) = u_0.
S-transform (truncated to N modes) gives the deterministic, a-parametrized IVP
    d_t u = Lap u + h_a,   u(0,.) = u_0,   h_a = sum_k a_k e_k.
To use the terminal-value BSDE solver we time-reverse: v(tau,x) = u(T-tau,x)
solves  d_tau v + Lap v + h_a(T-tau,.) = 0,  v(T) = u_0. As a backward
Kolmogorov problem this is an FBSDE with dX = sqrt(2) dW (generator Lap),
terminal g = u_0, driver f = h_a(T-tau,.). The solver returns v; physical
u(t,x) = v(T-t,x).

We use time-constant Fourier space modes e_k(s,y) = phi_k(y) (so the driver is
time-independent and the T-tau flip is a no-op here; time modes are a later
extension). phi for "sin"/"cos" at integer frequency f is sqrt(2)*sin/cos(2 pi f x),
a Laplacian eigenfunction with eigenvalue (2 pi f)^2. u_0 = sin(2 pi x).

Oracle (closed form, affine in a):
    u(t,x;a) = c0(t,x) + sum_k a_k c_k(t,x),
    c0(t,x)  = e^{-lam0 t} sin(2 pi x),        lam0 = (2 pi)^2,
    c_k(t,x) = phi_k(x) (1 - e^{-lam_k t}) / lam_k.
So chaos order 0 is c0, order 1 are the c_k, and every order >= 2 is exactly 0.
"""

import numpy as np
import torch

from spde_st.equations.base import Equation

SQRT2 = float(np.sqrt(2.0))


class AdditiveHeat(Equation):
    def __init__(self, modes=None, x0=0.0, T=0.05):
        if modes is None:
            modes = [("sin", 1), ("cos", 1), ("sin", 2)]
        self.modes = modes
        self.N = len(modes)
        self.lam = [float((2 * np.pi * f) ** 2) for (_, f) in modes]
        self.lam0 = float((2 * np.pi) ** 2)  # eigenvalue of u_0 = sin(2 pi x)
        super().__init__(x_0=[x0], T=T, dim_x=1, dim_y=1, dim_d=1)

    def _phi(self, x, k):
        """phi_k evaluated at x [bs,1] -> [bs,1]."""
        kind, f = self.modes[k]
        arg = 2 * np.pi * f * x
        base = torch.sin(arg) if kind == "sin" else torch.cos(arg)
        return SQRT2 * base

    def _phi_all(self, x):
        """[phi_0(x) .. phi_{N-1}(x)] -> [bs, N]."""
        return torch.cat([self._phi(x, k) for k in range(self.N)], dim=1)

    # --- FBSDE data ---
    def b(self, t, x):
        return torch.zeros_like(x)

    def sigma(self, t, x):
        return SQRT2 * torch.ones(x.size(0), self.dim_x, self.dim_d)

    def g(self, x):
        return torch.sin(2 * np.pi * x).reshape(-1, 1)

    def f(self, t, x, y, z, a):
        # driver h_a(x) = sum_k a_k phi_k(x); time-constant modes
        return (a * self._phi_all(x)).sum(dim=1, keepdim=True)

    # --- oracle (closed form) ---
    def _ck_fields(self, t, x):
        """First-order coefficient fields c_k(t,x) -> [bs, N]."""
        factors = torch.tensor(
            [(1.0 - np.exp(-self.lam[k] * t)) / self.lam[k] for k in range(self.N)]
        )
        return self._phi_all(x) * factors  # broadcast [bs,N] * [N]

    def oracle_c0(self, t, x):
        return np.exp(-self.lam0 * t) * torch.sin(2 * np.pi * x).reshape(-1, 1)

    def oracle_u(self, t, x, a):
        """u(t,x;a) at physical time t. x [bs,1], a [bs,N] -> [bs,1]."""
        return self.oracle_c0(t, x) + (a * self._ck_fields(t, x)).sum(dim=1, keepdim=True)
