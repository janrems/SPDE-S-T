"""Additive-noise stochastic heat equation, S-transformed and time-reversed.

Physical SPDE:  d_t U = Lap U + Wdot,  U(0,.) = u_0.
S-transform (truncated to N modes) gives the deterministic, a-parametrized IVP
    d_t u = Lap u + h_a,   u(0,.) = u_0,   h_a = sum_k a_k e_k.
To use the terminal-value BSDE solver we time-reverse: v(tau,x) = u(T-tau,x)
solves  d_tau v + Lap v + h_a(T-tau,.) = 0,  v(T) = u_0. As a backward
Kolmogorov problem this is an FBSDE with dX = sqrt(2) dW (generator Lap),
terminal g = u_0, driver f = h_a(T-tau,.). The solver returns v; physical
u(t,x) = v(T-t,x).

We use time-constant space modes e_k(s,y) = phi_k(y) (so the driver is
time-independent and the T-tau flip is a no-op here; time modes are a later
extension). phi for "sin"/"cos" at wavenumber w is sqrt(2)*sin/cos(w x), a
Laplacian eigenfunction with eigenvalue lam = w^2. Moderate wavenumbers
(w ~ 1, 2) keep lam small, so the chaos response is a substantial, slowly
varying signal (large w gives near-instant decay and a tiny, ill-conditioned
response). u_0 = u0_amp * sin(x), eigenvalue lam0 = 1.

Oracle (closed form, affine in a):
    u(t,x;a) = c0(t,x) + sum_k a_k c_k(t,x),
    c0(t,x)  = u0_amp e^{-t} sin(x),
    c_k(t,x) = phi_k(x) (1 - e^{-lam_k t}) / lam_k.
So chaos order 0 is c0, order 1 are the c_k, and every order >= 2 is exactly 0.
"""

import numpy as np
import torch

from spde_st.equations.base import Equation

SQRT2 = float(np.sqrt(2.0))


class AdditiveHeat(Equation):
    def __init__(self, modes=None, x0=0.0, T=1.0, u0_amp=0.0):
        # modes: list of (kind, wavenumber); phi = sqrt(2)*sin/cos(w x), lam = w^2
        if modes is None:
            modes = [("sin", 1.0), ("cos", 1.0), ("sin", 2.0)]
        self.modes = modes
        self.N = len(modes)
        self.lam = [float(w**2) for (_, w) in modes]
        self.lam0 = 1.0  # eigenvalue of u_0 = sin(x)
        # initial condition u_0 = u0_amp * sin(x). Default 0: the solution is then
        # the pure chaos response sum_k a_k c_k, so solve error and coefficient
        # recovery are well conditioned (no large deterministic c0 background).
        self.u0_amp = u0_amp
        super().__init__(x_0=[x0], T=T, dim_x=1, dim_y=1, dim_d=1)

    def _phi(self, x, k):
        """phi_k evaluated at x [bs,1] -> [bs,1]."""
        kind, w = self.modes[k]
        arg = w * x
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
        return self.u0_amp * torch.sin(x).reshape(-1, 1)

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
        return self.u0_amp * np.exp(-self.lam0 * t) * torch.sin(x).reshape(-1, 1)

    def oracle_u(self, t, x, a):
        """u(t,x;a) at physical time t. x [bs,1], a [bs,N] -> [bs,1]."""
        return self.oracle_c0(t, x) + (a * self._ck_fields(t, x)).sum(dim=1, keepdim=True)
