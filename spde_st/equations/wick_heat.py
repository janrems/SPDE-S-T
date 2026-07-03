"""Multiplicative (Wick) stochastic heat equation, S-transformed.

Physical SPDE:  d_t U = Lap U + U (Wick) Wdot,  U(0,.) = u_0.
By the Wick-to-product property of the S-transform, the transformed equation is
    d_t u = Lap u + u h_a,   u(0,.) = u_0,   h_a = sum_k a_k phi_k,
i.e. a heat equation with a *potential* h_a. The driver is now nonlinear in the
solution, f(t,x,y,z,a) = y h_a(x), so this is the first genuinely nonlinear
BSDE. Time-reversed to terminal-value form exactly as in the additive case
(dX = sqrt(2) dW, terminal g = u_0), the physical solution is u(t,x)=v(T-t,x).

Feynman-Kac gives an exponential dependence on a,
    u(t,x;a) = E[ u_0(X_t) exp( int_0^t h_a(X_s) ds ) ],
so every chaos coefficient is nonzero:
    c_alpha(t,x) = (1/alpha!) E[ u_0(X_t) prod_k A_k^{alpha_k} ],  A_k = int_0^t phi_k(X_s) ds.
This is the regime that tests stable recovery of high-order coefficients.

Oracles:
  * constant mode (phi = const, lam = 0): the potential is deterministic, so
    u(t,x;a) = exp(a phi t) e^{t Lap} u_0 in closed form, with
    c_(m)(t,x) = (phi t)^m / m! * e^{t Lap} u_0 -- exact, all orders nonzero;
  * general spatial modes: Monte-Carlo estimates of the expectations above.
"""

from math import factorial

import numpy as np
import torch

from spde_st.equations.additive_heat import SQRT2, AdditiveHeat


class WickHeat(AdditiveHeat):
    def f(self, t, x, y, z, a):
        # nonlinear driver: y * h_a(x), h_a = sum_k a_k phi_k
        return y * (a * self._phi_all(x)).sum(dim=1, keepdim=True)

    # --- exact oracle: single spatially-constant mode ---
    def _assert_constant(self):
        if not (self.N == 1 and self.lam[0] == 0.0):
            raise ValueError("exact oracle only valid for a single constant mode (lam=0)")

    def _phi_const(self):
        return float(self._phi(torch.zeros(1, 1), 0)[0, 0])  # phi_0 value (constant)

    def oracle_u_exact(self, t, x, a):
        """u(t,x;a) = exp(a phi t) * e^{t Lap} u_0, valid for a constant mode."""
        self._assert_constant()
        heat = self.u0_amp * np.exp(-self.lam0 * t) * torch.sin(x)  # e^{t Lap} u_0
        return torch.exp(a[:, :1] * self._phi_const() * t) * heat

    def oracle_coeff_exact(self, t, x, m):
        """c_(m)(t,x) = (phi t)^m / m! * e^{t Lap} u_0, valid for a constant mode."""
        self._assert_constant()
        heat = self.u0_amp * np.exp(-self.lam0 * t) * torch.sin(x)
        return (self._phi_const() * t) ** m / factorial(m) * heat  # [bs,1]

    # --- Monte-Carlo oracle: general modes ---
    def _mc_paths(self, x0, t, n_sub):
        """Euler paths dX = sqrt(2) dW from per-sample starts x0 [B,1] -> [B,1,n_sub+1]."""
        dt = t / n_sub
        B = x0.size(0)
        X = torch.zeros(B, 1, n_sub + 1)
        X[:, :, 0] = x0
        for n in range(n_sub):
            X[:, :, n + 1] = X[:, :, n] + SQRT2 * torch.randn(B, 1) * np.sqrt(dt)
        return X

    def oracle_u_mc(self, t, x, a, n_paths=2000, n_sub=200):
        """MC estimate of u(t,x;a) at grid points x [G,1], a [G,N] -> [G,1]."""
        g = x.size(0)
        x0 = x.repeat_interleave(n_paths, 0)
        aB = a.repeat_interleave(n_paths, 0)
        X = self._mc_paths(x0, t, n_sub)
        dt = t / n_sub
        integ = torch.zeros(x0.size(0), 1)
        for n in range(n_sub):
            integ += (aB * self._phi_all(X[:, :, n])).sum(dim=1, keepdim=True) * dt
        val = self.g(X[:, :, -1]) * torch.exp(integ)
        return val.reshape(g, n_paths).mean(dim=1, keepdim=True)

    def oracle_coeff_mc(self, t, x, alpha, n_paths=4000, n_sub=200):
        """MC estimate of c_alpha(t,x) at grid points x [G,1] -> [G]."""
        g = x.size(0)
        x0 = x.repeat_interleave(n_paths, 0)
        X = self._mc_paths(x0, t, n_sub)
        dt = t / n_sub
        A = torch.zeros(x0.size(0), self.N)  # occupation integrals A_k
        for n in range(n_sub):
            A += self._phi_all(X[:, :, n]) * dt
        prod = torch.ones(x0.size(0))
        for k, ak in enumerate(alpha):
            prod = prod * A[:, k] ** ak
        val = self.g(X[:, :, -1]).squeeze(1) * prod
        denom = 1
        for o in alpha:
            denom *= factorial(o)
        return val.reshape(g, n_paths).mean(dim=1) / denom
