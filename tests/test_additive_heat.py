"""Tests for the AdditiveHeat equation and its oracle.

The decisive check is that the closed-form oracle agrees with a Monte-Carlo
Feynman-Kac estimate -- this validates the oracle AND that the equation's
sigma/driver conventions are consistent.
"""

import torch

from spde_st.bsde.solver import simulate_paths
from spde_st.equations.additive_heat import AdditiveHeat


def test_data_shapes():
    eq = AdditiveHeat()
    x = torch.linspace(-0.3, 0.3, 7).reshape(-1, 1)
    a = torch.zeros(7, eq.N)
    assert eq.b(0.0, x).shape == (7, 1)
    assert eq.sigma(0.0, x).shape == (7, 1, 1)
    assert eq.g(x).shape == (7, 1)
    assert eq.f(0.0, x, None, None, a).shape == (7, 1)


def test_oracle_affine_in_a():
    eq = AdditiveHeat()
    x = torch.linspace(-0.3, 0.3, 5).reshape(-1, 1)
    a = torch.randn(5, eq.N)
    t = 0.03
    base = eq.oracle_u(t, x, torch.zeros_like(a))
    lin_a = eq.oracle_u(t, x, a) - base
    lin_2a = eq.oracle_u(t, x, 2 * a) - base
    assert torch.allclose(
        lin_2a, 2 * lin_a, atol=1e-5
    )  # affine => doubling a doubles the linear part


def test_oracle_matches_montecarlo():
    """u(t,x;a) closed form vs MC of the forward Feynman-Kac representation
    u(t,x) = E[u_0(X_t) + int_0^t h_a(X_s) ds],  dX = sqrt(2) dW from x."""
    torch.manual_seed(0)
    base = AdditiveHeat(T=0.05)
    t, x0 = base.T, 0.3
    a = torch.tensor([[0.5, -0.3, 0.2]])

    sim = AdditiveHeat(modes=base.modes, x0=x0, T=t)
    M, nst = 60_000, 100
    X, _ = simulate_paths(sim, M, nst)  # [M,1,nst+1]
    dt = t / nst
    a_M = a.expand(M, base.N)
    integral = torch.zeros(M, 1)
    for n in range(nst):  # left Riemann sum of int_0^t h_a(X_s) ds
        integral += sim.f(n * dt, X[:, :, n], None, None, a_M) * dt
    u_mc = (sim.g(X[:, :, -1]) + integral).mean()

    x0t = torch.tensor([[x0]])
    u_oracle = float(base.oracle_u(t, x0t, a))
    assert abs(float(u_mc) - u_oracle) < 0.03, f"MC {float(u_mc):.4f} vs oracle {u_oracle:.4f}"


def test_higher_order_coefficients_vanish():
    """The solution is affine in a, so c0 and c_k are the only nonzero
    coefficients (a sanity check on the oracle, not the solver)."""
    eq = AdditiveHeat()
    x = torch.tensor([[0.2]])
    # second difference in a single direction is 0 (affine)
    a0 = torch.zeros(1, eq.N)
    e0 = torch.zeros(1, eq.N)
    e0[0, 0] = 1.0
    h = 0.1
    second_diff = (
        eq.oracle_u(0.02, x, h * e0) - 2 * eq.oracle_u(0.02, x, a0) + eq.oracle_u(0.02, x, -h * e0)
    )
    assert abs(float(second_diff)) < 1e-6
