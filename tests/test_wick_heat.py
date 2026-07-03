"""Tests for the Wick (multiplicative) heat equation and its oracles.

The exact constant-mode oracle is cross-checked against the Monte-Carlo oracle,
which also validates the nonlinear driver / potential convention.
"""

import pytest
import torch

from spde_st.bsde.solver import relative_l2
from spde_st.equations.wick_heat import WickHeat

CONST_MODE = [("cos", 0.0)]  # phi = sqrt(2), lambda = 0 (spatially constant)


def test_driver_is_nonlinear_in_y():
    eq = WickHeat(modes=CONST_MODE, u0_amp=1.0)
    x = torch.linspace(-1, 1, 6).reshape(-1, 1)
    y = torch.ones(6, 1)
    a = torch.full((6, 1), 0.5)
    # f = y * h_a, so doubling y doubles f
    f1 = eq.f(0.1, x, y, None, a)
    f2 = eq.f(0.1, x, 2 * y, None, a)
    assert torch.allclose(f2, 2 * f1)


def test_exact_oracle_requires_constant_mode():
    eq = WickHeat(u0_amp=1.0)  # default trig modes
    with pytest.raises(ValueError):
        eq.oracle_u_exact(0.3, torch.zeros(2, 1), torch.zeros(2, eq.N))


def test_exact_u_matches_montecarlo():
    torch.manual_seed(0)
    eq = WickHeat(modes=CONST_MODE, u0_amp=1.0, T=0.5)
    t = eq.T
    x = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    a = torch.full((8, 1), 0.5)
    exact = eq.oracle_u_exact(t, x, a)
    mc = eq.oracle_u_mc(t, x, a, n_paths=8000, n_sub=100)
    assert relative_l2(exact, mc) < 0.05


def test_exact_coefficients_match_montecarlo():
    torch.manual_seed(0)
    eq = WickHeat(modes=CONST_MODE, u0_amp=1.0, T=0.5)
    t = eq.T
    x = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    for m in range(3):  # exponential in a => all orders nonzero
        exact = eq.oracle_coeff_exact(t, x, m).squeeze(1)
        mc = eq.oracle_coeff_mc(t, x, (m,), n_paths=12000, n_sub=100)
        assert relative_l2(exact, mc) < 0.08, f"order m={m}"
