"""Smoke test: the DBDP solver recovers a known closed-form PDE solution.

LinearHeat has u(t, x) = |x|^2 + dim*(T - t). We train the solver once and
check it matches, both at an interior time step (over the diffusion's support)
and at the initial point u(0, x_0). This validates the solver before any SPDE,
transform, or spectral-net machinery is added. With f = 0 the BSDE scheme has
no discretization error, so any gap is pure network fit.
"""

import pytest
import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.nets.mlp import mlp_factory
from tests.fixtures import LinearHeat


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    eq = LinearHeat(x0=1.0, T=1.0, dim=1)
    solver = DBDPSolver(eq, mlp_factory(dim_h=32), n_steps=6, lr=1e-3, stats_samples=20_000)
    solver.train(batch_size=512, itr=1500)
    return eq, solver


def test_interior_matches_oracle(trained):
    eq, solver = trained
    n_eval = solver.n_steps // 2
    t = eq.T * n_eval / solver.n_steps
    X, _ = simulate_paths(eq, 4096, solver.n_steps)
    x = X[:, :, n_eval]
    err = relative_l2(eq.oracle(x, t), solver.predict_u(x, n_eval))
    assert err < 0.05, f"interior relative L2 = {err:.4f}"


def test_initial_point_matches_oracle(trained):
    eq, solver = trained
    x0 = eq.x_0.view(1, -1)
    pred = float(solver.predict_u(x0, 0))
    true = float(eq.oracle(x0, 0.0))  # 1 + 1*(1-0) = 2
    assert abs(pred - true) / true < 0.05, f"u(0,x0) pred={pred:.4f} true={true:.4f}"
