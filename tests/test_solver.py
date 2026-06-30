"""Structural tests for the DBDP solver: shapes, finite loss, gradients."""

import torch

from spde_st.bsde.solver import DBDPSolver, path_stats, simulate_paths
from spde_st.nets.mlp import mlp_factory
from tests.fixtures import LinearHeat


def test_simulate_paths_shapes():
    eq = LinearHeat(dim=2)
    N, bs = 5, 8
    X, dW = simulate_paths(eq, bs, N)
    assert X.shape == (bs, eq.dim_x, N + 1)
    assert dW.shape == (bs, eq.dim_d, N)
    # path starts at x_0
    assert torch.allclose(X[:, :, 0], eq.x_0.view(1, -1).expand(bs, -1))


def test_path_stats_shapes():
    eq = LinearHeat(dim=2)
    N = 5
    mu, sd = path_stats(eq, N, n_samples=2000)
    assert mu.shape == (eq.dim_x, N + 1)
    assert sd.shape == (eq.dim_x, N + 1)
    assert (sd > 0).all()


def test_yz_shapes():
    eq = LinearHeat(dim=1)
    solver = DBDPSolver(eq, mlp_factory(dim_h=8), N=4, stats_samples=2000)
    net = solver.net_factory(solver.dim_in, solver.dim_out)
    X, _ = simulate_paths(eq, 6, solver.N)
    y, z = solver._yz(net, X[:, :, 1], 1)
    assert y.shape == (6, eq.dim_y)
    assert z.shape == (6, eq.dim_y, eq.dim_d)


def test_step_loss_finite_and_differentiable():
    eq = LinearHeat(dim=1)
    solver = DBDPSolver(eq, mlp_factory(dim_h=8), N=4, stats_samples=2000)
    net = solver.net_factory(solver.dim_in, solver.dim_out)
    X, dW = simulate_paths(eq, 16, solver.N)
    loss = solver.step_loss(net, X, dW, n=solver.N - 1)  # terminal target
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
