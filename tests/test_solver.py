"""Structural tests for the DBDP solver: shapes, finite loss, gradients."""

import pytest
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
    solver = DBDPSolver(eq, mlp_factory(dim_h=8), n_steps=4, stats_samples=2000)
    net = solver.net_factory(solver.dim_in, solver.dim_out)
    X, _ = simulate_paths(eq, 6, solver.n_steps)
    y, z = solver._yz(net, X[:, :, 1], 1, a=None)
    assert y.shape == (6, eq.dim_y)
    assert z.shape == (6, eq.dim_y, eq.dim_d)


def test_param_threading():
    """A param_sampler grows the net input and reaches predict; training runs."""
    eq = LinearHeat(dim=1)  # ignores a, but a must still thread through
    sampler = lambda bs: torch.zeros(bs, 2)  # noqa: E731
    solver = DBDPSolver(
        eq, mlp_factory(dim_h=8), n_steps=3, stats_samples=2000, param_sampler=sampler
    )
    assert solver.n_param == 2
    assert solver.dim_in == eq.dim_x + 1 + 2
    solver.train(batch_size=64, itr=20)
    pred = solver.predict_u(eq.x_0.view(1, -1), 0, a=torch.zeros(1, 2))
    assert torch.isfinite(pred).all()


def test_step_loss_finite_and_differentiable():
    eq = LinearHeat(dim=1)
    solver = DBDPSolver(eq, mlp_factory(dim_h=8), n_steps=4, stats_samples=2000)
    net = solver.net_factory(solver.dim_in, solver.dim_out)
    X, dW = simulate_paths(eq, 16, solver.n_steps)
    loss = solver.step_loss(net, X, dW, n=solver.n_steps - 1, a=None)  # terminal target
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_save_load_round_trip(tmp_path):
    """A reloaded solver predicts exactly what the trained one did."""
    eq = LinearHeat(dim=1)
    factory = mlp_factory(dim_h=8)
    trained = DBDPSolver(eq, factory, n_steps=3, stats_samples=2000)
    trained.train(batch_size=64, itr=20)
    ckpt = tmp_path / "solver.pt"
    trained.save(ckpt)

    restored = DBDPSolver(eq, factory, n_steps=3, stats_samples=2000).load(ckpt)
    x = torch.linspace(-1, 1, 16).reshape(-1, 1)
    for n in range(3):
        assert torch.allclose(trained.predict_u(x, n), restored.predict_u(x, n))
    assert restored.loss_history.keys() == trained.loss_history.keys()

    # a mismatched grid must not silently load
    with pytest.raises(ValueError):
        DBDPSolver(eq, factory, n_steps=5, stats_samples=2000).load(ckpt)
