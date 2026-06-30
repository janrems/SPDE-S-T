"""Tests for the spectral net: Chebyshev features, accuracy, clean derivatives.

The DBDP head-to-head against the MLP lives in scripts/m0_compare.py. Here we
unit-test the layer's intrinsic properties (the smoke test already proves DBDP
itself works) plus a cheap check that it plugs into the solver.
"""

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2
from spde_st.nets.spectral import SpectralNet, chebyshev_features, spectral_factory
from tests.fixtures import LinearHeat


def test_chebyshev_features_values():
    s = torch.tensor([[0.5]])
    feats = chebyshev_features(s, degree=3)  # [1, 4]
    # T_0=1, T_1=s, T_2=2s^2-1, T_3=4s^3-3s
    expected = torch.tensor([[1.0, 0.5, -0.5, -1.0]])
    assert torch.allclose(feats, expected, atol=1e-6)


def test_forward_shape():
    net = SpectralNet(dim_in=3, dim_out=2, dim_spectral=1, degree=4)
    out = net(torch.randn(8, 3))
    assert out.shape == (8, 2)


def _grad(out, inp):
    g = torch.autograd.grad(out.sum(), inp, create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inp) if g is None else g


def test_fit_and_clean_derivatives():
    """On a bounded domain the polynomial net fits x^2 and gives clean
    derivatives: d2 ~ 2, d3 ~ 0. Chebyshev is a bounded-domain basis, so we
    train on [-1.5, 1.5] (input_scale maps it to [-1, 1]); this is also M1's
    situation, where the spectral variable a lives in a box near 0. Contrast
    M0's diffusion variable x, which is Gaussian/unbounded -- a poor Chebyshev
    domain (see scripts/m0_compare.py)."""
    torch.manual_seed(0)
    half = 1.5
    net = SpectralNet(dim_in=1, dim_out=1, dim_spectral=1, degree=4, input_scale=half)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(4000):
        x = (torch.rand(512, 1) * 2 - 1) * half
        loss = ((net(x) - x**2) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    xs = torch.linspace(-half, half, 200).reshape(-1, 1).requires_grad_(True)
    y = net(xs)
    d1 = _grad(y, xs)
    d2 = _grad(d1, xs)
    d3 = _grad(d2, xs)
    assert relative_l2(xs.detach() ** 2, y.detach()) < 0.01
    assert (d1.detach().squeeze() - 2 * xs.detach().squeeze()).abs().mean() < 0.05
    assert (d2.detach() - 2.0).abs().mean() < 0.05
    assert d3.detach().abs().mean() < 0.1  # truth is 0; clean high-order derivative


def test_integrates_with_solver():
    """Spectral net plugs into the solver via the factory and predicts finite values."""
    torch.manual_seed(0)
    eq = LinearHeat(dim=1)
    solver = DBDPSolver(eq, spectral_factory(dim_spectral=1, degree=4), N=3, stats_samples=2000)
    solver.train(batch_size=128, itr=50)
    pred = solver.predict_u(eq.x_0.view(1, -1), 0)
    assert torch.isfinite(pred).all()
