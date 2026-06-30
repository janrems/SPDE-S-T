"""Tests for poly helpers, the chaos net, and coefficient recovery."""

import torch

from spde_st.nets.chaos import ChaosNet, chaos_factory
from spde_st.poly import multi_indices, tensor_chebyshev
from spde_st.recovery.coefficients import taylor_coefficient


def test_multi_indices_count():
    # C(n+d, d): for n=3, d=3 -> 20
    assert len(multi_indices(3, 3)) == 20
    assert (0, 0, 0) in multi_indices(3, 3)
    assert all(sum(b) <= 3 for b in multi_indices(3, 3))


def test_tensor_chebyshev_at_zero():
    betas = multi_indices(2, 2)
    a = torch.zeros(1, 2)
    psi = tensor_chebyshev(a, betas, degree=2)[0]
    # T_0(0)=1, T_1(0)=0, T_2(0)=-1; Psi_beta(0) = prod_i T_{beta_i}(0)
    for col, beta in enumerate(betas):
        expected = 1.0
        for k in beta:
            expected *= {0: 1.0, 1: 0.0, 2: -1.0}[k]
        assert abs(float(psi[col]) - expected) < 1e-6


def test_chaosnet_forward_shape():
    net = ChaosNet(dim_in=1 + 1 + 3, dim_out=2, dim_x=1, n_param=3, max_degree=3)
    feat = torch.randn(8, 1 + 1 + 3)
    assert net(feat).shape == (8, 2)


def test_taylor_coefficient_on_known_polynomial():
    # u(a) = 3 + 2 a0 - a1 + 5 a0 a1 ; coefficients are exact and known.
    def u_of_a(a):
        return 3 + 2 * a[:, 0] - a[:, 1] + 5 * a[:, 0] * a[:, 1]

    bs, n = 4, 2

    def c(alpha):
        return float(taylor_coefficient(u_of_a, n, alpha, bs)[0])

    assert abs(c((0, 0)) - 3) < 1e-6
    assert abs(c((1, 0)) - 2) < 1e-6
    assert abs(c((0, 1)) + 1) < 1e-6
    assert abs(c((1, 1)) - 5) < 1e-6  # mixed second derivative
    assert abs(c((2, 0))) < 1e-6  # genuinely zero, not vacuous


def test_chaosnet_recovers_affine_field():
    """Direct-fit the chaos net to an affine-in-a field with (t,x)-dependent
    coefficients, then recover them: c_(1,0) ~ g1(t,x), cross/2nd order ~ 0."""
    torch.manual_seed(0)
    dim_x, n_param = 1, 2
    net = ChaosNet(
        dim_in=dim_x + 1 + n_param,
        dim_out=1,
        dim_x=dim_x,
        n_param=n_param,
        max_degree=3,
        input_scale=1.0,
    )
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    def target(xt, a):  # g0 + g1*a0 + g2*a1, g_i functions of (x,t)
        x, t = xt[:, 0], xt[:, 1]
        g0, g1, g2 = x * t, torch.sin(x), t + 0.5
        return (g0 + g1 * a[:, 0] + g2 * a[:, 1]).reshape(-1, 1)

    for _ in range(3000):
        xt = torch.rand(256, 2) * 2 - 1
        a = (torch.rand(256, n_param) * 2 - 1) * 0.8
        feat = torch.cat([xt, a], dim=1)
        loss = ((net(feat) - target(xt, a)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # recover coefficients at fixed (x,t) points
    xt = torch.rand(64, 2) * 2 - 1

    def u_of_a(a):
        return net(torch.cat([xt, a], dim=1))

    c10 = taylor_coefficient(u_of_a, n_param, (1, 0), 64)
    c01 = taylor_coefficient(u_of_a, n_param, (0, 1), 64)
    c11 = taylor_coefficient(u_of_a, n_param, (1, 1), 64)
    c20 = taylor_coefficient(u_of_a, n_param, (2, 0), 64)
    assert (c10 - torch.sin(xt[:, 0])).abs().mean() < 0.05
    assert (c01 - (xt[:, 1] + 0.5)).abs().mean() < 0.05
    assert c11.abs().mean() < 0.05  # cross term, truth 0 (non-vacuous: net CAN represent it)
    assert c20.abs().mean() < 0.05


def test_chaos_factory_builds():
    factory = chaos_factory(dim_x=1, n_param=3, max_degree=2)
    net = factory(1 + 1 + 3, 2)
    assert net(torch.randn(5, 1 + 1 + 3)).shape == (5, 2)
