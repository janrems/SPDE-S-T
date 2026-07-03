"""M1 end-to-end: chaos-net solver recovers the additive-heat chaos coefficients.

Uses the cheap well-conditioned config (small T, few steps) with a nontrivial
terminal u_0 = sin(x). The full T=1 comparison vs the MLP lives in
scripts/m1_additive.py. The solution is affine in a, so this validates the
pipeline (solve -> recover c_0 and c_k -> order>=2 vanish); the spectral net's
high-order advantage is exercised in M2 (Wick), not here.
"""

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.equations.additive_heat import AdditiveHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.poly import multi_indices
from spde_st.recovery.coefficients import chaos_coefficients


def test_chaos_pipeline_recovers_coefficients():
    torch.manual_seed(0)
    eq = AdditiveHeat(T=0.05, u0_amp=1.0)  # nontrivial terminal, well conditioned
    box, n_steps = 1.0, 5

    def sampler(bs):
        return (torch.rand(bs, eq.N) * 2 - 1) * box

    solver = DBDPSolver(
        eq,
        chaos_factory(dim_x=1, n_param=eq.N, max_degree=3, input_scale=box),
        n_steps=n_steps,
        lr=1e-3,
        stats_samples=20_000,
        param_sampler=sampler,
    )
    solver.train(batch_size=512, itr=3000)

    n = n_steps // 2
    t = eq.T - eq.T * n / n_steps  # physical time = T - tau
    X, _ = simulate_paths(eq, 2048, n_steps)
    x = X[:, :, n]
    a = sampler(x.size(0))

    # solve accuracy against the closed form
    assert relative_l2(eq.oracle_u(t, x, a), solver.predict_u(x, n, a)) < 0.15

    # order-0 (deterministic) coefficient recovery: c_0 ~ e^{-t} sin(x)
    coeffs = chaos_coefficients(solver, n, x, max_order=2)
    assert (coeffs[(0,) * eq.N] - eq.oracle_c0(t, x).squeeze(1)).abs().mean() < 0.05, "c_0 off"

    # first-order coefficient recovery: c_(e_k) ~ c_k(t,x)
    ck_true = eq._ck_fields(t, x)
    for k in range(eq.N):
        alpha = tuple(1 if i == k else 0 for i in range(eq.N))
        assert (coeffs[alpha] - ck_true[:, k]).abs().mean() < 0.05, f"c_k off at k={k}"

    # second order vanishes (affine solution); non-vacuous since the net can represent it
    for a2 in multi_indices(eq.N, 2):
        if sum(a2) == 2:
            assert coeffs[a2].abs().mean() < 0.05, f"order-2 {a2} not ~0"
