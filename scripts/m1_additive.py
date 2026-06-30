"""M1 additive heat: solve with the chaos net, recover chaos coefficients,
benchmark against the closed-form oracle. Also contrast the MLP baseline's
high-order derivative noise.

Run: PYTHONPATH=. uv run python scripts/m1_additive.py
"""

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.equations.additive_heat import AdditiveHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.nets.mlp import mlp_factory
from spde_st.poly import multi_indices
from spde_st.recovery.coefficients import chaos_coefficients

N_STEPS, ITR, BOX = 10, 4000, 1.0


def make_sampler(n_param, box):
    return lambda bs: (torch.rand(bs, n_param) * 2 - 1) * box


def train(factory, eq):
    torch.manual_seed(0)
    sampler = make_sampler(eq.N, BOX)
    solver = DBDPSolver(
        eq, factory, n_steps=N_STEPS, lr=1e-3, stats_samples=20_000, param_sampler=sampler
    )
    solver.train(batch_size=512, itr=ITR)
    return solver


def evaluate(name, solver, eq):
    n = solver.n_steps // 2
    tau = eq.T * n / solver.n_steps
    t = eq.T - tau  # physical time
    X, _ = simulate_paths(eq, 4096, solver.n_steps)
    x = X[:, :, n]
    a = make_sampler(eq.N, BOX)(x.size(0))

    solve = relative_l2(eq.oracle_u(t, x, a), solver.predict_u(x, n, a))

    coeffs = chaos_coefficients(solver, n, x, max_order=2)
    c0_err = (coeffs[(0,) * eq.N] - eq.oracle_c0(t, x).squeeze(1)).abs().mean()
    ck_true = eq._ck_fields(t, x)
    ck_err = 0.0
    for k in range(eq.N):
        alpha = tuple(1 if i == k else 0 for i in range(eq.N))
        ck_err = max(ck_err, float((coeffs[alpha] - ck_true[:, k]).abs().mean()))
    order2 = max(float(coeffs[a2].abs().mean()) for a2 in multi_indices(eq.N, 2) if sum(a2) == 2)
    print(
        f"{name:7s}  solve={solve:.4f}  c0_err={float(c0_err):.4f}  "
        f"ck_err={ck_err:.4f}  max|order2|={order2:.4f}"
    )


if __name__ == "__main__":
    eq = AdditiveHeat()  # T=1, moderate wavenumbers
    print(f"N={eq.N} modes, T={eq.T}, n_steps={N_STEPS}, itr={ITR}, box={BOX}")
    evaluate(
        "chaos", train(chaos_factory(dim_x=1, n_param=eq.N, max_degree=3, input_scale=BOX), eq), eq
    )
    evaluate("mlp", train(mlp_factory(dim_h=32), eq), eq)
