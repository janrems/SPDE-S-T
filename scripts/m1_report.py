"""M1 takeaway: per-time-slice accuracy of (1) the learned map u and
(2) the recovered chaos coefficients, against the closed-form oracle.

Physical time t = T - tau_n runs over every solver step. For each slice we
report the solve error (relative L2 of u over the x-support and a-box) and the
order-1 coefficient error (relative L2 of the recovered c_k fields), plus the
order-2 leakage (truth 0) as a fraction of the order-1 signal.

Run: PYTHONPATH=. uv run python scripts/m1_report.py
"""

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.equations.additive_heat import AdditiveHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.poly import multi_indices
from spde_st.recovery.coefficients import chaos_coefficients

N_STEPS, ITR, BOX, M = 10, 5000, 1.0, 4096


def sampler(bs):
    return (torch.rand(bs, 3) * 2 - 1) * BOX


def main():
    torch.manual_seed(0)
    eq = AdditiveHeat()  # T=1, wavenumbers 1,1,2
    solver = DBDPSolver(
        eq,
        chaos_factory(dim_x=1, n_param=eq.N, max_degree=3, input_scale=BOX),
        n_steps=N_STEPS,
        lr=1e-3,
        stats_samples=20_000,
        param_sampler=sampler,
    )
    solver.train(batch_size=512, itr=ITR)

    order2_idx = [b for b in multi_indices(eq.N, 2) if sum(b) == 2]
    print(f"AdditiveHeat T={eq.T}, N={eq.N} modes, n_steps={N_STEPS}, itr={ITR}\n")
    print(f"{'t':>6} {'u relL2':>9} {'c_k relL2':>10} {'order2/ck':>10}")
    for n in range(N_STEPS):  # t = T - tau_n, from T down to dt
        t = eq.T - eq.T * n / N_STEPS
        X, _ = simulate_paths(eq, M, N_STEPS)
        x = X[:, :, n]
        a = sampler(M)

        u_rel = relative_l2(eq.oracle_u(t, x, a), solver.predict_u(x, n, a))

        coeffs = chaos_coefficients(solver, n, x, max_order=2)
        ck_hat = torch.stack(
            [coeffs[tuple(1 if i == k else 0 for i in range(eq.N))] for k in range(eq.N)], dim=1
        )
        ck_true = eq._ck_fields(t, x)
        ck_rel = relative_l2(ck_true, ck_hat)
        ck_scale = ck_true.abs().mean().item() + 1e-12
        order2 = max(coeffs[b].abs().mean().item() for b in order2_idx)
        print(f"{t:6.2f} {u_rel:9.4f} {ck_rel:10.4f} {order2 / ck_scale:10.4f}")


if __name__ == "__main__":
    main()
