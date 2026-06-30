"""M1 takeaway: per-time-slice accuracy of (1) the learned map u and (2) the
recovered chaos coefficients, against the closed-form oracle.

Nontrivial terminal u_0 = sin(x) (u0_amp=1) so the benchmark is not degenerate.
Reports both absolute (RMS) and relative L2 per physical time slice t = T - tau,
plus the signal magnitudes, so small-signal slices don't masquerade as failures.

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


def rms(v):
    return float((v**2).mean().sqrt())


def main():
    torch.manual_seed(0)
    eq = AdditiveHeat(u0_amp=1.0)  # T=1, wavenumbers 1,1,2, nontrivial terminal
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
    print(f"AdditiveHeat T={eq.T}, u0=sin(x), N={eq.N}, n_steps={N_STEPS}, itr={ITR}\n")
    head = [
        "t",
        "u_rmse",
        "|u|rms",
        "u_rel",
        "c0_rmse",
        "ck_rmse",
        "|ck|rms",
        "ck_rel",
        "ord2_rmse",
    ]
    print(" ".join(f"{h:>9}" for h in head))
    for n in range(1, N_STEPS):  # skip n=0: point support at tau=0
        t = eq.T - eq.T * n / N_STEPS
        X, _ = simulate_paths(eq, M, N_STEPS)
        x = X[:, :, n]
        a = sampler(M)

        u_true, u_net = eq.oracle_u(t, x, a), solver.predict_u(x, n, a)
        coeffs = chaos_coefficients(solver, n, x, max_order=2)
        c0_hat, c0_true = coeffs[(0,) * eq.N], eq.oracle_c0(t, x).squeeze(1)
        ck_hat = torch.stack(
            [coeffs[tuple(1 if i == k else 0 for i in range(eq.N))] for k in range(eq.N)], dim=1
        )
        ck_true = eq._ck_fields(t, x)
        ord2 = max(rms(coeffs[b]) for b in order2_idx)

        row = [
            t,
            rms(u_net - u_true),
            rms(u_true),
            relative_l2(u_true, u_net),
            rms(c0_hat - c0_true),
            rms(ck_hat - ck_true),
            rms(ck_true),
            relative_l2(ck_true, ck_hat),
            ord2,
        ]
        print(" ".join(f"{v:9.4f}" for v in row))


if __name__ == "__main__":
    main()
