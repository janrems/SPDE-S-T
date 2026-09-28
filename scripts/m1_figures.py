"""Generate M1 figures into figures/ (gitignored).

Trains the chaos-net solver once on additive heat (nontrivial terminal) and emits:
  loss_convergence.png  - per-step training loss vs iteration
  a_slices.png          - u vs each a_k at fixed (t,x): net vs oracle, slope at 0 = c_k
  coeff_fields.png      - recovered c0, c_k(x) vs closed form
  u_heatmap.png         - u over (t,x) at fixed a: net, oracle, error
  error_vs_time.png     - absolute RMSE of u and c_k vs t, with signal magnitude

Run: PYTHONPATH=. uv run python scripts/m1_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spde_st.bsde.solver import DBDPSolver, simulate_paths
from spde_st.equations.additive_heat import AdditiveHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.recovery.coefficients import chaos_coefficients

N_STEPS, ITR, BOX, M = 10, 5000, 1.0, 4096
FIGDIR = "figures"
CKPT = "results/m1_solver.pt"  # delete to force a retrain


def sampler(bs):
    return (torch.rand(bs, 3) * 2 - 1) * BOX


def rms(v):
    return float((v**2).mean().sqrt())


def physical_t(eq, n):
    return eq.T - eq.T * n / N_STEPS


def train(eq):
    torch.manual_seed(0)
    solver = DBDPSolver(
        eq,
        chaos_factory(dim_x=1, n_param=eq.N, max_degree=3, input_scale=BOX),
        n_steps=N_STEPS,
        lr=1e-3,
        stats_samples=20_000,
        param_sampler=sampler,
    )
    if os.path.exists(CKPT):
        return solver.load(CKPT)
    solver.train(batch_size=512, itr=ITR)
    solver.save(CKPT)
    return solver


def fig_loss(solver):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis
    for n in sorted(solver.loss_history):
        ax.plot(solver.loss_history[n], color=cmap(n / N_STEPS), lw=0.8, label=f"step {n}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("DBDP loss")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/loss_convergence.png", dpi=130)
    plt.close(fig)


def fig_a_slices(solver, eq):
    n = N_STEPS // 2
    t = physical_t(eq, n)
    x0 = 0.5
    G = 200
    grid = torch.linspace(-BOX, BOX, G).reshape(-1, 1)
    x_rep = torch.full((G, 1), x0)
    fig, axes = plt.subplots(1, eq.N, figsize=(4 * eq.N, 3.4), sharey=True)
    for k in range(eq.N):
        a = torch.zeros(G, eq.N)
        a[:, k] = grid[:, 0]
        net = solver.predict_u(x_rep, n, a).squeeze(1)
        true = eq.oracle_u(t, x_rep, a).squeeze(1)
        ax = axes[k]
        ax.plot(grid[:, 0], true, "k--", label="oracle")
        ax.plot(grid[:, 0], net.detach(), "C0", lw=1.6, label="net")
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set_xlabel(f"$a_{{{k + 1}}}$")
        ax.set_title(f"slope at 0 = $c_{{e_{k + 1}}}$")
    axes[0].set_ylabel(f"$u(t={t:.2f}, x={x0})$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/a_slices.png", dpi=130)
    plt.close(fig)


def fig_coeff_fields(solver, eq):
    n = N_STEPS // 2
    t = physical_t(eq, n)
    G = 200
    x = torch.linspace(-2.5, 2.5, G).reshape(-1, 1)
    coeffs = chaos_coefficients(solver, n, x, max_order=1)
    c0_true = eq.oracle_c0(t, x).squeeze(1)
    ck_true = eq._ck_fields(t, x)
    fig, axes = plt.subplots(1, eq.N + 1, figsize=(4 * (eq.N + 1), 3.4))
    axes[0].plot(x[:, 0], c0_true, "k--", label="oracle")
    axes[0].plot(x[:, 0], coeffs[(0,) * eq.N], "C3", lw=1.4, label="recovered")
    axes[0].set_title("order 0 (deterministic)")
    axes[0].legend(fontsize=8)
    for k in range(eq.N):
        alpha = tuple(1 if i == k else 0 for i in range(eq.N))
        axes[k + 1].plot(x[:, 0], ck_true[:, k], "k--")
        axes[k + 1].plot(x[:, 0], coeffs[alpha], f"C{k}", lw=1.4)
        axes[k + 1].set_title(f"order 1, mode {k + 1}: $c_{{e_{k + 1}}}(x)$")
    for ax in axes:
        ax.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/coeff_fields.png", dpi=130)
    plt.close(fig)


def fig_u_heatmap(solver, eq):
    a_fix = torch.tensor([[0.5, -0.5, 0.5]])
    G = 120
    x = torch.linspace(-2.5, 2.5, G).reshape(-1, 1)
    a_rep = a_fix.expand(G, eq.N)
    # skip n=0: at tau=0 the diffusion is a point (X_0=x_0), so the field is only
    # known at x_0 and standardization (sd~0) blows up off it. Sort rows by t.
    rows = []
    for n in range(1, N_STEPS):
        t = physical_t(eq, n)
        net_row = solver.predict_u(x, n, a_rep).squeeze(1).detach().numpy()
        true_row = eq.oracle_u(t, x, a_rep).squeeze(1).numpy()
        rows.append((t, net_row, true_row))
    import numpy as np

    rows.sort(key=lambda r: r[0])
    ts = [r[0] for r in rows]
    net = np.array([r[1] for r in rows])
    true = np.array([r[2] for r in rows])
    err = net - true
    extent = [-2.5, 2.5, min(ts), max(ts)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title in zip(
        axes, [true, net, err], ["oracle u", "net u", "error (net - oracle)"]
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_xlabel("x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("t")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/u_heatmap.png", dpi=130)
    plt.close(fig)


def fig_error_vs_time(solver, eq):
    ts, u_err, u_sig, ck_err, ck_sig = [], [], [], [], []
    for n in range(1, N_STEPS):  # skip n=0 (point support at tau=0)
        t = physical_t(eq, n)
        X, _ = simulate_paths(eq, M, N_STEPS)
        x = X[:, :, n]
        a = sampler(M)
        u_err.append(rms(solver.predict_u(x, n, a) - eq.oracle_u(t, x, a)))
        u_sig.append(rms(eq.oracle_u(t, x, a)))
        coeffs = chaos_coefficients(solver, n, x, max_order=1)
        ck_hat = torch.stack(
            [coeffs[tuple(1 if i == k else 0 for i in range(eq.N))] for k in range(eq.N)], dim=1
        )
        ck_true = eq._ck_fields(t, x)
        ck_err.append(rms(ck_hat - ck_true))
        ck_sig.append(rms(ck_true))
        ts.append(t)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, u_err, "C0-o", label="u RMSE")
    ax.plot(ts, u_sig, "C0:", alpha=0.5, label="||u|| (signal)")
    ax.plot(ts, ck_err, "C3-o", label="c_k RMSE")
    ax.plot(ts, ck_sig, "C3:", alpha=0.5, label="||c_k|| (signal)")
    ax.set_xlabel("t")
    ax.set_ylabel("absolute RMS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/error_vs_time.png", dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    eq = AdditiveHeat(u0_amp=1.0)
    solver = train(eq)
    fig_loss(solver)
    fig_a_slices(solver, eq)
    fig_coeff_fields(solver, eq)
    fig_u_heatmap(solver, eq)
    fig_error_vs_time(solver, eq)
    print(f"figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
