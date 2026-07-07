"""M2 (Wick) figures, analogues of the additive-case figures, into .claude/figures/.

Constant-mode Wick heat (exact oracle), same config as the reported T=1 run.
Trains the monomial chaos net once and emits:
  m2_a_slice.png      - u vs a: the exponential curve (nonlinearity), with the
                        linear (tangent-at-0) part for contrast
  m2_coeff_fields.png - recovered c_0..c_3(x) vs the closed form (now all nonzero)
  m2_u_heatmap.png    - u(t,x) at fixed a: oracle, net, error
  m2_loss.png         - training objective per backward step

Run: PYTHONPATH=. uv run python scripts/m2_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spde_st.bsde.solver import DBDPSolver
from spde_st.equations.wick_heat import WickHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.recovery.coefficients import chaos_coefficients

N_STEPS, ITR, BOX, DEGREE, T, DIM_H = 20, 20000, 1.0, 6, 1.0, 64
FIGDIR = ".claude/figures"


def sampler(bs):
    return (torch.rand(bs, 1) * 2 - 1) * BOX


def physical_t(eq, n):
    return eq.T - eq.T * n / N_STEPS


def train(eq):
    torch.manual_seed(0)
    solver = DBDPSolver(
        eq,
        chaos_factory(dim_x=1, n_param=1, max_degree=DEGREE, dim_h=DIM_H, basis="monomial"),
        n_steps=N_STEPS,
        lr=1e-3,
        stats_samples=20_000,
        param_sampler=sampler,
    )
    solver.train(batch_size=512, itr=ITR)
    return solver


def fig_a_slice(solver, eq):
    n = N_STEPS // 2
    t = physical_t(eq, n)
    x0 = 0.5
    g = 200
    a = torch.linspace(-BOX, BOX, g).reshape(-1, 1)
    x_rep = torch.full((g, 1), x0)
    net = solver.predict_u(x_rep, n, a).squeeze(1)
    exact = eq.oracle_u_exact(t, x_rep, a).squeeze(1)
    # linear (tangent at a=0) part: u(0) + a * d_a u(0)
    u0 = eq.oracle_coeff_exact(t, torch.tensor([[x0]]), 0).item()
    slope = eq.oracle_coeff_exact(t, torch.tensor([[x0]]), 1).item()
    tangent = u0 + a.squeeze(1) * slope
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a.squeeze(1), exact, "k--", label="exact (Wick exponential)")
    ax.plot(a.squeeze(1), net.detach(), "C0", lw=1.6, label="net")
    ax.plot(a.squeeze(1), tangent, "C1:", label="linear part (tangent at 0)")
    ax.axvline(0, color="0.8", lw=0.7)
    ax.set_xlabel("$a$")
    ax.set_ylabel(f"$u(t={t:.2f}, x={x0})$")
    ax.set_title("u vs noise parameter: the curvature is the nonlinearity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/m2_a_slice.png", dpi=130)
    plt.close(fig)


def fig_coeff_fields(solver, eq):
    n = N_STEPS // 2
    t = physical_t(eq, n)
    x = torch.linspace(-2.5, 2.5, 200).reshape(-1, 1)
    coeffs = chaos_coefficients(solver, n, x, max_order=3)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for m in range(4):
        exact = eq.oracle_coeff_exact(t, x, m).squeeze(1)
        axes[m].plot(x[:, 0], exact, "k--", label="exact")
        axes[m].plot(x[:, 0], coeffs[(m,)], f"C{m}", lw=1.4, label="recovered")
        axes[m].set_title(f"$c_{m}(x)$")
        axes[m].set_xlabel("x")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Wick coefficient fields at t={t:.2f}: recovered vs closed form (all nonzero)")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/m2_coeff_fields.png", dpi=130)
    plt.close(fig)


def fig_u_heatmap(solver, eq):
    a_fix = torch.tensor([[0.5]])
    g = 120
    x = torch.linspace(-2.5, 2.5, g).reshape(-1, 1)
    a_rep = a_fix.expand(g, 1)
    rows = []
    for n in range(1, N_STEPS):  # skip n=0 (point support at tau=0)
        t = physical_t(eq, n)
        net_row = solver.predict_u(x, n, a_rep).squeeze(1).detach().numpy()
        true_row = eq.oracle_u_exact(t, x, a_rep).squeeze(1).numpy()
        rows.append((t, net_row, true_row))
    rows.sort(key=lambda r: r[0])
    ts = [r[0] for r in rows]
    net = np.array([r[1] for r in rows])
    true = np.array([r[2] for r in rows])
    extent = [-2.5, 2.5, min(ts), max(ts)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title in zip(
        axes, [true, net, net - true], ["oracle u", "net u", "error (net - oracle)"]
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_xlabel("x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("t")
    fig.suptitle(f"Wick u(t,x) at fixed a={a_fix.item()}")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/m2_u_heatmap.png", dpi=130)
    plt.close(fig)


def fig_loss(solver):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis
    for n in sorted(solver.loss_history):
        ax.plot(solver.loss_history[n], color=cmap(n / N_STEPS), lw=0.8, label=f"step {n}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("DBDP loss")
    ax.set_title("Training loss per backward step (Wick, T=1)")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/m2_loss.png", dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    eq = WickHeat(modes=[("cos", 0.0)], u0_amp=1.0, T=T)
    solver = train(eq)
    fig_a_slice(solver, eq)
    fig_coeff_fields(solver, eq)
    fig_u_heatmap(solver, eq)
    fig_loss(solver)
    print(f"figures written to {FIGDIR}/ (m2_a_slice, m2_coeff_fields, m2_u_heatmap, m2_loss)")


if __name__ == "__main__":
    main()
