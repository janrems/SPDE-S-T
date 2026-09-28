"""M2 decisive experiment: high-order chaos-coefficient recovery, spectral vs MLP.

Constant-mode Wick heat, where the exact coefficients c_m = (phi t)^m/m! e^{tLap}u_0
are known for all orders. We train two nets to high accuracy and compare how well
each recovers c_0..c_M at a=0:
  * chaos net, monomial basis (polynomial in a; c_beta read directly, exact derivatives);
  * plain MLP (derivatives at 0 by automatic differentiation).
The question: does the polynomial net degrade more gracefully at high order?

This run is meant to be launched by hand (it is long). Results and a figure are
written to disk; nothing needs to stay attached while it runs.

    PYTHONPATH=. uv run python scripts/m2_experiment.py                 # accurate defaults
    PYTHONPATH=. uv run python scripts/m2_experiment.py --itr 4000 --n_steps 8   # quick

Outputs:
    results/m2_results.txt            table + config
    figures/m2_coeff_error.png
"""

import argparse
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.equations.wick_heat import WickHeat
from spde_st.nets.chaos import chaos_factory
from spde_st.nets.mlp import mlp_factory
from spde_st.recovery.coefficients import chaos_coefficients

FIGDIR = "figures"
RESULTS = "results/m2_results.txt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--itr", type=int, default=20000)
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--degree", type=int, default=6)
    p.add_argument("--max_order", type=int, default=4)
    p.add_argument("--T", type=float, default=1.0)  # the value reported in the paper
    p.add_argument("--dim_h", type=int, default=64)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", type=str, default="", help="suffix for output files (sweep runs)")
    p.add_argument("--net", choices=["both", "chaos", "mlp"], default="both")
    return p.parse_args()


def train(kind, eq, args):
    torch.manual_seed(0)

    def sampler(bs):
        return (torch.rand(bs, 1) * 2 - 1) * args.box

    if kind == "chaos":
        fac = chaos_factory(
            dim_x=1, n_param=1, max_degree=args.degree, dim_h=args.dim_h, basis="monomial"
        )
    else:
        fac = mlp_factory(dim_h=args.dim_h)
    solver = DBDPSolver(
        eq, fac, n_steps=args.n_steps, lr=args.lr, stats_samples=20_000, param_sampler=sampler
    )
    t0 = time.time()
    solver.train(batch_size=args.batch, itr=args.itr)
    return solver, sampler, time.time() - t0


def evaluate(solver, sampler, eq, args):
    n = args.n_steps // 2
    t = eq.T - eq.T * n / args.n_steps
    X, _ = simulate_paths(eq, 4096, args.n_steps)
    x = X[:, :, n]
    a = sampler(x.size(0))
    solve = relative_l2(eq.oracle_u_exact(t, x, a), solver.predict_u(x, n, a))
    coeffs = chaos_coefficients(solver, n, x, max_order=args.max_order)
    rows = []
    for m in range(args.max_order + 1):
        exact = eq.oracle_coeff_exact(t, x, m).squeeze(1)
        hat = coeffs[(m,)]
        rows.append(
            (
                m,
                float(exact.abs().mean()),
                relative_l2(exact, hat),
                float((hat - exact).abs().mean()),
            )
        )

    # U reconstruction error in L^2(Omega): ||U - sum_{m<=M} hat c_m H_m|| / ||U||,
    # with the chaos norm weight m! (Hermite orthogonality). The true tail (m>M) uses
    # the exact coefficients. Shows whether noisy high-order c_m actually matter.
    mhigh = 10
    exact_all = {m: eq.oracle_coeff_exact(t, x, m).squeeze(1) for m in range(mhigh + 1)}
    hat_all = {m: coeffs[(m,)] for m in range(args.max_order + 1)}
    full2 = sum(math.factorial(m) * exact_all[m] ** 2 for m in range(mhigh + 1))
    recon = []
    for mtrunc in range(args.max_order + 1):
        err2 = sum(math.factorial(m) * (hat_all[m] - exact_all[m]) ** 2 for m in range(mtrunc + 1))
        err2 = err2 + sum(
            math.factorial(m) * exact_all[m] ** 2 for m in range(mtrunc + 1, mhigh + 1)
        )
        recon.append((mtrunc, float((err2.mean() / full2.mean()).sqrt())))
    return t, float(solve), rows, recon


def main():
    args = parse()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    eq = WickHeat(modes=[("cos", 0.0)], u0_amp=1.0, T=args.T)  # constant mode, exact oracle

    lines = [
        f"M2 constant-mode Wick: n_steps={args.n_steps} itr={args.itr} box={args.box} "
        f"degree={args.degree} T={args.T} dim_h={args.dim_h} batch={args.batch} lr={args.lr}",
        "",
    ]
    kinds = {"both": ("chaos", "mlp"), "chaos": ("chaos",), "mlp": ("mlp",)}[args.net]
    results = {}
    for kind in kinds:
        print(f"training {kind} ...", flush=True)
        solver, sampler, secs = train(kind, eq, args)
        t, solve, rows, recon = evaluate(solver, sampler, eq, args)
        results[kind] = (rows, recon)
        lines.append(f"[{kind}] train {secs:.0f}s  solve relL2={solve:.4f}  (interior t={t:.3f})")
        lines.append(f"  {'order':>5} {'|c_m|':>10} {'relL2':>10} {'abs_err':>10}")
        for m, mag, rel, ae in rows:
            lines.append(f"  {m:>5} {mag:>10.5f} {rel:>10.4f} {ae:>10.5f}")
        lines.append("  U reconstruction rel L2(Omega) by truncation order M:")
        for mt, rel in recon:
            lines.append(f"    M<={mt}: {rel:.4f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    suffix = f"_{args.tag}" if args.tag else ""
    results_path = RESULTS.replace(".txt", f"{suffix}.txt")
    with open(results_path, "w") as fh:
        fh.write(report + "\n")

    style = {
        "chaos": ("C0-o", "chaos (polynomial, exact deriv)"),
        "mlp": ("C3-s", "MLP (autodiff deriv)"),
    }

    # figure 1: coefficient relative error vs order
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in kinds:
        rows = results[kind][0]
        ax.plot([r[0] for r in rows], [r[2] for r in rows], style[kind][0], label=style[kind][1])
    ax.set_yscale("log")
    ax.set_xlabel("chaos order m")
    ax.set_ylabel("relative $L^2$ error of $c_m$")
    ax.legend()
    fig.tight_layout()
    coeff_fig = f"{FIGDIR}/m2_coeff_error{suffix}.png"
    fig.savefig(coeff_fig, dpi=130)
    plt.close(fig)

    # figure 2: U reconstruction error vs truncation order
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in kinds:
        recon = results[kind][1]
        ax.plot([m for m, _ in recon], [r for _, r in recon], style[kind][0], label=style[kind][1])
    ax.set_yscale("log")
    ax.set_xlabel("truncation order M")
    ax.set_ylabel("rel $L^2(\\Omega)$ error of reconstructed $U$")
    ax.legend()
    fig.tight_layout()
    recon_fig = f"{FIGDIR}/m2_reconstruction{suffix}.png"
    fig.savefig(recon_fig, dpi=130)
    plt.close(fig)
    print(f"\nwrote {results_path}, {coeff_fig}, {recon_fig}")


if __name__ == "__main__":
    main()
