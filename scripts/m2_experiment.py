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
    .claude/m2_results.txt            table + config
    .claude/figures/m2_coeff_error.png
"""

import argparse
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

FIGDIR = ".claude/figures"
RESULTS = ".claude/m2_results.txt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--itr", type=int, default=20000)
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--degree", type=int, default=6)
    p.add_argument("--max_order", type=int, default=4)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--dim_h", type=int, default=64)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
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
        rows.append((m, float(exact.abs().mean()), relative_l2(exact, hat), float((hat - exact).abs().mean())))
    return t, float(solve), rows


def main():
    args = parse()
    os.makedirs(FIGDIR, exist_ok=True)
    eq = WickHeat(modes=[("cos", 0.0)], u0_amp=1.0, T=args.T)  # constant mode, exact oracle

    lines = [
        f"M2 constant-mode Wick: n_steps={args.n_steps} itr={args.itr} box={args.box} "
        f"degree={args.degree} T={args.T} dim_h={args.dim_h} batch={args.batch} lr={args.lr}",
        "",
    ]
    results = {}
    for kind in ("chaos", "mlp"):
        print(f"training {kind} ...", flush=True)
        solver, sampler, secs = train(kind, eq, args)
        t, solve, rows = evaluate(solver, sampler, eq, args)
        results[kind] = rows
        lines.append(f"[{kind}] train {secs:.0f}s  solve relL2={solve:.4f}  (interior t={t:.3f})")
        lines.append(f"  {'order':>5} {'|c_m|':>10} {'relL2':>10} {'abs_err':>10}")
        for m, mag, rel, ae in rows:
            lines.append(f"  {m:>5} {mag:>10.5f} {rel:>10.4f} {ae:>10.5f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(RESULTS, "w") as fh:
        fh.write(report + "\n")

    # figure: relative error vs order, chaos vs mlp
    fig, ax = plt.subplots(figsize=(7, 4))
    orders = [r[0] for r in results["chaos"]]
    ax.plot(orders, [r[2] for r in results["chaos"]], "C0-o", label="chaos (polynomial, exact deriv)")
    ax.plot(orders, [r[2] for r in results["mlp"]], "C3-s", label="MLP (autodiff deriv)")
    ax.set_yscale("log")
    ax.set_xlabel("chaos order m")
    ax.set_ylabel("relative $L^2$ error of $c_m$")
    ax.set_title("High-order coefficient recovery: spectral vs MLP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/m2_coeff_error.png", dpi=130)
    print(f"\nwrote {RESULTS} and {FIGDIR}/m2_coeff_error.png")


if __name__ == "__main__":
    main()
