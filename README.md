# NeSTR — Neural S-Transform Reconstruction

Code accompanying the paper

> **NeSTR: Neural S-Transform Reconstruction for Gaussian SPDEs**
> Nacira Agram, Fred Espen Benth, Jan Rems

<!-- TODO: preprint link -->

## What it does

NeSTR solves Gaussian Wick-type SPDEs by learning a finite-dimensional
restriction of the S-transform. The transform turns the SPDE into a
deterministic parabolic problem carrying the truncated noise coordinate
`a` in R^N as a parameter, so the map to learn is

    (t, x, a) -> u^(N)(t, x; a),

and the Wiener-chaos coefficients of the stochastic solution are its Taylor
coefficients at the origin,

    c_alpha(t, x) = (1/alpha!) d_a^alpha u^(N)(t, x; 0).

The transformed problem is solved through its Feynman–Kac form by deep backward
dynamic programming (Huré–Pham–Warin): one network per time level, trained
backwards from the terminal datum. The network is a *polynomial* in `a` with
coefficient fields given by an ordinary MLP in `(t, x)`, so the derivatives at
`a = 0` — and hence the chaos coefficients — are read off exactly rather than
by automatic differentiation.

## Layout

    spde_st/
      bsde/solver.py            DBDP solver: forward Euler paths, frozen per-step
                                standardization, one net per time level, warm
                                starts, save/load of trained solvers
      equations/base.py         Equation interface (b, sigma, f, g, dims, T)
      equations/additive_heat.py   additive noise heat equation + closed-form oracle
      equations/wick_heat.py       multiplicative (Wick) heat equation + exact and
                                   Monte-Carlo oracles
      nets/mlp.py               plain feedforward baseline
      nets/spectral.py          polynomial-in-x net (used in the M0 comparison)
      nets/chaos.py             MLP in (t,x) times a tensor polynomial in a;
                                Chebyshev or monomial basis
      recovery/coefficients.py  chaos coefficients from the polynomial part
      poly.py                   Chebyshev and monomial feature helpers, multi-indices

    scripts/                    experiments and figure generation (see below)
    tests/                      unit tests plus end-to-end milestone tests

## Install and run

Uses [uv](https://docs.astral.sh/uv/); torch comes from the CPU-only index.

    uv sync
    uv run pytest                       # full suite
    uv run pytest --deselect tests/test_m1_additive.py   # skip the ~30 s end-to-end test
    uv run ruff check .

Scripts need the package on the path:

    PYTHONPATH=. uv run python scripts/<name>.py

## Scripts

| script | what it produces |
| --- | --- |
| `m0_compare.py` | deterministic benchmark: MLP vs polynomial net, solve accuracy and derivative quality |
| `m1_additive.py` | additive heat end to end: solve, recover coefficients, compare against the oracle |
| `m1_report.py` | per-time-slice accuracy of the learned map and of the recovered coefficients |
| `m1_figures.py` | the additive-case figures |
| `m2_experiment.py` | Wick high-order recovery, polynomial net vs MLP; writes a results table and two comparison figures |
| `m2_figures.py` | the Wick-case figures |
| `m2_replot_T1.py` | redraws the two comparison figures from a stored `m2_experiment.py` results file, without retraining |

Output goes to `.claude/` (figures under `.claude/figures/`), which is
gitignored.

`m1_figures.py` and `m2_figures.py` write a checkpoint of the trained solver on
their first run and reuse it afterwards, so re-rendering a figure does not
repeat the training. Delete the `.pt` file to force a retrain.

## Notes

Both figure scripts seed with `torch.manual_seed(0)`, so a rerun reproduces the
same numbers.

The two benchmarks use different polynomial bases in `a`: Chebyshev for the
additive equation, monomial for the multiplicative one. Both span the same
polynomial space, so recovery is exact either way; with the monomial basis the
learned coefficient of `a^nu` is the chaos coefficient itself, up to the
factorial normalization.

Recoverable chaos order is limited by solve accuracy. The absolute recovery
error is roughly constant across orders while the exact coefficients decay, so
beyond some order the signal falls below the error. The paper quantifies this
and gives the resulting rule for choosing the chaos truncation.
