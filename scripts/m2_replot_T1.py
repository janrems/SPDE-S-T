"""Redraw the two M2 comparison figures from a stored results file.

The figures in the paper came from one `m2_experiment.py --net both --T 1.0`
run whose networks were not persisted. Re-styling them therefore had to either
retrain (about 40 minutes) or replot from the recorded numbers. This script does
the latter, so the figures keep exactly the values quoted in the paper's tables.
Styling matches m2_experiment.py, minus the axes titles that duplicated the
LaTeX captions.

Run: PYTHONPATH=. uv run python scripts/m2_replot_T1.py
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results/m2_results_T1.txt"
FIGDIR = "figures"
SUFFIX = "_T1"
STYLE = {
    "chaos": ("C0-o", "chaos (polynomial, exact deriv)"),
    "mlp": ("C3-s", "MLP (autodiff deriv)"),
}


def parse(path):
    """-> {kind: {"rows": [(m, relL2)], "recon": [(M, relL2)]}}"""
    out, kind = {}, None
    for line in open(path):
        block = re.match(r"\[(\w+)\]", line)
        if block:
            kind = block.group(1)
            out[kind] = {"rows": [], "recon": []}
            continue
        if kind is None:
            continue
        row = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if row:
            out[kind]["rows"].append((int(row.group(1)), float(row.group(3))))
            continue
        rec = re.match(r"\s+M<=(\d+):\s+([\d.]+)\s*$", line)
        if rec:
            out[kind]["recon"].append((int(rec.group(1)), float(rec.group(2))))
    return out


def plot(data, key, xlabel, ylabel, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, series in data.items():
        xs = [x for x, _ in series[key]]
        ys = [y for _, y in series[key]]
        ax.plot(xs, ys, STYLE[kind][0], label=STYLE[kind][1])
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    data = parse(RESULTS)
    assert set(data) == {"chaos", "mlp"}, f"expected both nets, got {sorted(data)}"
    for kind in data:
        assert len(data[kind]["rows"]) == 5, f"{kind}: {len(data[kind]['rows'])} orders"
        assert len(data[kind]["recon"]) == 5, f"{kind}: {len(data[kind]['recon'])} truncations"
    plot(
        data,
        "rows",
        "chaos order m",
        "relative $L^2$ error of $c_m$",
        f"{FIGDIR}/m2_coeff_error{SUFFIX}.png",
    )
    plot(
        data,
        "recon",
        "truncation order M",
        "rel $L^2(\\Omega)$ error of reconstructed $U$",
        f"{FIGDIR}/m2_reconstruction{SUFFIX}.png",
    )
