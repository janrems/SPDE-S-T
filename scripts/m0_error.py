"""Example 1 (deterministic benchmark): relative L2 error of the learned solution.

Runs the configuration reported in the paper's hyperparameter table: d=1, T=1,
N_t=20, batch 4096, three hidden layers of 64 tanh units, Adam at 1e-3, 3000
iterations per backward step, seed 1234. Reports the error of the learned map
against the closed form u(t,x)=|x|^2+d(T-t) over the support of the simulated
diffusion, per time level. Step n=0 is skipped: the forward path is still the
point x_0 there, so the field is not observable off that point.

Run: PYTHONPATH=. uv run python scripts/m0_error.py
"""

import argparse
import time

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.nets.mlp import mlp_factory
from tests.fixtures import LinearHeat

p = argparse.ArgumentParser()
p.add_argument("--n_steps", type=int, default=20)
p.add_argument("--itr", type=int, default=3000)
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--dim_h", type=int, default=64)
p.add_argument("--seed", type=int, default=1234)
args = p.parse_args()

torch.manual_seed(args.seed)
eq = LinearHeat(x0=1.0, T=1.0, dim=1)
solver = DBDPSolver(eq, mlp_factory(dim_h=args.dim_h, n_hidden=3), n_steps=args.n_steps, lr=1e-3)

t0 = time.time()
solver.train(batch_size=args.batch, itr=args.itr)
print(
    f"\nExample 1: d={eq.dim_x} T={eq.T} N_t={args.n_steps} itr={args.itr} "
    f"batch={args.batch} dim_h={args.dim_h} seed={args.seed}"
)
print(f"train {time.time() - t0:.0f}s\n")
print(f"{'t':>6} {'rel L2':>9}")

X, _ = simulate_paths(eq, 8192, solver.n_steps)
errs = []
for n in range(1, solver.n_steps):
    t = eq.T * n / solver.n_steps
    x = X[:, :, n]
    e = relative_l2(eq.oracle(x, t), solver.predict_u(x, n))
    errs.append(e)
    print(f"{t:6.2f} {e:9.4f}")
print(f"\nmean {sum(errs) / len(errs):.4f}   max {max(errs):.4f}")
