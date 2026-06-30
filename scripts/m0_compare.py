"""Head-to-head on M0: MLP vs spectral net, solve accuracy and derivative quality.

Both solve the deterministic LinearHeat PDE (u = x^2 + (T-t), so d/dx = 2x,
d2/dx2 = 2, d3/dx3 = 0). We report interior solve error and the error of each
derivative order against the known truth. The spectral net is a polynomial in
x, so its high-order derivatives stay clean; the tanh MLP degrades. This
previews the M1 stability story, with x standing in for the noise parameter a.

Run: PYTHONPATH=. uv run python scripts/m0_compare.py
"""

import torch

from spde_st.bsde.solver import DBDPSolver, relative_l2, simulate_paths
from spde_st.nets.mlp import mlp_factory
from spde_st.nets.spectral import spectral_factory
from tests.fixtures import LinearHeat

# Both nets trained at the same budget. The spectral net (a linear model on
# Chebyshev features) averages out the DBDP target noise more slowly than the
# MLP, so it needs the larger budget to converge.
N, ITR = 6, 5000


def train(factory, seed=0):
    torch.manual_seed(seed)
    eq = LinearHeat(x0=1.0, T=1.0, dim=1)
    solver = DBDPSolver(eq, factory, N=N, lr=1e-3, stats_samples=20_000)
    solver.train(batch_size=512, itr=ITR)
    return eq, solver


def _grad(out, inp):
    # allow_unused: a derivative beyond the polynomial degree is structurally
    # zero, so autograd reports the input as unused -> return zeros.
    g = torch.autograd.grad(out.sum(), inp, create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inp) if g is None else g


def derivatives(solver, n, xs):
    x = xs.reshape(-1, 1).clone().requires_grad_(True)
    y, _ = solver._yz(solver.nets[n], x, n)
    d1 = _grad(y, x)
    d2 = _grad(d1, x)
    d3 = _grad(d2, x)
    return y.detach(), d1.detach(), d2.detach(), d3.detach()


def report(name, eq, solver):
    n = solver.N // 2
    t = eq.T * n / solver.N
    # solve accuracy over the support
    X, _ = simulate_paths(eq, 4096, solver.N)
    x = X[:, :, n]
    solve = relative_l2(eq.oracle(x, t), solver.predict_u(x, n))
    # derivatives on a grid within +-2 std of the step-n marginal
    mu, sd = float(solver.mu[0, n]), float(solver.sd[0, n])
    xs = torch.linspace(mu - 2 * sd, mu + 2 * sd, 200)
    _, d1, d2, d3 = derivatives(solver, n, xs)
    e1 = (d1.squeeze() - 2 * xs).abs().mean()
    e2 = (d2.squeeze() - 2.0).abs().mean()
    e3 = d3.squeeze().abs().mean()  # truth is 0
    print(f"{name:9s}  solve relL2={solve:.4f}  |d1-2x|={e1:.4f}  |d2-2|={e2:.4f}  |d3|={e3:.4f}")


if __name__ == "__main__":
    eq, mlp = train(mlp_factory(dim_h=32))
    report("mlp", eq, mlp)
    eq, spec = train(spectral_factory(dim_spectral=1, degree=4))
    report("spectral", eq, spec)
