"""Deep backward dynamic programming (Hure-Pham-Warin) BSDE solver.

Ported and cleaned from the RobustOptimalStopping reflected-BSDE code:
  - reflection (obstacle clamping) dropped: these PDEs are plain parabolic;
  - forward noise sampled in dim_d (the original loop used dim_x, correct only
    when dim_x == dim_d);
  - input standardization uses per-step statistics estimated once by Monte
    Carlo from the forward simulator and then frozen, so train and predict see
    the same scaling. This is general -- no closed-form assumption on X;
  - dead BatchNorm and the needless retain_graph removed.

One network per time step approximates (y, z) = (u(t_n, .), grad-part). The
backward loop trains step n_steps-1 down to 0, warm-starting each net from its
successor. The solver is equation-agnostic and net-agnostic (net via factory).

Static parameter (M1+): an equation may carry a finite-dimensional parameter a
(the truncated S-transform variable, a in R^N). A param_sampler draws a per
minibatch; a is appended to the net input and passed to the driver f. It is
static along each trajectory (same a at steps n and n+1). The forward diffusion
does not depend on a, so paths are reused across a. Without a sampler the solver
behaves exactly as in M0. NOTE: n_steps is the number of *time* steps; the
parameter dimension N (modes) is separate.
"""

import numpy as np
import torch


def simulate_paths(eq, batch_size, n_steps):
    """Euler-Maruyama forward paths.

    Returns X [bs, dim_x, n_steps+1] and Brownian increments dW [bs, dim_d, n_steps].
    """
    dt = eq.T / n_steps
    X = torch.zeros(batch_size, eq.dim_x, n_steps + 1)
    X[:, :, 0] = eq.x_0.view(1, -1)
    dW = torch.randn(batch_size, eq.dim_d, n_steps) * np.sqrt(dt)
    for n in range(n_steps):
        x = X[:, :, n]
        drift = eq.b(n * dt, x)
        diff = torch.matmul(eq.sigma(n * dt, x), dW[:, :, n].unsqueeze(-1)).squeeze(-1)
        X[:, :, n + 1] = x + drift * dt + diff
    return X, dW


def path_stats(eq, n_steps, n_samples=100_000, eps=1e-4):
    """Per-step mean/std of X_n, estimated once and frozen. Shapes [dim_x, n_steps+1]."""
    X, _ = simulate_paths(eq, n_samples, n_steps)
    mu = X.mean(dim=0)
    sd = X.std(dim=0) + eps
    return mu, sd


class DBDPSolver:
    def __init__(
        self, eq, net_factory, n_steps, lr=1e-3, stats_samples=100_000, param_sampler=None
    ):
        self.eq = eq
        self.net_factory = net_factory
        self.n_steps = n_steps
        self.lr = lr
        self.param_sampler = param_sampler  # callable(batch_size) -> [bs, N], or None
        self.n_param = 0 if param_sampler is None else param_sampler(1).shape[1]
        self.dim_in = eq.dim_x + 1 + self.n_param  # state + time + parameter a
        self.dim_out = eq.dim_y + eq.dim_y * eq.dim_d  # y and z
        self.nets = {}  # n -> trained module
        self.mu, self.sd = path_stats(eq, n_steps, n_samples=stats_samples)

    def _features(self, x, n, a):
        dt = self.eq.T / self.n_steps
        xs = (x - self.mu[:, n]) / self.sd[:, n]
        tcol = torch.full((x.size(0), 1), n * dt)
        parts = [xs, tcol]
        if a is not None:
            parts.append(a)
        return torch.cat(parts, dim=1)

    def _yz(self, net, x, n, a):
        out = net(self._features(x, n, a))
        y = out[:, : self.eq.dim_y]
        z = out[:, self.eq.dim_y :].reshape(-1, self.eq.dim_y, self.eq.dim_d)
        return y, z

    def _y_next(self, x_np1, n, a):
        """Target value at step n+1: terminal g, else the successor net (same a)."""
        if n == self.n_steps - 1:
            return self.eq.g(x_np1)
        with torch.no_grad():
            y, _ = self._yz(self.nets[n + 1], x_np1, n + 1, a)
        return y

    def step_loss(self, net, X, dW, n, a):
        """One-step DBDP residual: y_next ~= y - f*dt + z . dW_n."""
        dt = self.eq.T / self.n_steps
        x_n, x_np1, w_n = X[:, :, n], X[:, :, n + 1], dW[:, :, n]
        y, z = self._yz(net, x_n, n, a)
        y_next = self._y_next(x_np1, n, a)
        zdw = torch.matmul(z, w_n.unsqueeze(-1)).squeeze(-1)
        est = y - self.eq.f(n * dt, x_n, y, z, a) * dt + zdw
        return ((y_next - est) ** 2).sum(dim=1).mean()

    def train(self, batch_size=512, itr=2000, verbose=False):
        for n in range(self.n_steps - 1, -1, -1):
            net = self.net_factory(self.dim_in, self.dim_out)
            if n < self.n_steps - 1:
                net.load_state_dict(self.nets[n + 1].state_dict())
            opt = torch.optim.Adam(net.parameters(), self.lr)
            for it in range(itr):
                X, dW = simulate_paths(self.eq, batch_size, self.n_steps)
                a = None if self.param_sampler is None else self.param_sampler(batch_size)
                loss = self.step_loss(net, X, dW, n, a)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and it % 200 == 0:
                    print(f"step {n} itr {it} loss {float(loss):.4e}")
            net.eval()
            self.nets[n] = net
        return self

    def predict_u(self, x, n, a=None):
        """u(t_n, x; a). For n == n_steps returns the terminal g(x)."""
        if n == self.n_steps:
            return self.eq.g(x)
        with torch.no_grad():
            y, _ = self._yz(self.nets[n], x, n, a)
        return y


def relative_l2(true, est):
    """Relative L2 error over a batch."""
    return float(torch.norm(true - est) / (torch.norm(true) + 1e-12))
