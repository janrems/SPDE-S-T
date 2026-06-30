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
backward loop trains step N-1 down to 0, warm-starting each net from its
successor. The solver is equation-agnostic and net-agnostic (net via factory).
"""

import numpy as np
import torch


def simulate_paths(eq, batch_size, N):
    """Euler-Maruyama forward paths.

    Returns X [bs, dim_x, N+1] and Brownian increments dW [bs, dim_d, N].
    """
    dt = eq.T / N
    X = torch.zeros(batch_size, eq.dim_x, N + 1)
    X[:, :, 0] = eq.x_0.view(1, -1)
    dW = torch.randn(batch_size, eq.dim_d, N) * np.sqrt(dt)
    for n in range(N):
        x = X[:, :, n]
        drift = eq.b(n * dt, x)
        diff = torch.matmul(eq.sigma(n * dt, x), dW[:, :, n].unsqueeze(-1)).squeeze(-1)
        X[:, :, n + 1] = x + drift * dt + diff
    return X, dW


def path_stats(eq, N, n_samples=100_000, eps=1e-4):
    """Per-step mean/std of X_n, estimated once and frozen. Shapes [dim_x, N+1]."""
    X, _ = simulate_paths(eq, n_samples, N)
    mu = X.mean(dim=0)
    sd = X.std(dim=0) + eps
    return mu, sd


class DBDPSolver:
    def __init__(self, eq, net_factory, N, lr=1e-3, stats_samples=100_000):
        self.eq = eq
        self.net_factory = net_factory
        self.N = N
        self.lr = lr
        self.dim_in = eq.dim_x + 1  # state + time
        self.dim_out = eq.dim_y + eq.dim_y * eq.dim_d  # y and z
        self.nets = {}  # n -> trained module
        self.mu, self.sd = path_stats(eq, N, n_samples=stats_samples)

    def _features(self, x, n):
        dt = self.eq.T / self.N
        xs = (x - self.mu[:, n]) / self.sd[:, n]
        tcol = torch.full((x.size(0), 1), n * dt)
        return torch.cat([xs, tcol], dim=1)

    def _yz(self, net, x, n):
        out = net(self._features(x, n))
        y = out[:, : self.eq.dim_y]
        z = out[:, self.eq.dim_y :].reshape(-1, self.eq.dim_y, self.eq.dim_d)
        return y, z

    def _y_next(self, x_np1, n):
        """Target value at step n+1: terminal g, else the successor net."""
        if n == self.N - 1:
            return self.eq.g(x_np1)
        with torch.no_grad():
            y, _ = self._yz(self.nets[n + 1], x_np1, n + 1)
        return y

    def step_loss(self, net, X, dW, n):
        """One-step DBDP residual: y_next ~= y - f*dt + z . dW_n."""
        dt = self.eq.T / self.N
        x_n, x_np1, w_n = X[:, :, n], X[:, :, n + 1], dW[:, :, n]
        y, z = self._yz(net, x_n, n)
        y_next = self._y_next(x_np1, n)
        zdw = torch.matmul(z, w_n.unsqueeze(-1)).squeeze(-1)
        est = y - self.eq.f(n * dt, x_n, y, z) * dt + zdw
        return ((y_next - est) ** 2).sum(dim=1).mean()

    def train(self, batch_size=512, itr=2000, verbose=False):
        for n in range(self.N - 1, -1, -1):
            net = self.net_factory(self.dim_in, self.dim_out)
            if n < self.N - 1:
                net.load_state_dict(self.nets[n + 1].state_dict())
            opt = torch.optim.Adam(net.parameters(), self.lr)
            for it in range(itr):
                X, dW = simulate_paths(self.eq, batch_size, self.N)
                loss = self.step_loss(net, X, dW, n)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and it % 200 == 0:
                    print(f"step {n} itr {it} loss {float(loss):.4e}")
            net.eval()
            self.nets[n] = net
        return self

    def predict_u(self, x, n):
        """u(t_n, x). For n == N returns the terminal g(x)."""
        if n == self.N:
            return self.eq.g(x)
        with torch.no_grad():
            y, _ = self._yz(self.nets[n], x, n)
        return y


def relative_l2(true, est):
    """Relative L2 error over a batch."""
    return float(torch.norm(true - est) / (torch.norm(true) + 1e-12))
