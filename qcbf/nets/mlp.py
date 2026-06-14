"""Framework-free ReLU MLPs (NumPy only).

Design notes
------------
* The deployed artifact is deliberately *all piecewise-linear*:
  ReLU hidden layers everywhere, and the policy head is a hardtanh clamp
      pi(x) = clip(w h + b, -omega_max, omega_max)
           = ReLU(y + omega) - ReLU(y - omega) - omega ,
  i.e. exactly expressible with two extra ReLU units.  Every activation
  therefore admits an exact-piece CROWN relaxation; no transcendental
  activation bounds enter the verifier.

* Training is plain supervised regression with a hand-rolled Adam; the
  trained weights are exported as float64 .npz and FROZEN.  Training code
  is *not* part of the trusted computing base - only the exported weights
  and the verifier are.

* The policy supports a witness-margin fine-tuning stage: with V, Q frozen,
  maximize  hinge( m_target - [ min_k Q(x, pi(x), d_k) - gamma V(x) ] )
  by backpropagating through Q's input-gradient w.r.t. u.  This is training
  pressure for non-vacuity (T6 composition-margin rationale); it is never a
  soundness assumption.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
class MLP:
    """Sequential affine/ReLU network; last layer affine (no activation)."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            s = np.sqrt(2.0 / n_in)
            self.W.append(rng.normal(0.0, s, size=(n_in, n_out)).astype(np.float64))
            self.b.append(np.zeros(n_out, dtype=np.float64))

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray, cache: bool = False):
        h = x
        zs, hs = [], [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(z, 0.0)
            else:
                h = z
            if cache:
                zs.append(z)
                hs.append(h)
        return (h, zs, hs) if cache else h

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ------------------------------------------------------------------ #
    def backward(self, zs, hs, dout):
        """Gradients of sum(dout * output) w.r.t. params and the input."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ delta
            gb[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (zs[i - 1] > 0)
        return gW, gb, delta  # delta = grad w.r.t. network input

    def input_grad(self, x: np.ndarray, dout: np.ndarray) -> np.ndarray:
        _, zs, hs = self.forward(x, cache=True)
        _, _, gx = self.backward(zs, hs, dout)
        return gx

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        arrs = {}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            arrs[f"W{i}"] = W
            arrs[f"b{i}"] = b
        np.savez(path, n_layers=len(self.W), **arrs)

    @staticmethod
    def load(path: str) -> "MLP":
        z = np.load(path)
        n = int(z["n_layers"])
        net = MLP.__new__(MLP)
        net.W = [z[f"W{i}"].astype(np.float64) for i in range(n)]
        net.b = [z[f"b{i}"].astype(np.float64) for i in range(n)]
        return net


def policy_forward(pi: MLP, x: np.ndarray, omega_max: float) -> np.ndarray:
    """Deployed policy: hardtanh clamp of the MLP head (exactly PWL)."""
    y = pi(x)
    return np.clip(y, -omega_max, omega_max)


# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, net: MLP, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, *betas, eps
        self.mW = [np.zeros_like(W) for W in net.W]
        self.vW = [np.zeros_like(W) for W in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]
        self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        corr1 = 1 - b1 ** self.t
        corr2 = 1 - b2 ** self.t
        for i in range(len(self.net.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.net.W[i] -= self.lr * (self.mW[i] / corr1) / (
                np.sqrt(self.vW[i] / corr2) + self.eps)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            self.net.b[i] -= self.lr * (self.mb[i] / corr1) / (
                np.sqrt(self.vb[i] / corr2) + self.eps)


# --------------------------------------------------------------------------- #
def train_regression(net: MLP, X: np.ndarray, Y: np.ndarray,
                     epochs: int, batch: int, lr: float, seed: int = 0,
                     tag: str = "net", verbose: bool = True) -> float:
    """Plain MSE regression with Adam.  Returns final epoch MSE."""
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr=lr)
    n = len(X)
    Y = Y.reshape(n, -1)
    mse = np.inf
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], Y[idx]
            out, zs, hs = net.forward(xb, cache=True)
            err = out - yb
            tot += float(np.sum(err ** 2)); cnt += len(idx)
            gW, gb, _ = net.backward(zs, hs, 2.0 * err / len(idx))
            opt.step(gW, gb)
        mse = tot / cnt
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [{tag}] epoch {ep:3d}  mse = {mse:.5f}")
    return mse


def finetune_witness_margin(pi: MLP, q: MLP, v: MLP,
                            X: np.ndarray, d_grid: np.ndarray,
                            gamma: float, omega_max: float,
                            m_target: float, epochs: int, batch: int,
                            lr: float, seed: int = 0,
                            label_anchor: np.ndarray | None = None,
                            anchor_w: float = 0.05,
                            verbose: bool = True) -> None:
    """Fine-tune pi (V, Q frozen) to maximize the verified-composition margin

        m(x) = min_k Q(x, pi(x), d_k) - gamma * V(x)

    via the hinge loss  E[ relu(m_target - m(x)) ]  (+ small anchor to the
    oracle labels to prevent drift).  Gradients flow through dQ/du only.
    """
    n = len(X)
    if n == 0:
        if verbose:
            print("  [pi-margin] skipped: empty training set (no states with V* >= 0)")
        return
    rng = np.random.default_rng(seed)
    opt = Adam(pi, lr=lr)
    vx = v(X).ravel()
    for ep in range(epochs):
        order = rng.permutation(n)
        tot_hinge, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            B = len(idx)
            y, zs, hs = pi.forward(xb, cache=True)        # pre-clip head
            y = y.ravel()
            u = np.clip(y, -omega_max, omega_max)
            # evaluate Q at all probe disturbances, take the min
            qvals = np.empty((B, len(d_grid)))
            for k, dk in enumerate(d_grid):
                z = np.concatenate([xb, u[:, None],
                                    np.full((B, 1), dk)], axis=1)
                qvals[:, k] = q(z).ravel()
            kmin = np.argmin(qvals, axis=1)
            m = qvals[np.arange(B), kmin] - gamma * vx[idx]
            active = m < m_target
            tot_hinge += float(np.sum(np.maximum(m_target - m, 0.0))); cnt += B
            # dL/du = -1{active} * dQ/du at (x, u, d_kmin); clamp gradient
            dmin = d_grid[kmin]
            z = np.concatenate([xb, u[:, None], dmin[:, None]], axis=1)
            gz = q.input_grad(z, np.ones((B, 1)))
            dq_du = gz[:, 3]                               # u is input dim 3
            dL_du = np.where(active, -dq_du, 0.0) / B
            # through the clamp: zero where saturated against the push direction
            sat_hi = (y >= omega_max) & (dL_du < 0)
            sat_lo = (y <= -omega_max) & (dL_du > 0)
            dL_dy = np.where(sat_hi | sat_lo, 0.0, dL_du)
            if label_anchor is not None:
                dL_dy = dL_dy + anchor_w * 2.0 * (y - label_anchor[idx]) / B
            gW, gb, _ = pi.backward(zs, hs, dL_dy[:, None])
            opt.step(gW, gb)
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [pi-margin] epoch {ep:3d}  hinge = {tot_hinge / max(cnt, 1):.5f}")

