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
# Differentiable interval bound propagation (for verifier-in-the-loop training)
# --------------------------------------------------------------------------- #
def ibp_forward(net: "MLP", lo: np.ndarray, hi: np.ndarray):
    """IBP output bounds of `net` over input boxes [lo, hi] (B, n_in).

    Returns (lb_out, ub_out, cache) where lb_out, ub_out are (B, n_out) and
    `cache` lets ``ibp_backward`` differentiate a scalar loss in those bounds
    w.r.t. the network weights.  Same arithmetic as verify/bounds.ibp_*,
    duplicated here only to expose a backward pass (the verifier copy stays the
    trusted one)."""
    c = 0.5 * (lo + hi)
    r = 0.5 * (hi - lo)
    cs, rs, m_lo, m_hi = [c], [r], [], []
    L = len(net.W)
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        c_pre = c @ W + b
        r_pre = r @ np.abs(W)
        if i < L - 1:
            lo_pre, hi_pre = c_pre - r_pre, c_pre + r_pre
            mlo, mhi = (lo_pre > 0.0), (hi_pre > 0.0)
            lo_post, hi_post = np.maximum(lo_pre, 0.0), np.maximum(hi_pre, 0.0)
            c, r = 0.5 * (lo_post + hi_post), 0.5 * (hi_post - lo_post)
            cs.append(c); rs.append(r); m_lo.append(mlo); m_hi.append(mhi)
        else:
            lb_out, ub_out = c_pre - r_pre, c_pre + r_pre
    return lb_out, ub_out, (cs, rs, m_lo, m_hi)


def ibp_backward(net: "MLP", cache, d_lb: np.ndarray, d_ub: np.ndarray):
    """Gradients of sum(d_lb*lb_out + d_ub*ub_out) w.r.t. net weights/biases.

    Mirrors ``ibp_forward``; returns (gW, gb) lists.  Hand-checked against finite
    differences (T10)."""
    cs, rs, m_lo, m_hi = cache
    gW = [np.zeros_like(W) for W in net.W]
    gb = [np.zeros_like(b) for b in net.b]
    L = len(net.W)
    # output layer: lb = c_pre - r_pre, ub = c_pre + r_pre
    d_c = d_lb + d_ub
    d_r = d_ub - d_lb
    for i in range(L - 1, -1, -1):
        c_in, r_in, W = cs[i], rs[i], net.W[i]
        gW[i] = c_in.T @ d_c + np.sign(W) * (r_in.T @ d_r)
        gb[i] = d_c.sum(axis=0)
        if i == 0:
            break
        d_c_in = d_c @ W.T
        d_r_in = d_r @ np.abs(W).T
        # back through the ReLU of layer i-1: c,r are post-relu of that layer
        d_lo_post = 0.5 * d_c_in - 0.5 * d_r_in
        d_hi_post = 0.5 * d_c_in + 0.5 * d_r_in
        d_lo_pre = d_lo_post * m_lo[i - 1]
        d_hi_pre = d_hi_post * m_hi[i - 1]
        d_c = d_lo_pre + d_hi_pre
        d_r = -d_lo_pre + d_hi_pre
    return gW, gb


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


def train_v_cbf(net: MLP, X: np.ndarray, Y: np.ndarray, g_vals: np.ndarray,
                model, u_menu: np.ndarray, d_grid: np.ndarray, gamma: float,
                epochs: int, batch: int, lr: float,
                floor_w: float, floor_margin: float,
                dec_w: float, dec_margin: float, fit_w: float = 1.0,
                seed: int = 0, tag: str = "V", verbose: bool = True) -> float:
    """Shape V_theta into a certifiable CBF (training pressures only; the
    verifier still checks the frozen net):

        loss = mean (V(x) - Y)^2                                    # track teacher
             + floor_w * mean relu(V(x) - g(x) + floor_margin)      # C1: {V>=0} in K
             + dec_w   * mean relu(gamma V(x) + dec_margin - B(x))  # decrease margin

    where B(x) = max_{u in menu} min_{d in grid} V(f(x,u,d)) is the best robust
    one-step successor value.  The decrease hinge pushes B(x) >= gamma V(x) +
    dec_margin, i.e. it makes the witnessed one-step decrease clear a positive
    margin.  No Lipschitz/weight-decay term -- that would change the object.
    Returns final epoch MSE-to-Y."""
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr=lr)
    n = len(X)
    Y = Y.reshape(n, 1)
    gv = np.asarray(g_vals, float).reshape(n, 1)
    U, D = np.asarray(u_menu, float), np.asarray(d_grid, float)
    mse = np.inf
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, fviol, dviol, cnt = 0.0, 0.0, 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb, gb = X[idx], Y[idx], gv[idx]
            B = len(idx)
            out, zs, hs = net.forward(xb, cache=True)        # V(x)
            # B(x) = max_u min_d V(f(x,u,d)); track the (u*,d*) successor.
            # Vectorized: one forward over all B*nu*nd successors.
            nu, nd = len(U), len(D)
            Xrep = np.repeat(xb, nu * nd, axis=0)
            uu = np.tile(np.repeat(U, nd), B)
            dd = np.tile(np.tile(D, nu), B)
            S = model.step(Xrep, uu, dd)
            VS = net.forward(S).reshape(B, nu, nd)
            mind = VS.min(axis=2)                            # min over d
            amax = mind.argmax(axis=1)                       # argmax over u
            best = mind[np.arange(B), amax]
            amin = VS[np.arange(B), amax, :].argmin(axis=1)
            flat = np.arange(B) * (nu * nd) + amax * nd + amin
            s_star = S[flat]                                 # f(x,u*,d*)
            err = out - yb
            over_f = out.ravel() - gb.ravel() + floor_margin
            act_f = (over_f > 0.0).astype(np.float64)
            over_d = gamma * out.ravel() + dec_margin - best
            act_d = (over_d > 0.0).astype(np.float64)
            tot += float(np.sum(err ** 2)); cnt += B
            fviol += float(np.sum(np.maximum(over_f, 0.0)))
            dviol += float(np.sum(np.maximum(over_d, 0.0)))
            # grad at x: (anchored) teacher fit + floor + decrease-through-gamma*V(x)
            dout_x = (fit_w * 2.0 * err.ravel() + floor_w * act_f
                      + dec_w * act_d * gamma) / B
            gW, gB, _ = net.backward(zs, hs, dout_x[:, None])
            # grad at s* : decrease-through -V(f(x,u*,d*))
            out_s, zs_s, hs_s = net.forward(s_star, cache=True)
            dout_s = (-dec_w * act_d) / B
            gW_s, gB_s, _ = net.backward(zs_s, hs_s, dout_s[:, None])
            for i in range(len(gW)):
                gW[i] = gW[i] + gW_s[i]
                gB[i] = gB[i] + gB_s[i]
            opt.step(gW, gB)
        mse = tot / cnt
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [{tag}] epoch {ep:3d}  mse={mse:.5f}  "
                  f"floor_viol={fviol / cnt:.5f}  dec_viol={dviol / cnt:.5f}")
    return mse


def train_q_oneside(net: MLP, v: MLP, X: np.ndarray, model,
                    epochs: int, batch: int, lr: float,
                    c4_w: float, c4_margin: float,
                    seed: int = 0, tag: str = "Q", verbose: bool = True) -> float:
    """Train Q to TRACK the frozen learned successor value V_theta(f(x,u,d))
    from BELOW (Bellman-consistent, C4-aligned).  On samples X=(px,py,psi,u,d):

        Vf   = V_theta(f(x,u,d))        (frozen, the true successor value)
        loss = mean (Q - Vf)^2  +  c4_w * mean relu(Q - Vf + c4_margin).

    The squared term keeps Q ~= Vf (so the gate min_d Q >= gamma V stays
    meaningful for C3); the hinge biases Q just BELOW Vf, giving C4
    (min_d Q <= min_d V(f)) since Q(x,u,d) <= V(f(x,u,d)) for every d.  No oracle
    Q labels are used -- Q is purely a one-sided copy of V_theta o f, which is
    the deployment structure (Q primary / V = max-min-Q reverses the distillation
    direction but keeps the same consistency).  Returns final MSE-to-Vf."""
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr=lr)
    n = len(X)
    nxt = model.step(X[:, :3], X[:, 3], X[:, 4])    # f(x,u,d); x,u,d fixed
    Vf = v(nxt).reshape(n, 1)                       # frozen successor value
    mse = np.inf
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, viol, cnt = 0.0, 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, vfb = X[idx], Vf[idx]
            out, zs, hs = net.forward(xb, cache=True)
            B = len(idx)
            err = out - vfb
            over = out - vfb + c4_margin
            active = (over > 0.0).astype(np.float64)
            tot += float(np.sum(err ** 2)); cnt += B
            viol += float(np.sum(np.maximum(over, 0.0)))
            dout = (2.0 * err + c4_w * active) / B
            gW, gB, _ = net.backward(zs, hs, dout)
            opt.step(gW, gB)
        mse = tot / cnt
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [{tag}] epoch {ep:3d}  mse_to_Vf={mse:.5f}  "
                  f"c4_viol={viol / cnt:.5f}")
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

