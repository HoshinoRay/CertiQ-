"""Compile the certificate's composite predicates into single sequential
ReLU networks, so that ONE generic sound bound routine (verify/bounds.py)
discharges every condition.

The deployed artifact is all piecewise-linear, so the compositions below are
*exact* network rewrites (verified by the equality test in tests/):

  h3(x, d) = Q_theta(x, clip(pi(x)), d) - gamma * V_theta(x) - eps     (C3)

Building blocks used by the compiler:

  * identity carry through a ReLU stage:  t = ReLU(t) - ReLU(-t)
    (a +/- unit pair per carried scalar);
  * hardtanh clamp via two ReLU units:
    clip(y, -w, w) = ReLU(y + w) - ReLU(y - w) - w.

The compiled network is then bounded jointly over (x, d) by CROWN, which
preserves the x <-> u correlation through the policy (much tighter than the
decoupled bound that intervals pi(x) first).  A decoupled fallback is also
provided for cross-checking; both are sound.
"""
from __future__ import annotations

import numpy as np

from ..nets.mlp import MLP
from .bounds import SeqNet


# --------------------------------------------------------------------------- #
# Low-level block-matrix builder
# --------------------------------------------------------------------------- #
class _StageBuilder:
    """Accumulates output blocks of one affine stage.

    Each output block is given by (rows -> cols) coefficient assignments on
    the previous stage's representation, plus a bias.
    """

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.cols: list[np.ndarray] = []     # each (n_in,) column
        self.bias: list[float] = []

    def add_block(self, W: np.ndarray, b: np.ndarray,
                  row_index: list[int] | np.ndarray) -> slice:
        """Add columns z = h[row_index] @ W + b.  Returns the output slice."""
        start = len(self.cols)
        n_out = W.shape[1]
        for j in range(n_out):
            col = np.zeros(self.n_in)
            col[np.asarray(row_index)] = W[:, j]
            self.cols.append(col)
            self.bias.append(float(b[j]))
        return slice(start, start + n_out)

    def add_linear(self, coeffs: dict[int, float], bias: float) -> int:
        """Add a single unit  z = sum coeffs[i] * h[i] + bias."""
        col = np.zeros(self.n_in)
        for i, c in coeffs.items():
            col[i] = c
        self.cols.append(col)
        self.bias.append(bias)
        return len(self.cols) - 1

    def add_carry(self, pair_expr: list[tuple[int, float]]) -> tuple[int, int]:
        """Carry the scalar  s = sum c_i h_i  through the next ReLU as a
        +/- pair (s+, s-) with  s = ReLU(s) - ReLU(-s)."""
        ip = self.add_linear({i: c for i, c in pair_expr}, 0.0)
        im = self.add_linear({i: -c for i, c in pair_expr}, 0.0)
        return ip, im

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        W = np.stack(self.cols, axis=1) if self.cols else np.zeros((self.n_in, 0))
        return W, np.array(self.bias)


def _carry_pairs(builder: _StageBuilder,
                 pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Re-carry existing +/- pairs through the next stage."""
    out = []
    for ip, im in pairs:
        out.append(builder.add_carry([(ip, 1.0), (im, -1.0)]))
    return out


# --------------------------------------------------------------------------- #
def compile_h3(pi: MLP, q: MLP, v: MLP, gamma: float, eps: float,
               omega_max: float) -> SeqNet:
    """Compile  h3(x,d) = Q(x, clip(pi(x), +-omega), d) - gamma V(x) - eps
    into a single SeqNet with input z = (px, py, psi, d) in R^4.

    Requires pi and v to have the same hidden depth (the config default);
    Q's depth is arbitrary.
    """
    n_pi = len(pi.W) - 1   # hidden layers of pi
    n_v = len(v.W) - 1
    n_q = len(q.W) - 1
    assert n_pi == n_v, "compiler assumes equal hidden depth for pi and V"

    Ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []

    # ----- stage 1: first hidden layers of pi and V; carry x and d --------- #
    sb = _StageBuilder(4)
    pi_sl = sb.add_block(pi.W[0], pi.b[0], [0, 1, 2])
    v_sl = sb.add_block(v.W[0], v.b[0], [0, 1, 2])
    x_pairs = [sb.add_carry([(j, 1.0)]) for j in range(3)]
    d_pair = [sb.add_carry([(3, 1.0)])]
    W, b = sb.finish()
    Ws.append(W); bs.append(b)
    cur = W.shape[1]

    # ----- stages 2..n_pi: remaining hidden layers of pi and V ------------- #
    for layer in range(1, n_pi):
        sb = _StageBuilder(cur)
        pi_sl_new = sb.add_block(pi.W[layer], pi.b[layer],
                                 list(range(pi_sl.start, pi_sl.stop)))
        v_sl_new = sb.add_block(v.W[layer], v.b[layer],
                                list(range(v_sl.start, v_sl.stop)))
        x_pairs = _carry_pairs(sb, x_pairs)
        d_pair = _carry_pairs(sb, d_pair)
        W, b = sb.finish()
        Ws.append(W); bs.append(b)
        cur = W.shape[1]
        pi_sl, v_sl = pi_sl_new, v_sl_new

    # ----- stage n_pi+1: pi head + clamp pre-units; V head as +/- pair ----- #
    sb = _StageBuilder(cur)
    wpi = pi.W[-1][:, 0]
    bpi = float(pi.b[-1][0])
    pi_rows = list(range(pi_sl.start, pi_sl.stop))
    clipA = sb.add_linear({r: wpi[k] for k, r in enumerate(pi_rows)},
                          bpi + omega_max)               # y + omega
    clipB = sb.add_linear({r: wpi[k] for k, r in enumerate(pi_rows)},
                          bpi - omega_max)               # y - omega
    wv = v.W[-1][:, 0]
    bv = float(v.b[-1][0])
    v_rows = list(range(v_sl.start, v_sl.stop))
    v_pair = [sb.add_carry([(r, wv[k]) for k, r in enumerate(v_rows)])]
    # bias of V output handled below (carry pair has zero bias by design):
    # represent V = (v+ - v-) + bv  ->  fold bv into the final affine stage.
    x_pairs = _carry_pairs(sb, x_pairs)
    d_pair = _carry_pairs(sb, d_pair)
    W, b = sb.finish()
    Ws.append(W); bs.append(b)
    cur = W.shape[1]

    # ----- Q hidden stages -------------------------------------------------- #
    # u = ReLU(clipA) - ReLU(clipB) - omega
    qW0, qb0 = q.W[0], q.b[0]      # input order (px, py, psi, u, d)
    sb = _StageBuilder(cur)
    start = len(sb.cols)
    for j in range(qW0.shape[1]):
        coeffs: dict[int, float] = {}
        for k in range(3):                       # x via +/- pairs
            ip, im = x_pairs[k]
            coeffs[ip] = coeffs.get(ip, 0.0) + qW0[k, j]
            coeffs[im] = coeffs.get(im, 0.0) - qW0[k, j]
        coeffs[clipA] = coeffs.get(clipA, 0.0) + qW0[3, j]
        coeffs[clipB] = coeffs.get(clipB, 0.0) - qW0[3, j]
        ip, im = d_pair[0]
        coeffs[ip] = coeffs.get(ip, 0.0) + qW0[4, j]
        coeffs[im] = coeffs.get(im, 0.0) - qW0[4, j]
        sb.add_linear(coeffs, float(qb0[j]) - omega_max * qW0[3, j])
    q_sl = slice(start, start + qW0.shape[1])
    v_pair = _carry_pairs(sb, v_pair)
    W, b = sb.finish()
    Ws.append(W); bs.append(b)
    cur = W.shape[1]

    for layer in range(1, n_q):
        sb = _StageBuilder(cur)
        q_sl_new = sb.add_block(q.W[layer], q.b[layer],
                                list(range(q_sl.start, q_sl.stop)))
        v_pair = _carry_pairs(sb, v_pair)
        W, b = sb.finish()
        Ws.append(W); bs.append(b)
        cur = W.shape[1]
        q_sl = q_sl_new

    # ----- final affine: h3 = Q_out - gamma * (V_out + bv) - eps ----------- #
    sb = _StageBuilder(cur)
    wq = q.W[-1][:, 0]
    bq = float(q.b[-1][0])
    coeffs = {r: wq[k] for k, r in enumerate(range(q_sl.start, q_sl.stop))}
    vp, vm = v_pair[0]
    coeffs[vp] = coeffs.get(vp, 0.0) - gamma
    coeffs[vm] = coeffs.get(vm, 0.0) + gamma
    sb.add_linear(coeffs, bq - gamma * bv - eps)
    W, b = sb.finish()
    Ws.append(W); bs.append(b)

    return SeqNet(Ws, bs)


# --------------------------------------------------------------------------- #
def compile_policy(pi: MLP, omega_max: float) -> SeqNet:
    """u(x) = clip(pi(x), -omega, omega) as a SeqNet (for decoupled bounds)."""
    Ws = [W.copy() for W in pi.W[:-1]]
    bs = [b.copy() for b in pi.b[:-1]]
    w_head, b_head = pi.W[-1], pi.b[-1]
    # head -> (y + w, y - w), ReLU, then [1, -1] - w
    Ws.append(np.concatenate([w_head, w_head], axis=1))
    bs.append(np.array([float(b_head[0]) + omega_max,
                        float(b_head[0]) - omega_max]))
    Ws.append(np.array([[1.0], [-1.0]]))
    bs.append(np.array([-omega_max]))
    return SeqNet(Ws, bs)


def as_seqnet(mlp: MLP) -> SeqNet:
    return SeqNet.from_mlp(mlp)
