"""Sound bound propagation for sequential affine/ReLU networks (batched).

This module is the core of the certificate's trusted computing base.  It is
deliberately small, NumPy-only and float64 throughout, so it can be audited
line by line.  Two methods are provided:

  * IBP   - interval bound propagation (fast, loose),
  * CROWN - backward linear relaxation with per-neuron adaptive ReLU slopes
            (Zhang et al., 2018), optionally with CROWN-tightened
            intermediate pre-activation bounds.

Soundness contract (covered by randomized tests in tests/):

    for every box [lb, ub] and every x in the box:
        lo(box) - tol  <=  net(x)  <=  hi(box) + tol

Looseness only ever shrinks the certified set; it can never certify a false
inequality (theory_core Prop. T2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class SeqNet:
    """z_{i} = h_{i-1} @ W_i + b_i, ReLU between layers, last layer affine."""
    W: list[np.ndarray]
    b: list[np.ndarray]

    @property
    def in_dim(self) -> int:
        return self.W[0].shape[0]

    @property
    def out_dim(self) -> int:
        return self.W[-1].shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return h

    @staticmethod
    def from_mlp(mlp) -> "SeqNet":
        return SeqNet([W.astype(np.float64) for W in mlp.W],
                      [b.astype(np.float64) for b in mlp.b])


# --------------------------------------------------------------------------- #
def ibp_preact_bounds(net: SeqNet, lb: np.ndarray, ub: np.ndarray):
    """Pre-activation interval bounds for every layer (incl. the output).

    lb, ub : (B, n_in).  Returns list of (l_i, u_i), each (B, n_i).
    """
    lo, hi = lb.astype(np.float64), ub.astype(np.float64)
    out = []
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        c = 0.5 * (lo + hi) @ W + b
        r = 0.5 * (hi - lo) @ np.abs(W)
        l, u = c - r, c + r
        out.append((l, u))
        if i < len(net.W) - 1:
            lo, hi = np.maximum(l, 0.0), np.maximum(u, 0.0)
    return out


def _relu_relaxation(l: np.ndarray, u: np.ndarray):
    """Per-neuron sound linear relaxation of ReLU on [l, u].

    Returns (s_lo, s_up, t_up): lower line  y >= s_lo * z  (intercept 0),
    upper line  y <= s_up * z + t_up.  Adaptive lower slope: 1 if u >= -l
    else 0 (standard CROWN heuristic; any slope in [0,1] is sound).
    """
    pos = l >= 0.0
    neg = u <= 0.0
    unstable = ~(pos | neg)
    denom = np.where(unstable, u - l, 1.0)
    s_up = np.where(pos, 1.0, np.where(neg, 0.0, u / denom))
    t_up = np.where(unstable, -l * u / denom, 0.0)
    s_lo = np.where(pos, 1.0, np.where(neg, 0.0,
                    (u >= -l).astype(np.float64)))
    return s_lo, s_up, t_up


def _crown_backward_affine(net: SeqNet, upto: int,
                           spec: np.ndarray,
                           preact: list[tuple[np.ndarray, np.ndarray]]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Sound affine LOWER functional of  spec @ z_{upto}  in the network INPUT.

    Returns (A, bias) with A : (B, m, n_in), bias : (B, m) such that, for every
    x in the box used to build `preact`,
        spec @ z_{upto}(x)  >=  einsum("bmj,bj->bm", A, x) + bias .
    This is the back-substituted CROWN linear bound *before* it is contracted
    against the input box -- exposing it lets a caller substitute a (correlated,
    nonlinear) successor map x+ = phi(cell) into the SAME functional and minimise
    A @ phi over the true successor, which is >= the box-contracted scalar
    (direct composition; no intermediate outward-rounded successor box).
    """
    A = np.ascontiguousarray(spec)
    bias = np.einsum("bmj,j->bm", A, net.b[upto])
    A = A @ net.W[upto].T                       # coefficients on h_{upto-1}
    for i in range(upto - 1, -1, -1):
        l, u = preact[i]
        s_lo, s_up, t_up = _relu_relaxation(l, u)
        Apos = np.maximum(A, 0.0)
        Aneg = np.minimum(A, 0.0)
        # lower-bounding: positive coeffs take the lower line, negative the upper
        bias = bias + np.einsum("bmj,bj->bm", Aneg, t_up)
        A = Apos * s_lo[:, None, :] + Aneg * s_up[:, None, :]
        bias = bias + np.einsum("bmj,j->bm", A, net.b[i])
        A = A @ net.W[i].T                      # coefficients on h_{i-1}
    return A, bias                              # functional on the input layer


def _crown_backward_lower(net: SeqNet, upto: int,
                          spec: np.ndarray,
                          preact: list[tuple[np.ndarray, np.ndarray]],
                          lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Sound lower bound of  spec @ z_{upto}  over the input box.

    spec : (B, m, n_upto) linear specification on the pre-activation of layer
    `upto` (0-based).  Uses ReLU relaxations built from `preact` for layers
    0..upto-1.  Returns (B, m).
    """
    A, bias = _crown_backward_affine(net, upto, spec, preact)
    Apos = np.maximum(A, 0.0)
    Aneg = np.minimum(A, 0.0)
    return (np.einsum("bmj,bj->bm", Apos, lb)
            + np.einsum("bmj,bj->bm", Aneg, ub) + bias)


def crown_preact_bounds(net: SeqNet, lb: np.ndarray, ub: np.ndarray):
    """Pre-activation bounds via progressive backward CROWN, intersected with
    IBP (both are sound, so the elementwise tighter bound is sound)."""
    B = lb.shape[0]
    ibp = ibp_preact_bounds(net, lb, ub)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(net.W)):
        n_i = net.W[i].shape[1]
        eye = np.eye(n_i)
        spec = np.broadcast_to(eye, (B, n_i, n_i))
        lo = _crown_backward_lower(net, i, spec, out, lb, ub)
        hi = -_crown_backward_lower(net, i, -spec, out, lb, ub)
        l = np.maximum(lo, ibp[i][0])
        u = np.minimum(hi, ibp[i][1])
        # numerical guard: keep l <= u
        l, u = np.minimum(l, u), np.maximum(l, u)
        out.append((l, u))
    return out


def crown_bounds(net: SeqNet, lb: np.ndarray, ub: np.ndarray,
                 tighten_intermediate: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sound (lo, hi) bounds of the network output over input boxes.

    lb, ub : (B, n_in)  ->  lo, hi : (B, n_out)
    """
    lb = np.asarray(lb, np.float64)
    ub = np.asarray(ub, np.float64)
    if tighten_intermediate:
        preact = crown_preact_bounds(net, lb, ub)
    else:
        preact = ibp_preact_bounds(net, lb, ub)
    B = lb.shape[0]
    m = net.out_dim
    L = len(net.W) - 1
    eye = np.eye(m)
    spec = np.broadcast_to(eye, (B, m, m))
    lo = _crown_backward_lower(net, L, spec, preact[:L], lb, ub)
    hi = -_crown_backward_lower(net, L, -spec, preact[:L], lb, ub)
    return lo, hi


def crown_lower_affine(net: SeqNet, lb: np.ndarray, ub: np.ndarray,
                       tighten_intermediate: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sound affine LOWER functional of the network output over the input box.

    Returns (A, beta) : (B, m, n_in), (B, m) such that for every x in [lb, ub]
        net(x)  >=  einsum("bmj,bj->bm", A, x) + beta .
    Contracting A against the box reproduces ``crown_bounds(...)[0]`` exactly;
    substituting a nonlinear successor map x+ = phi(cell) (with phi(cell) subset
    of [lb, ub]) and minimising A @ phi + beta over the cell yields a sound,
    *tighter-or-equal* lower bound of net(phi(.)) -- this is the direct-
    composition primitive (P1) that avoids the outward-rounded successor box.
    """
    lb = np.asarray(lb, np.float64)
    ub = np.asarray(ub, np.float64)
    preact = (crown_preact_bounds(net, lb, ub) if tighten_intermediate
              else ibp_preact_bounds(net, lb, ub))
    B = lb.shape[0]
    m = net.out_dim
    L = len(net.W) - 1
    eye = np.eye(m)
    spec = np.broadcast_to(eye, (B, m, m))
    return _crown_backward_affine(net, L, spec, preact[:L])


def crown_upper_affine(net: SeqNet, lb: np.ndarray, ub: np.ndarray,
                       tighten_intermediate: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sound affine UPPER functional of the network output over the input box.

    Returns (A, beta) : (B, m, n_in), (B, m) such that for every x in [lb, ub]
        net(x)  <=  einsum("bmj,bj->bm", A, x) + beta .
    Mirror of ``crown_lower_affine``: the backward CROWN pass on ``-spec`` yields
    a lower functional of ``-net``, whose negation is an upper functional of
    ``net``.  Contracting A against the box (A_pos @ ub + A_neg @ lb + beta)
    reproduces ``crown_bounds(...)[1]`` exactly.  Used by the relational decrease
    bound to subtract a shared V_theta(x) copy with cancelling linear part.
    """
    lb = np.asarray(lb, np.float64)
    ub = np.asarray(ub, np.float64)
    preact = (crown_preact_bounds(net, lb, ub) if tighten_intermediate
              else ibp_preact_bounds(net, lb, ub))
    B = lb.shape[0]
    m = net.out_dim
    L = len(net.W) - 1
    eye = np.eye(m)
    spec = np.broadcast_to(eye, (B, m, m))
    A_neg, bias_neg = _crown_backward_affine(net, L, -spec, preact[:L])
    return -A_neg, -bias_neg


def crown_bounds_chunked(net: SeqNet, lb: np.ndarray, ub: np.ndarray,
                         tighten_intermediate: bool = True,
                         chunk: int = 512,
                         progress: str | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Memory-bounded wrapper around crown_bounds.

    ``progress`` is an optional label; when given, a live percentage + ETA is
    printed over the chunk loop.  This is pure observability around the
    numeric core (``crown_bounds`` and below) and changes no bound.
    """
    n = len(lb)
    if n == 0:                          # empty cell set -> well-shaped empties
        m = net.out_dim
        return np.zeros((0, m)), np.zeros((0, m))
    los, his = [], []
    pb = None
    if progress is not None:
        from qcbf.util.progress import Progress
        pb = Progress(n, progress)
    for s in range(0, n, chunk):
        lo, hi = crown_bounds(net, lb[s:s + chunk], ub[s:s + chunk],
                              tighten_intermediate)
        los.append(lo)
        his.append(hi)
        if pb is not None:
            pb.update(min(s + chunk, n))
    if pb is not None:
        pb.done()
    return np.concatenate(los), np.concatenate(his)
