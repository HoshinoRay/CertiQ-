"""Deployed certified Q-CBF runtime filter.

Consistency with the certificate (the key soundness handshake):

  * a candidate action u is applied ONLY if a SOUND lower bound of
        min_{d in D} Q_theta(x, u, d)
    is  >= gamma * V_theta(x).  Soundness of the bound implies the true
    antecedent  a(x, u) >= 0,  hence the (cell, u-cell) containing (x, u)
    was NOT skipped by the verifier, hence its robust successors were
    verified to stay in the accepted set (C2).

  * if no candidate passes, the filter applies the certified fallback
        u = clip(pi_theta(x), +-omega_max),
    whose true antecedent margin is >= eps > 0 by C3 on every accepted
    cell, so its u-cell is likewise never skipped and its successors are
    likewise covered by C2.  The filter therefore NEVER gets stuck and the
    closed loop NEVER leaves the certified set (Theorem A).

The filter is fully batched so the audit can roll out hundreds of episodes
in lockstep.
"""
from __future__ import annotations

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.nets.mlp import MLP, policy_forward
from qcbf.verify.bounds import SeqNet, crown_bounds


# --------------------------------------------------------------------------- #
class CertifiedFilter:
    def __init__(self, lattice: CellLattice, accepted: np.ndarray,
                 v: MLP, q: MLP, pi: MLP, q_net: SeqNet,
                 gamma: float, omega_max: float, d_max: float,
                 n_candidates: int = 21, tighten: bool = False):
        self.lat = lattice
        self.accepted = accepted
        self.v, self.q, self.pi = v, q, pi
        self.q_net = q_net
        self.gamma = gamma
        self.om = omega_max
        self.d_max = d_max
        self.cand_u = np.linspace(-omega_max, omega_max, n_candidates)
        self.tighten = tighten          # any sound mode is consistent

    # ------------------------------------------------------------------ #
    def is_certified(self, x: np.ndarray) -> np.ndarray:
        idx = self.lat.cell_index(np.atleast_2d(x))
        ok = idx >= 0
        out = np.zeros(len(idx), dtype=bool)
        out[ok] = self.accepted[idx[ok]]
        return out

    # ------------------------------------------------------------------ #
    def feasibility_lb(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Sound lower bound of min_d Q(x, u, d) over D (batched)."""
        B = len(x)
        lo = np.column_stack([x, u, np.full(B, -self.d_max)])
        hi = np.column_stack([x, u, np.full(B, +self.d_max)])
        lb, _ = crown_bounds(self.q_net, lo, hi, self.tighten)
        return lb[:, 0]

    # ------------------------------------------------------------------ #
    def batch_select(self, X: np.ndarray, u_task: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose actions for a batch of states.

        Returns (u (B,), used_fallback (B,) bool, best_margin (B,)).
        Among feasible candidates, picks the one closest to u_task
        (minimum-intervention principle); fallback if none is feasible.
        """
        B, nc = len(X), len(self.cand_u)
        thresh = self.gamma * self.v(X).ravel()
        Xr = np.repeat(X, nc, axis=0)
        Ur = np.tile(self.cand_u, B)
        lb = self.feasibility_lb(Xr, Ur).reshape(B, nc)
        feasible = lb >= thresh[:, None]

        dist = np.abs(self.cand_u[None, :] - u_task[:, None])
        dist = np.where(feasible, dist, np.inf)
        pick = np.argmin(dist, axis=1)
        any_feas = feasible.any(axis=1)

        u = self.cand_u[pick]
        fb = policy_forward(self.pi, X, self.om).ravel()
        u = np.where(any_feas, u, fb)
        margin = lb.max(axis=1) - thresh
        return u, ~any_feas, margin
