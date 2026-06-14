"""Deployed certified Q-CBF runtime filter.

The verifier proves the frozen artifact on the continuous superlevel set
``{x : V_theta(x) >= 0}`` over the Dubins certification domain.  The exported
``accepted`` cell mask is an inner-volume lower bound and a convenient source
of audit initial states; it is not the mathematical invariant set.

Runtime actions come only from the same finite menu used by the verifier.  A
menu action is used when a sound lower bound proves the robust Q-gate

    min_d Q_theta(x,u,d) >= gamma_deploy * V_theta(x).

If no menu action passes, the deployed witness ``u = pi_theta(x)`` is used; C3
certifies its gate non-vacuity and C4 certifies that Q_theta does not overstate
the successor value.  The gate (lower bound here) and the verifier's C4 use the
SAME gamma_deploy, which is what makes the certificate apply to this loop.
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
                 gamma_deploy: float, omega_max: float, d_max: float,
                 n_menu: int, tighten: bool = False):
        self.lat = lattice
        self.accepted = accepted
        self.v, self.q, self.pi = v, q, pi
        self.q_net = q_net
        self.gamma_deploy = gamma_deploy
        self.om = omega_max
        self.d_max = d_max
        self.cand_u = (np.array([0.0]) if n_menu <= 1
                       else np.linspace(-omega_max, omega_max, n_menu))
        self.tighten = tighten          # any sound mode is consistent

    # ------------------------------------------------------------------ #
    def is_certified(self, x: np.ndarray) -> np.ndarray:
        """Membership in the certified continuous domain: grid cell plus V>=0."""
        X = np.atleast_2d(x)
        idx = self.lat.cell_index(X)
        return (idx >= 0) & (self.v(X).ravel() >= 0.0)

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
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Choose actions for a batch of states.

        Returns ``(u, used_fallback, best_margin, certified)``, each ``(B,)``.
        Among feasible candidates picks the one closest to ``u_task``
        (minimum-intervention); fallback ``clip(pi(x))`` if none is feasible.

        IMPORTANT precondition: the certificate (and hence every safety claim)
        only covers states in the certified domain ``{V_theta >= 0}`` of a
        PASSED certificate.  ``certified`` reports membership in that domain; an
        action is still returned for ``~certified`` states, but it carries NO
        guarantee and must not be treated as a global safe controller.
        """
        B, nc = len(X), len(self.cand_u)
        thresh = self.gamma_deploy * self.v(X).ravel()
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
        return u, ~any_feas, margin, self.is_certified(X)
