"""Grid safety value iteration for the Dubins safety game (teacher only).

This oracle produces *labels*; the certificate is on the frozen networks.  The
deployed one-step CBF inequality the filter and the certificate check is

    V(f(x,u,d)) >= gamma_deploy * V(x).

The teacher value is solved with the **discounted safety backup** (Fisac /
Akametalu), with teacher discount ``gamma`` == gamma_teach in (0, 1):

    V^(0)(x) = g(x),
    F[V](x)  = max_u min_d V(f(x,u,d)),
    V^(k+1)(x) = (1 - gamma) * g(x) + gamma * min(g(x), F[V^(k)](x)).

The discount makes the backup a ``gamma``-contraction (unique fixed point,
geometric convergence) and -- crucially -- the ``(1 - gamma) g`` source term
pins V near g on the safe interior, which is what keeps the multilinear-interp
minimax from draining the avoid value to g_fail everywhere (see DESIGN_REVIEW,
"anti-spec trap").  ``gamma`` near 1 = lighter discount = larger Omega*; we use
~0.92.  The other forms are NOT usable here: undiscounted ``min(g, F)``
(gamma=1) collapses under interpolation, ``min(g, gamma F)`` is identically
<= 0 for gamma<1, and ``min(g, F/gamma)`` diverges.

The teacher is an under-approximation, not an exact CBF: it carries a positive
deployed margin (~ (1 - gamma_deploy) V) on the safe interior but can be
slightly negative on the V=0 boundary shell.  That is fine -- it only seeds the
labels and the reference volume Omega* = {V >= 0}; soundness comes solely from
the post-hoc verifier on (V_theta, Q_theta, pi_theta) at gamma_deploy.

V and Q are learned by two *separate* networks, so the action value is taken to
be the successor value directly (no ``V = max_u min_d Q`` identity is imposed):

    Q_HJ(x,u,d) = V_HJ(f(x,u,d)).

``q_star`` returns exactly Q_HJ = V_HJ o f.

Performance note (this is what makes the paper grid run in minutes on CPU):
for the fixed-speed Dubins car the successor position
    (px + dt v cos psi, py + dt v sin psi)
does not depend on (u, d) at all, and the successor heading is a *uniform*
shift psi + dt (u + d) of the (uniform, periodic) heading grid.  Hence

  * the bilinear interpolation stencil in (px, py) is precomputed ONCE,
  * for each (u, d) pair a sweep only needs a periodic roll + lerp in psi
    followed by a gather with the precomputed stencil.

Out-of-domain successors (in position) receive the failure value g_fail;
they are never silently clamped to the boundary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import DubinsConfig, OracleConfig
from ..dynamics.dubins import DubinsModel, TWO_PI


@dataclass
class OracleGrid:
    px: np.ndarray          # (n_px,)
    py: np.ndarray          # (n_py,)
    psi: np.ndarray         # (n_psi,)  cell-centered, periodic
    u: np.ndarray           # (n_u,)
    d: np.ndarray           # (n_d,)

    @staticmethod
    def build(dyn: DubinsConfig, cfg: OracleConfig) -> "OracleGrid":
        px = np.linspace(dyn.p_lo, dyn.p_hi, cfg.n_px)
        py = np.linspace(dyn.p_lo, dyn.p_hi, cfg.n_py)
        # cell-centered periodic heading grid: spacing 2pi/n, no duplicate at pi
        psi = -np.pi + (np.arange(cfg.n_psi) + 0.5) * (TWO_PI / cfg.n_psi)
        u = np.linspace(-dyn.control_max, dyn.control_max, cfg.n_u)
        d = np.linspace(-dyn.d_max, dyn.d_max, cfg.n_d)
        return OracleGrid(px, py, psi, u, d)


class GridOracle:
    """Dubins CBVF/HJ teacher: V_HJ, Q_HJ samples and greedy witness labels."""

    def __init__(self, dyn_cfg, cfg: OracleConfig, gamma: float, model=None):
        # gamma == gamma_teach: the discount in the discounted safety backup.
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("teacher discount gamma_teach must lie in (0, 1].")
        self.dyn_cfg = dyn_cfg
        self.cfg = cfg
        self.gamma = float(gamma)
        self.model = model if model is not None else DubinsModel(dyn_cfg)
        self.grid = OracleGrid.build(dyn_cfg, cfg)
        self._build_position_stencil()

    # ------------------------------------------------------------------ #
    def _build_position_stencil(self) -> None:
        g = self.grid
        c = self.dyn_cfg
        PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
        npx = PX + c.dt * c.v * np.cos(PSI)
        npy = PY + c.dt * c.v * np.sin(PSI)

        hx = g.px[1] - g.px[0]
        hy = g.py[1] - g.py[0]
        fx = (npx - g.px[0]) / hx
        fy = (npy - g.py[0]) / hy
        self.out_of_domain = ((npx < c.p_lo) | (npx > c.p_hi)
                              | (npy < c.p_lo) | (npy > c.p_hi))

        ix = np.clip(np.floor(fx).astype(np.int64), 0, len(g.px) - 2)
        iy = np.clip(np.floor(fy).astype(np.int64), 0, len(g.py) - 2)
        ax = np.clip(fx - ix, 0.0, 1.0).astype(np.float32)
        ay = np.clip(fy - iy, 0.0, 1.0).astype(np.float32)

        n_psi = len(g.psi)
        # flat indices into a (n_px*n_py, n_psi) value table, 4 xy-neighbors
        base00 = (ix * len(g.py) + iy)
        base10 = ((ix + 1) * len(g.py) + iy)
        base01 = (ix * len(g.py) + (iy + 1))
        base11 = ((ix + 1) * len(g.py) + (iy + 1))
        self.nbr_idx = np.stack([base00, base10, base01, base11], axis=-1)  # (...,4)
        w00 = (1 - ax) * (1 - ay)
        w10 = ax * (1 - ay)
        w01 = (1 - ax) * ay
        w11 = ax * ay
        self.nbr_w = np.stack([w00, w10, w01, w11], axis=-1)                # (...,4)
        _ = n_psi  # heading handled per (u, d) via uniform shift

    # ------------------------------------------------------------------ #
    def _future_value(self, V: np.ndarray) -> np.ndarray:
        """Compute F[V](x) = max_u min_d V(f(x,u,d)) for all grid states."""
        g = self.grid
        c = self.dyn_cfg
        n_psi = len(g.psi)
        h_psi = TWO_PI / n_psi
        Vflat = V.reshape(-1, n_psi)            # (n_px*n_py, n_psi)
        best = np.full(V.shape, -np.inf, dtype=np.float32)
        k_idx = np.arange(n_psi)[None, None, :, None]

        for u in g.u:
            worst = np.full(V.shape, np.inf, dtype=np.float32)
            for d in g.d:
                s = c.dt * float(self.model.heading_rate(u, d))  # uniform shift
                m = int(np.floor(s / h_psi))
                a = np.float32(s / h_psi - m)
                # psi-bracketing slices: V0[:, k] = V(.,., psi_k + m*h),
                #                        V1[:, k] = V(.,., psi_k + (m+1)*h)
                V0 = np.roll(Vflat, -m, axis=1)
                V1 = np.roll(Vflat, -(m + 1), axis=1)
                Vs = (1 - a) * V0 + a * V1          # lerp in psi
                vals = Vs[self.nbr_idx, k_idx]      # (npx,npy,npsi,4)
                interp = np.einsum("...k,...k->...", vals, self.nbr_w)
                interp = np.where(self.out_of_domain, np.float32(c.g_fail), interp)
                np.minimum(worst, interp, out=worst)
            np.maximum(best, worst, out=best)
        return best

    # ------------------------------------------------------------------ #
    def solve(self, verbose: bool = True) -> dict:
        g = self.grid
        PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
        states = np.stack([PX, PY, PSI], axis=-1)
        gval = self.model.g(states).astype(np.float32)

        V = gval.copy()
        t0 = time.time()
        history = []
        lam = np.float32(self.gamma)            # teacher discount (gamma_teach)
        for it in range(self.cfg.max_iters):
            fut = self._future_value(V)
            # discounted safety backup (Fisac/Akametalu):
            #   V <- (1 - lam) g + lam * min(g, max_u min_d V(f))
            Vn = (np.float32(1.0) - lam) * gval + lam * np.minimum(gval, fut)
            delta = float(np.max(np.abs(Vn - V)))
            history.append(delta)
            V = Vn
            if verbose and (it % 10 == 0 or delta <= self.cfg.tol):
                print(f"  VI iter {it:4d}  |dV|_inf = {delta:.3e}  "
                      f"({time.time()-t0:.1f}s)")
            if delta <= self.cfg.tol:
                break
        return {
            "V": V, "g": gval, "states": states.astype(np.float32),
            "iters": it + 1, "residual": history[-1], "history": np.array(history),
            "wall_s": time.time() - t0, "gamma": self.gamma,
        }

    # ------------------------------------------------------------------ #
    def interp_V(self, V: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of V at arbitrary states (periodic in psi).

        Out-of-domain positions evaluate to g_fail.
        """
        g = self.grid
        c = self.dyn_cfg
        px, py, psi = x[..., 0], x[..., 1], x[..., 2]
        hx = g.px[1] - g.px[0]
        hy = g.py[1] - g.py[0]
        hp = TWO_PI / len(g.psi)

        ood = ((px < c.p_lo) | (px > c.p_hi) | (py < c.p_lo) | (py > c.p_hi))
        fx = (px - g.px[0]) / hx
        fy = (py - g.py[0]) / hy
        fp = (psi - g.psi[0]) / hp
        ix = np.clip(np.floor(fx).astype(np.int64), 0, len(g.px) - 2)
        iy = np.clip(np.floor(fy).astype(np.int64), 0, len(g.py) - 2)
        ip = np.floor(fp).astype(np.int64)
        ax = np.clip(fx - ix, 0, 1)
        ay = np.clip(fy - iy, 0, 1)
        ap = fp - ip
        n_psi = len(g.psi)
        ip0 = ip % n_psi
        ip1 = (ip + 1) % n_psi

        out = np.zeros(np.shape(px), dtype=np.float64)
        for dx_, wx in ((0, 1 - ax), (1, ax)):
            for dy_, wy in ((0, 1 - ay), (1, ay)):
                for ipk, wp in ((ip0, 1 - ap), (ip1, ap)):
                    out += wx * wy * wp * V[ix + dx_, iy + dy_, ipk]
        return np.where(ood, c.g_fail, out)

    def q_star(self, V: np.ndarray, x: np.ndarray, u: np.ndarray,
               d: np.ndarray) -> np.ndarray:
        """Q_HJ(x,u,d) = V_HJ(f(x,u,d)) with interpolation."""
        nxt = self.model.step(x, u, d)
        return self.interp_V(V, nxt)

    def fallback_labels(self, V: np.ndarray) -> np.ndarray:
        """u_HJ(x) = argmax_u min_d Q_HJ(x,u,d) on the state grid."""
        g = self.grid
        shape = V.shape
        best_val = np.full(shape, -np.inf)
        best_u = np.zeros(shape)
        PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
        X = np.stack([PX, PY, PSI], axis=-1)
        for u in g.u:
            worst = np.full(shape, np.inf)
            for d in g.d:
                q = self.q_star(V, X, np.full(shape, u), np.full(shape, d))
                np.minimum(worst, q, out=worst)
            sel = worst > best_val
            best_val = np.where(sel, worst, best_val)
            best_u = np.where(sel, u, best_u)
        return best_u.astype(np.float32), best_val.astype(np.float32)


# Backwards-compatible name: the Dubins stage scripts import ``DubinsOracle``.
DubinsOracle = GridOracle
