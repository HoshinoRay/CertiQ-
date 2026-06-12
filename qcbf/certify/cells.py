"""Cell lattice over X = [p_lo, p_hi]^2 x S^1 (Mode B / Prop T4 certificate).

The certificate object is a boolean mask over a uniform lattice of closed
boxes (cells).  The heading axis is periodic; cells never straddle the
-pi/+pi seam, and successor boxes that wrap have already been split by
`successor_boxes`, so every box handled here has  psi_lo <= psi_hi  within
[-pi, pi].

Two conservative primitives are provided:

  * `cell_index(x)`      - which cell a state belongs to (runtime membership),
  * `box_index_ranges`   - the inclusive lattice index ranges overlapped by a
                           box, rounded OUTWARD (touching a cell boundary
                           counts as overlapping both neighbours).  Outward
                           rounding can only add constraints, so it is sound.

For the C2 fixed point, acceptance counts inside an index range are queried
with a 3-D summed-area table whose psi axis is tiled twice, so the seam never
needs special-casing (psi_hi == +pi maps to tiled index n_psi, which aliases
cell 0 -- exactly the periodic identification).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qcbf.config import CertConfig, DubinsConfig

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
@dataclass
class CellLattice:
    nx: int
    ny: int
    npsi: int
    p_lo: float
    p_hi: float
    hx: float
    hy: float
    hp: float

    # ------------------------------------------------------------------ #
    @staticmethod
    def build(dyn: DubinsConfig, cert: CertConfig) -> "CellLattice":
        nx, ny, npsi = cert.n_cells_px, cert.n_cells_py, cert.n_cells_psi
        hx = (dyn.p_hi - dyn.p_lo) / nx
        hy = (dyn.p_hi - dyn.p_lo) / ny
        hp = TWO_PI / npsi
        return CellLattice(nx, ny, npsi, dyn.p_lo, dyn.p_hi, hx, hy, hp)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.npsi

    @property
    def cell_volume(self) -> float:
        return self.hx * self.hy * self.hp

    # ------------------------------------------------------------------ #
    def boxes(self) -> np.ndarray:
        """(N, 6) array [px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi]."""
        ix, iy, ip = np.meshgrid(np.arange(self.nx), np.arange(self.ny),
                                 np.arange(self.npsi), indexing="ij")
        px_lo = self.p_lo + ix * self.hx
        py_lo = self.p_lo + iy * self.hy
        ps_lo = -np.pi + ip * self.hp
        out = np.stack([px_lo, px_lo + self.hx,
                        py_lo, py_lo + self.hy,
                        ps_lo, ps_lo + self.hp], axis=-1)
        return out.reshape(-1, 6)

    def linear_index(self, ix, iy, ip):
        return (ix * self.ny + iy) * self.npsi + ip

    # ------------------------------------------------------------------ #
    def cell_index(self, x: np.ndarray) -> np.ndarray:
        """Linear cell index of states (..., 3); -1 if off the position domain."""
        px, py, psi = x[..., 0], x[..., 1], x[..., 2]
        ix = np.floor((px - self.p_lo) / self.hx).astype(np.int64)
        iy = np.floor((py - self.p_lo) / self.hy).astype(np.int64)
        psi_w = psi - np.floor((psi + np.pi) / TWO_PI) * TWO_PI  # [-pi, pi)
        ip = np.floor((psi_w + np.pi) / self.hp).astype(np.int64) % self.npsi
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ixc = np.clip(ix, 0, self.nx - 1)
        iyc = np.clip(iy, 0, self.ny - 1)
        return np.where(ok, self.linear_index(ixc, iyc, ip), -1)

    # ------------------------------------------------------------------ #
    def box_index_ranges(self, boxes: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Inclusive index ranges overlapped by boxes (M, 6), rounded outward.

        Returns
        -------
        rng    : (M, 6) int16  [ix0, ix1, iy0, iy1, ip0, ip1]
                 psi indices live on the DOUBLED axis [0, 2*npsi); they never
                 wrap because input boxes never straddle the seam.
        dom_ok : (M,) bool  -- box position lies inside [p_lo, p_hi]^2
        """
        b = np.asarray(boxes, np.float64)
        dom_ok = ((b[:, 0] >= self.p_lo - 1e-12) & (b[:, 1] <= self.p_hi + 1e-12)
                  & (b[:, 2] >= self.p_lo - 1e-12) & (b[:, 3] <= self.p_hi + 1e-12))

        def rng_axis(lo, hi, origin, h, n_max):
            i0 = np.floor((lo - origin) / h).astype(np.int64)
            i1 = np.floor((hi - origin) / h).astype(np.int64)
            # hi exactly on an edge -> floor lands on the next cell: keep it
            # (outward rounding, conservative).  Clip to the valid index set.
            return (np.clip(i0, 0, n_max - 1).astype(np.int16),
                    np.clip(i1, 0, n_max - 1).astype(np.int16))

        ix0, ix1 = rng_axis(b[:, 0], b[:, 1], self.p_lo, self.hx, self.nx)
        iy0, iy1 = rng_axis(b[:, 2], b[:, 3], self.p_lo, self.hy, self.ny)
        ip0, ip1 = rng_axis(b[:, 4], b[:, 5], -np.pi, self.hp, 2 * self.npsi)
        rng = np.stack([ix0, ix1, iy0, iy1, ip0, ip1], axis=-1)
        return rng, dom_ok


# --------------------------------------------------------------------------- #
def acceptance_prefix_sum(lattice: CellLattice, accepted: np.ndarray) -> np.ndarray:
    """Zero-padded 3-D summed-area table of the accepted mask, psi tiled x2."""
    A = accepted.reshape(lattice.nx, lattice.ny, lattice.npsi)
    A2 = np.concatenate([A, A], axis=2).astype(np.int32)
    P = np.zeros((lattice.nx + 1, lattice.ny + 1, 2 * lattice.npsi + 1),
                 dtype=np.int32)
    P[1:, 1:, 1:] = A2.cumsum(0).cumsum(1).cumsum(2)
    return P


def ranges_fully_accepted(P: np.ndarray, rng: np.ndarray) -> np.ndarray:
    """True where every cell in the inclusive index range is accepted.

    P   : prefix sum from `acceptance_prefix_sum`
    rng : (M, 6) int  [ix0, ix1, iy0, iy1, ip0, ip1]
    """
    x0 = rng[:, 0].astype(np.int64); x1 = rng[:, 1].astype(np.int64) + 1
    y0 = rng[:, 2].astype(np.int64); y1 = rng[:, 3].astype(np.int64) + 1
    p0 = rng[:, 4].astype(np.int64); p1 = rng[:, 5].astype(np.int64) + 1
    s = (P[x1, y1, p1] - P[x0, y1, p1] - P[x1, y0, p1] - P[x1, y1, p0]
         + P[x0, y0, p1] + P[x0, y1, p0] + P[x1, y0, p0] - P[x0, y0, p0])
    need = (x1 - x0) * (y1 - y0) * (p1 - p0)
    return s == need
