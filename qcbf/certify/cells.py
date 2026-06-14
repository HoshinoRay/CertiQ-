"""Cell lattice over X = [p_lo, p_hi]^2 x S^1.

The certificate reports a boolean mask over a uniform lattice of closed boxes
(cells).  The heading axis is periodic; cells never straddle the -pi/+pi seam.

`boxes()` materialises the cell boxes for the verifier, and `cell_index(x)`
maps a runtime state to its cell (used by the runtime filter and the audit to
test membership in the exported inner-cell set).
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
