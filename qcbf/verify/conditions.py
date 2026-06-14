"""Verified C3 cell checks (Theorem A).

All checks here are SOUND: each can only fail to certify a true property,
never certify a false one.  Looseness only shrinks the certified set.

  C3  h3 = Q(x, clip(pi(x)), d) - gamma_deploy V(x) - eps >= 0 on cell x D
       -- CROWN lower bound on the COMPILED composite network, evaluated
          jointly in (x, d) so the x <-> u correlation is preserved.
          Staged: cheap bounds first, expensive bounds and sub-splitting
          only on the cells left undecided (all stages are sound, and a
          stage can only ADD certified cells, so staging itself is sound).

C1 (exact closed-form g box bounds) and C4 live in qcbf/certify/spec.py.
"""
from __future__ import annotations

import time

import numpy as np

from qcbf.config import CertConfig, DubinsConfig
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked


# --------------------------------------------------------------------------- #
def v_cell_bounds(v_net: SeqNet, boxes: np.ndarray, idx: np.ndarray,
                  n_cells: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Certified (lb, ub) of V_theta over the listed cells; -inf/+inf elsewhere."""
    lb = np.full(n_cells, -np.inf)
    ub = np.full(n_cells, np.inf)
    if len(idx) == 0:
        return lb, ub
    lo, hi = crown_bounds_chunked(v_net, boxes[idx][:, [0, 2, 4]],
                                  boxes[idx][:, [1, 3, 5]], True, chunk,
                                  progress="V-bounds")
    lb[idx] = lo[:, 0]
    ub[idx] = hi[:, 0]
    return lb, ub


# --------------------------------------------------------------------------- #
def _subsplit_boxes(lo: np.ndarray, hi: np.ndarray, splits: list[int]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Split (M, n) boxes into prod(splits) sub-boxes along each axis.

    Returns (M * S, n) arrays ordered box-major.
    """
    M, n = lo.shape
    los, his = [lo], [hi]
    for ax, k in enumerate(splits):
        if k <= 1:
            continue
        new_lo, new_hi = [], []
        for L, H in zip(los, his):
            w = (H[:, ax] - L[:, ax]) / k
            for j in range(k):
                l2, h2 = L.copy(), H.copy()
                l2[:, ax] = L[:, ax] + j * w
                h2[:, ax] = L[:, ax] + (j + 1) * w
                new_lo.append(l2)
                new_hi.append(h2)
        los, his = new_lo, new_hi
    S = len(los)
    lo_all = np.stack(los, axis=1).reshape(M * S, n)
    hi_all = np.stack(his, axis=1).reshape(M * S, n)
    return lo_all, hi_all


def check_c3_staged(h3_net: SeqNet, boxes: np.ndarray, idx: np.ndarray,
                    dyn: DubinsConfig, cert: CertConfig,
                    n_cells: int, verbose: bool = True
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Staged certified check of  h3 >= 0  on cell x D for cells `idx`.

    Stage A: whole cell, whole D, IBP intermediate bounds (fast).
    Stage B: whole cell, whole D, CROWN intermediate bounds.
    Stage C: state sub-split x d sub-split, CROWN intermediate bounds;
             a cell passes iff EVERY sub-box has lb >= 0.

    Returns (c3_ok (N,), c3_lb (N,) best certified lower bound, funnel dict).
    """
    c3_ok = np.zeros(n_cells, dtype=bool)
    c3_lb = np.full(n_cells, -np.inf)
    if len(idx) == 0:
        return c3_ok, c3_lb, {"stageA_pass": 0, "stageB_pass": 0,
                              "stageC_pass": 0, "checked": 0}
    d = dyn.d_max
    lo = np.column_stack([boxes[idx, 0], boxes[idx, 2], boxes[idx, 4],
                          np.full(len(idx), -d)])
    hi = np.column_stack([boxes[idx, 1], boxes[idx, 3], boxes[idx, 5],
                          np.full(len(idx), +d)])
    funnel = {"checked": int(len(idx))}
    t0 = time.time()

    # ---- stage A --------------------------------------------------------- #
    lbA, _ = crown_bounds_chunked(h3_net, lo, hi, False, cert.chunk,
                                  progress="C3-A" if verbose else None)
    passA = lbA[:, 0] >= 0.0
    c3_ok[idx[passA]] = True
    c3_lb[idx] = lbA[:, 0]
    funnel["stageA_pass"] = int(passA.sum())
    if verbose:
        print(f"  [C3-A] IBP-intermediate     pass {passA.sum():6d}/{len(idx)}"
              f"  ({time.time()-t0:.1f}s)")

    # ---- stage B --------------------------------------------------------- #
    und = ~passA
    t0 = time.time()
    if und.any():
        lbB, _ = crown_bounds_chunked(h3_net, lo[und], hi[und], True, cert.chunk,
                                      progress="C3-B" if verbose else None)
        passB = lbB[:, 0] >= 0.0
        sub_idx = idx[und]
        c3_ok[sub_idx[passB]] = True
        c3_lb[sub_idx] = np.maximum(c3_lb[sub_idx], lbB[:, 0])
        funnel["stageB_pass"] = int(passB.sum())
        if verbose:
            print(f"  [C3-B] CROWN-intermediate   pass {passB.sum():6d}/"
                  f"{und.sum()}  ({time.time()-t0:.1f}s)")
        und2 = sub_idx[~passB]
    else:
        funnel["stageB_pass"] = 0
        und2 = np.array([], dtype=np.int64)

    # ---- stage C --------------------------------------------------------- #
    t0 = time.time()
    if len(und2) and (cert.c3_state_subsplit > 1 or cert.c3_d_subsplit > 1):
        s = cert.c3_state_subsplit
        splits = [s, s, s, cert.c3_d_subsplit]
        pos = np.searchsorted(idx, und2)
        lo2, hi2 = _subsplit_boxes(lo[pos], hi[pos], splits)
        S = len(lo2) // len(und2)
        lbC, _ = crown_bounds_chunked(h3_net, lo2, hi2, True, cert.chunk,
                                      progress="C3-C" if verbose else None)
        lbC = lbC[:, 0].reshape(len(und2), S)
        worst = lbC.min(axis=1)
        passC = worst >= 0.0
        c3_ok[und2[passC]] = True
        c3_lb[und2] = np.maximum(c3_lb[und2], worst)
        funnel["stageC_pass"] = int(passC.sum())
        if verbose:
            print(f"  [C3-C] sub-split ({S}x)      pass {passC.sum():6d}/"
                  f"{len(und2)}  ({time.time()-t0:.1f}s)")
    else:
        funnel["stageC_pass"] = 0
    return c3_ok, c3_lb, funnel
