"""Reference robust capture basin (NON-sound calibration target).

The sound grow engine (`grow.py`) produces an UNDER-approximation `R_inf` of the
robust capture basin of the seed within the constraint set.  To know how much of
the residual `Omega* \\ R_inf` is *method conservatism* (recoverable by finer
grids / macro-actions / tighter envelopes) versus an *intrinsic capture-basin
gap* (states that cannot robustly reach the seed under ANY control), we compute a
reference capture basin by backward reachability on the SAMPLED true dynamics.

This is a NOMINAL grid reference, NOT a certificate: it evaluates the dynamics at
each cell CENTRE over a finite disturbance sample, so it neither soundly
under- nor over-approximates the true kernel.  Because the centre ignores
intra-cell spread it tends to be *optimistic* (a mild upper reference on what a
sound primitive could certify at this resolution), which is exactly what we want
as a headroom target:  `R_inf  <~  Capt_ref  <~  true capture basin`.

    Capt = lfp_C [ seed  u  { c in dom :  exists u in menu,  for all sampled d,
                              cell(f(centre(c), u, d)) in C } ]
"""
from __future__ import annotations

import numpy as np

from ..util.progress import Progress


def _cell_flat(grid, X):
    """Flat 4-D cell index of states X plus an in-(px,py)-domain mask."""
    npx, npy, npsi, nv = grid.shape
    wpx, wpy, wps, wv = grid.widths
    i = np.clip(((X[:, 0] - grid.pxe[0]) / wpx).astype(np.int64), 0, npx - 1)
    j = np.clip(((X[:, 1] - grid.pye[0]) / wpy).astype(np.int64), 0, npy - 1)
    psi = (X[:, 2] + np.pi) % (2 * np.pi) - np.pi
    k = np.mod(((psi - grid.pse[0]) / wps).astype(np.int64), npsi)
    l = np.clip(((X[:, 3] - grid.ve[0]) / wv).astype(np.int64), 0, nv - 1)
    in_dom = ((X[:, 0] >= grid.pxe[0]) & (X[:, 0] <= grid.pxe[-1])
              & (X[:, 1] >= grid.pye[0]) & (X[:, 1] <= grid.pye[-1]))
    return (((i * npy + j) * npsi + k) * nv + l), in_dom


def robust_capture_basin(grid, model, cfg, seed4d, dom4d, menu, n_d=3,
                         max_iter=200, verbose=True, tag="capt"):
    """Reference robust capture basin of ``seed4d`` inside ``dom4d`` (both bool
    4-D grids) under the menu and the true dynamics, adversarial over an n_d x n_d
    disturbance grid.  Returns the bool 4-D capture set (>= seed4d)."""
    npx, npy, npsi, nv = grid.shape
    C = seed4d.copy()
    # candidate domain cells (in dom, not yet captured) and their centres
    di, dj, dk, dl = np.nonzero(dom4d & ~seed4d)
    cx = 0.5 * (grid.pxe[di] + grid.pxe[di + 1])
    cy = 0.5 * (grid.pye[dj] + grid.pye[dj + 1])
    cp = 0.5 * (grid.pse[dk] + grid.pse[dk + 1])
    cv = 0.5 * (grid.ve[dl] + grid.ve[dl + 1])
    X = np.column_stack([cx, cy, cp, cv])
    nC = len(di)
    das = np.linspace(-cfg.d_a_max, cfg.d_a_max, n_d)
    dds = np.linspace(-cfg.d_delta_max, cfg.d_delta_max, n_d)
    dsamples = [(a, b) for a in das for b in dds]
    # precompute successor FLAT cell index per (action, d-sample): worst d is the
    # one whose successor is NOT in C, so a cell qualifies for action u iff EVERY
    # d-sample's successor is in C.  Dynamics are C-independent -> computed once.
    succ_flat = []                  # list over actions: list over d: (flat, in_dom)
    pb = Progress(len(menu), f"{tag}-precompute") if verbose else None
    for a, (acmd, dcmd) in enumerate(menu):
        u = np.column_stack([np.full(nC, acmd), np.full(nC, dcmd)])
        per_d = []
        for (da, dd) in dsamples:
            d = np.column_stack([np.full(nC, da), np.full(nC, dd)])
            flat, ind = _cell_flat(grid, model.step(X, u, d))
            per_d.append((flat, ind))
        succ_flat.append(per_d)
        if pb is not None:
            pb.update(a + 1)
    if pb is not None:
        pb.done()

    cflat = (((di * npy + dj) * npsi + dk) * nv + dl)
    Cf = C.reshape(-1)
    added = np.zeros(nC, bool)
    hist = [int(C.sum())]
    for it in range(max_iter):
        rem = np.flatnonzero(~added)
        if len(rem) == 0:
            break
        qualifies = np.zeros(len(rem), bool)
        for per_d in succ_flat:                     # any action
            ok = np.ones(len(rem), bool)
            for (flat, ind) in per_d:               # all sampled d (worst-case)
                ok &= Cf[flat[rem]] & ind[rem]
                if not ok.any():
                    break
            qualifies |= ok
        newly = rem[qualifies]
        if len(newly) == 0:
            break
        Cf[cflat[newly]] = True
        added[newly] = True
        hist.append(int(Cf.sum()))
    if verbose:
        print(f"  [{tag}] capture basin {int(C.sum())} cells "
              f"(+{int(C.sum()) - hist[0]} over seed) in {len(hist) - 1} sweeps",
              flush=True)
    return C, {"n_capt": int(C.sum()), "n_seed": int(seed4d.sum()),
               "sweeps": len(hist) - 1, "n_d_grid": n_d}
