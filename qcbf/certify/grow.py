"""Grow-from-seed certified invariant-set expansion (the lfp engine).

Implements the construction of ``grow_from_seed_certified_expansion.md`` Sec. 2-4
and 8 for the brakeable F1TENTH plant: from an already-certified brake seed
``R_0 = S_brake`` it grows a robustly forward-invariant set OUTWARD by the
verified greatest-under-approximation operator

    G_V(R) = R u ( K n V_0 n Pre_ver(R) ),
    Pre_ver(R) = { c : exists u in U_menu,  Enc(c,u) subset of  U R } ,

where ``Enc`` is the SOUND one-step successor box (A3) and membership is the
outward-rounded cell test (A4).  Every iterate ``R_k`` is itself a sound robust
invariant set (anytime soundness, Theorem A) -- unlike a gfp closure, no
intermediate over-approximation is ever trusted, so finite resolution cannot
mistake an un-converged set for an invariant one.

Soundness rests ONLY on {A3 envelope, A4 membership, A2 seed certificate}; the
learned ``V_theta`` enters solely through the B1 anchor ``V_0 = {lb V_theta >= 0}``
(pure set subtraction, never an invariance assumption).  The 4-D grid is
(px, py, psi, v): unlike the heading-free brake successor, a racing/steer menu
action evolves the heading, so growth is genuinely heading-dependent.

The engine is plant-agnostic in its containment core; the only plant hooks are
the successor box and heading interval passed in via ``encode_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..util.progress import Progress

_EPS = 1e-9


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid4D:
    """Edges + widths of the (px, py, psi, v) certification lattice."""
    pxe: np.ndarray
    pye: np.ndarray
    pse: np.ndarray
    ve: np.ndarray

    @property
    def shape(self):
        return (len(self.pxe) - 1, len(self.pye) - 1,
                len(self.pse) - 1, len(self.ve) - 1)

    @property
    def widths(self):
        return (self.pxe[1] - self.pxe[0], self.pye[1] - self.pye[0],
                self.pse[1] - self.pse[0], self.ve[1] - self.ve[0])

    @staticmethod
    def make(p_lo, p_hi, v_max, npx, npy, npsi, nv):
        return Grid4D(np.linspace(p_lo, p_hi, npx + 1),
                      np.linspace(p_lo, p_hi, npy + 1),
                      np.linspace(-np.pi, np.pi, npsi + 1),
                      np.linspace(0.0, v_max, nv + 1))


# --------------------------------------------------------------------------- #
def _prefix_sum(R):
    """4-D inclusive prefix sum P with a leading zero plane per axis, so the
    sum over inclusive index box [i0..i1]x[j0..j1]x[k0..k1]x[l0..l1] is a
    16-corner inclusion-exclusion at indices {i0,i1+1}x...  (P[m]=sum cells<m)."""
    npx, npy, npsi, nv = R.shape
    P = np.zeros((npx + 1, npy + 1, npsi + 1, nv + 1), dtype=np.int64)
    P[1:, 1:, 1:, 1:] = (R.astype(np.int64).cumsum(0).cumsum(1)
                         .cumsum(2).cumsum(3))
    return P


def _boxsum(P, i0, i1, j0, j1, k0, k1, l0, l1):
    """Vectorized sum of P's underlying array over inclusive index boxes.

    All args are int arrays of equal length; ranges are inclusive.  k-range is
    a plain (non-wrapping) contiguous range here -- wrap is split by the caller.
    """
    a0, a1 = i0, i1 + 1
    b0, b1 = j0, j1 + 1
    c0, c1 = k0, k1 + 1
    d0, d1 = l0, l1 + 1
    s = np.zeros(len(i0), dtype=np.int64)
    for ia, sa in ((a1, 1), (a0, -1)):
        for jb, sb in ((b1, 1), (b0, -1)):
            for kc, sc in ((c1, 1), (c0, -1)):
                for ld, sd in ((d1, 1), (d0, -1)):
                    s += sa * sb * sc * sd * P[ia, jb, kc, ld]
    return s


# --------------------------------------------------------------------------- #
class GrowEngine:
    """lfp expansion on the 4-D lattice.  ``encode_fn(a_cmd, delta_cmd, lo, hi)``
    returns the sound successor box (px+,py+,v+ over full D) and the raw
    (un-wrapped) psi+ interval, as eight arrays:
        npx_lo, npx_hi, npy_lo, npy_hi, npsi_lo, npsi_hi, nv_lo, nv_hi.
    """

    def __init__(self, grid: Grid4D, seed3d, VK3d, menu, encode_fn):
        self.g = grid
        self.menu = list(menu)                     # [(a_cmd, delta_cmd), ...]
        self.encode_fn = encode_fn
        self.npx, self.npy, self.npsi, self.nv = grid.shape
        self.seed3d = seed3d.astype(bool)          # (npx,npy,nv) brake seed
        VK3d = VK3d.astype(bool) | self.seed3d     # K & V_0 (seed is always in it)
        self.VK3d = VK3d
        # lift to 4-D (heading-independent masks): seed/VK do not depend on psi
        self.R = np.broadcast_to(seed3d[:, :, None, :],
                                 grid.shape).copy()       # bool, current invariant set
        self.VK = np.broadcast_to(VK3d[:, :, None, :], grid.shape).copy()
        self.layer = np.where(self.R, 0, -1).astype(np.int32)
        self.witness = np.full(grid.shape, -1, dtype=np.int32)
        self.n_seed4d = int(self.R.sum())

    # ------------------------------------------------------------------ #
    def _candidate_cells(self):
        """4-D candidate cells = headings of (VK3d & ~seed3d) -- the only cells
        G_V can ever add (R subset of K n V_0, seed already in R)."""
        cand3d = self.VK3d & ~self.seed3d
        ii, jj, ll = np.nonzero(cand3d)            # 3-D candidate cells
        n3 = len(ii)
        K = self.npsi
        # cartesian product with all headings
        ci = np.repeat(ii, K); cj = np.repeat(jj, K); cl = np.repeat(ll, K)
        ck = np.tile(np.arange(K), n3)
        return ci, cj, ck, cl

    # ------------------------------------------------------------------ #
    def _idx_ranges(self, box_lo, box_hi, edges, periodic=False):
        """Outward-rounded inclusive cell-index range [i0, i1] for a coord box.

        Non-periodic: returns (i0, i1, in_domain).  Periodic (heading): returns
        the RAW (possibly <0 or >=n) indices; wrap handled by the caller.
        """
        g_lo = edges[0]
        w = edges[1] - edges[0]
        n = len(edges) - 1
        i0 = np.floor((box_lo - g_lo) / w - _EPS).astype(np.int64)
        i1 = np.floor((box_hi - g_lo) / w + _EPS).astype(np.int64)
        if periodic:
            return i0, i1, None
        in_dom = (i0 >= 0) & (i1 <= n - 1)
        return np.clip(i0, 0, n - 1), np.clip(i1, 0, n - 1), in_dom

    def _precompute(self, ci, cj, ck, cl, verbose=True):
        """Per (candidate, action) successor index boxes + permanent validity
        (in-domain AND touched subset of K n V_0).  R-independent -> computed once."""
        gx, gy, gp, gv = self.g.pxe, self.g.pye, self.g.pse, self.g.ve
        lo = np.column_stack([gx[ci], gy[cj], gp[ck], gv[cl]])
        hi = np.column_stack([gx[ci + 1], gy[cj + 1], gp[ck + 1], gv[cl + 1]])
        PVK = _prefix_sum(self.VK)
        nc, na = len(ci), len(self.menu)
        store = []
        pb = Progress(na, "precompute-Enc") if verbose else None
        for a, (acmd, dcmd) in enumerate(self.menu):
            nx_lo, nx_hi, ny_lo, ny_hi, np_lo, np_hi, nv_lo, nv_hi = \
                self.encode_fn(acmd, dcmd, lo, hi)
            i0, i1, dom_x = self._idx_ranges(nx_lo, nx_hi, gx)
            j0, j1, dom_y = self._idx_ranges(ny_lo, ny_hi, gy)
            l0, l1, _ = self._idx_ranges(nv_lo, nv_hi, gv)        # v clipped -> in dom
            k0r, k1r, _ = self._idx_ranges(np_lo, np_hi, gp, periodic=True)
            kspan = np.minimum(k1r - k0r + 1, self.npsi)
            # heading wrap -> up to two contiguous index ranges
            full = (k1r - k0r + 1) >= self.npsi
            ka0 = np.mod(k0r, self.npsi); ka1 = np.mod(k1r, self.npsi)
            wrap = (ka0 > ka1) & ~full
            kA0 = np.where(full, 0, ka0)
            kA1 = np.where(full, self.npsi - 1, np.where(wrap, self.npsi - 1, ka1))
            kB0 = np.where(wrap, 0, 0)
            kB1 = np.where(wrap, ka1, -1)            # kB inactive when kB1 < kB0
            boxvol = ((i1 - i0 + 1) * (j1 - j0 + 1) * kspan * (l1 - l0 + 1)
                      ).astype(np.int64)
            # permanent validity: in-domain and every touched cell in K n V_0
            vk = _boxsum(PVK, i0, i1, j0, j1, kA0, kA1, l0, l1)
            kB_act = kB1 >= kB0
            vk = vk + np.where(kB_act,
                               _boxsum(PVK, i0, i1, j0, j1,
                                       np.where(kB_act, kB0, 0),
                                       np.where(kB_act, kB1, 0), l0, l1), 0)
            valid = dom_x & dom_y & (vk == boxvol)
            store.append(dict(i0=i0, i1=i1, j0=j0, j1=j1, l0=l0, l1=l1,
                              kA0=kA0, kA1=kA1, kB0=kB0, kB1=kB1, kB_act=kB_act,
                              boxvol=boxvol, valid=valid))
            if pb is not None:
                pb.update(a + 1)
        if pb is not None:
            pb.done()
        return store

    # ------------------------------------------------------------------ #
    def run(self, n_omega3d, max_waves=400, verbose=True):
        """Run the lfp to convergence.  Returns a diagnostics dict including the
        rho(k) history (rho = |R_4d| / (|Omega*_3d| * npsi))."""
        ci, cj, ck, cl = self._candidate_cells()
        nc = len(ci)
        denom = max(n_omega3d * self.npsi, 1)
        if verbose:
            print(f"  [grow] {nc} 4-D candidate cells "
                  f"({nc // max(self.npsi,1)} 3-D x {self.npsi} headings), "
                  f"seed {self.n_seed4d} 4-D cells, menu {len(self.menu)} actions",
                  flush=True)
        store = self._precompute(ci, cj, ck, cl, verbose=verbose)
        n_valid_pairs = int(sum(s["valid"].sum() for s in store))

        added = np.zeros(nc, bool)                  # candidate already in R?
        cflat = (((ci * self.npy + cj) * self.npsi + ck) * self.nv + cl)
        rho_hist = [self.n_seed4d / denom]
        wave_added = []
        t_idx = np.arange(nc)

        for wave in range(1, max_waves + 1):
            P = _prefix_sum(self.R)
            rem = np.flatnonzero(~added)            # still-candidate indices
            if len(rem) == 0:
                break
            pass_any = np.zeros(len(rem), bool)
            win_act = np.full(len(rem), -1, np.int32)
            for a, s in enumerate(store):
                m = s["valid"][rem] & ~pass_any     # untried/not-yet-won, valid
                if not m.any():
                    continue
                r = rem[m]
                bs = _boxsum(P, s["i0"][r], s["i1"][r], s["j0"][r], s["j1"][r],
                             s["kA0"][r], s["kA1"][r], s["l0"][r], s["l1"][r])
                kb = s["kB_act"][r]
                if kb.any():
                    bs = bs + np.where(kb, _boxsum(
                        P, s["i0"][r], s["i1"][r], s["j0"][r], s["j1"][r],
                        np.where(kb, s["kB0"][r], 0), np.where(kb, s["kB1"][r], 0),
                        s["l0"][r], s["l1"][r]), 0)
                ok = bs == s["boxvol"][r]
                idx_local = np.flatnonzero(m)[ok]
                pass_any[idx_local] = True
                win_act[idx_local] = a
            newly = rem[pass_any]
            if len(newly) == 0:
                break                                # fixed point reached
            # commit this wave
            gi, gj, gk, gl = ci[newly], cj[newly], ck[newly], cl[newly]
            self.R[gi, gj, gk, gl] = True
            self.layer[gi, gj, gk, gl] = wave
            self.witness[gi, gj, gk, gl] = win_act[pass_any]
            added[newly] = True
            wave_added.append(int(len(newly)))
            rho_hist.append(int(self.R.sum()) / denom)
            if verbose:
                print(f"    wave {wave:3d}: +{len(newly):6d} cells -> "
                      f"|R|={int(self.R.sum()):7d}  rho={rho_hist[-1]:.4f}",
                      flush=True)

        n_R = int(self.R.sum())
        grown = n_R - self.n_seed4d
        # failure decomposition over the candidates never added
        not_added = ~added
        reach_blocked = self._blocked_breakdown(ci, cj, ck, cl, store,
                                                not_added)
        return {
            "n_candidates_4d": int(nc),
            "n_valid_pairs": n_valid_pairs,
            "n_seed_4d": self.n_seed4d,
            "n_R_4d": n_R,
            "n_grown_4d": int(grown),
            "n_omega3d": int(n_omega3d), "npsi": int(self.npsi),
            "rho_seed": rho_hist[0],
            "rho_inf": n_R / denom,
            "rho_hist": rho_hist,
            "wave_added": wave_added,
            "n_waves": len(wave_added),
            "max_layer": int(self.layer.max()),
            "layer_hist": self._layer_hist(),
            "n_not_added": int(not_added.sum()),
            "blocked_breakdown": reach_blocked,
            "_cand": (ci, cj, ck, cl), "_added": added, "_store": store,
        }

    # ------------------------------------------------------------------ #
    def _layer_hist(self):
        ls = self.layer[self.layer >= 0]
        if ls.size == 0:
            return {}
        u, c = np.unique(ls, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    def _blocked_breakdown(self, ci, cj, ck, cl, store, not_added):
        """For candidates never added, classify the LEAST-bad blocking reason
        across the menu: 'no_valid_action' (every action leaves K n V_0 or domain
        = conservatism / boundary) vs 'in_region_blocked' (some valid action
        exists but its successor still touches a non-R cell inside K n V_0 = the
        capture-basin frontier the lfp could not reach)."""
        nb = int(not_added.sum())
        if nb == 0:
            return {"no_valid_action": 0, "in_region_blocked": 0}
        has_valid = np.zeros(len(ci), bool)
        for s in store:
            has_valid |= s["valid"]
        no_valid = int((not_added & ~has_valid).sum())
        in_region = int((not_added & has_valid).sum())
        return {"no_valid_action": no_valid, "in_region_blocked": in_region}
