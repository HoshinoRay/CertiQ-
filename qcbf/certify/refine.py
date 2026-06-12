"""Mode-B certificate: accepted-set-independent precompute + C2 fixed point.

The certified set Omega_cert(c) is the GREATEST subset A of the candidate set

    Cand(c) = { cells :  C1  and  C3  and  lbV >= max(c, 0) }

that is closed under the verified transition relation:

    (C2)  for every cell i in A and every u-cell j that is NOT skipped by the
          antecedent test, the interval successor over-approximation of
          cell_i x u-cell_j x D lies inside  union(A)  and inside the domain.

By Tarski / Prop T4 the greatest such A is obtained by pruning from Cand(c)
until no cell violates C2; the operator is monotone, so the iteration
converges and yields the unique greatest fixed point.

Everything that does not depend on the accepted set -- C1, C3, lbV, the skip
mask and the successor index ranges -- is computed ONCE; a c-sweep then costs
only the (cheap, prefix-sum based) fixed-point iterations per c.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qcbf.config import ExperimentConfig
from qcbf.certify.cells import (CellLattice, acceptance_prefix_sum,
                                ranges_fully_accepted)
from qcbf.dynamics.dubins import successor_boxes, g_bounds_on_box
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.verify.conditions import (antecedent_skip, check_c1,
                                    check_c3_staged, v_cell_bounds)


# --------------------------------------------------------------------------- #
@dataclass
class CertPrecompute:
    lattice: CellLattice
    boxes: np.ndarray          # (N, 6)
    c1_ok: np.ndarray          # (N,)  bool
    lbV: np.ndarray            # (N,)  certified lower bound of V_theta
    ubV: np.ndarray            # (N,)
    c3_ok: np.ndarray          # (N,)  bool
    c3_lb: np.ndarray          # (N,)  certified lower bound of h3 on cell x D
    cand: np.ndarray           # (M,)  candidate cell ids (C1 & C3 & lbV>=0)
    skip: np.ndarray           # (M, n_uc) bool antecedent-vacuous pairs
    succ_rng: np.ndarray       # (M, n_uc, S, 6) int16 index ranges
    succ_active: np.ndarray    # (M, n_uc, S) bool
    succ_dom_ok: np.ndarray    # (M, n_uc, S) bool
    funnel: dict
    wall_s: dict


# --------------------------------------------------------------------------- #
def _successor_ranges(pre_lat: CellLattice, boxes: np.ndarray,
                      cand: np.ndarray, cfg: ExperimentConfig,
                      verbose: bool = True, successor_boxes_fn=successor_boxes):
    """Interval successor boxes -> conservative lattice index ranges.

    For every candidate cell, every u-cell, every (u, d) sub-split and the
    optional wrap-split, compute the successor box of
    cell x u-subcell x d-subcell and map it to inclusive lattice index ranges
    (psi on the doubled axis).  Sub-splitting in (u, d) only tightens the
    over-approximation; the union over sub-boxes still covers the full
    product, so C2 enforced per sub-box is sound and less conservative.
    """
    dyn, cert = cfg.dynamics, cfg.cert
    M = len(cand)
    n_uc = cert.n_u_cells
    su, sd = cert.c2_u_subsplit, cert.c2_d_subsplit
    S = su * sd * 2                     # x2: wrap-split secondary boxes
    u_edges = np.linspace(-dyn.control_max, dyn.control_max, n_uc * su + 1)
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, sd + 1)

    rng = np.zeros((M, n_uc, S, 6), dtype=np.int16)
    active = np.zeros((M, n_uc, S), dtype=bool)
    dom_ok = np.zeros((M, n_uc, S), dtype=bool)
    cb = boxes[cand]
    t0 = time.time()
    for j in range(n_uc):
        s = 0
        for a in range(su):
            ue = (u_edges[j * su + a], u_edges[j * su + a + 1])
            for b in range(sd):
                de = (d_edges[b], d_edges[b + 1])
                b1, b2, m2 = successor_boxes_fn(
                    dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3],
                    cb[:, 4], cb[:, 5], ue[0], ue[1], de[0], de[1])
                r1, d1 = pre_lat.box_index_ranges(b1)
                r2, d2 = pre_lat.box_index_ranges(b2)
                rng[:, j, s] = r1
                active[:, j, s] = True
                dom_ok[:, j, s] = d1
                rng[:, j, s + 1] = r2
                active[:, j, s + 1] = m2
                dom_ok[:, j, s + 1] = d2
                s += 2
    if verbose:
        n_box = int(active.sum())
        print(f"  [C2 ] successor ranges: {n_box} active boxes over "
              f"{M}x{n_uc} pairs  ({time.time()-t0:.1f}s)")
    return rng, active, dom_ok


# --------------------------------------------------------------------------- #
def precompute_certificate(cfg: ExperimentConfig, v_net: SeqNet,
                           q_net: SeqNet, h3_net: SeqNet,
                           verbose: bool = True,
                           successor_boxes_fn=successor_boxes) -> CertPrecompute:
    dyn, cert = cfg.dynamics, cfg.cert
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    N = lat.n_cells
    wall: dict[str, float] = {}
    funnel: dict[str, int] = {"n_cells": N}

    # ---- C1 (exact) ------------------------------------------------------- #
    t0 = time.time()
    c1_ok = check_c1(dyn, boxes)
    wall["c1"] = time.time() - t0
    funnel["c1_pass"] = int(c1_ok.sum())
    if verbose:
        print(f"  [C1 ] exact g-bounds        pass {c1_ok.sum():6d}/{N}"
              f"  ({wall['c1']:.1f}s)")

    # ---- certified V bounds on C1-passing cells --------------------------- #
    t0 = time.time()
    idx1 = np.flatnonzero(c1_ok)
    lbV, ubV = v_cell_bounds(v_net, boxes, idx1, N, cert.chunk)
    wall["v_bounds"] = time.time() - t0
    base = c1_ok & (lbV >= 0.0)
    funnel["lbV_ge0_pass"] = int(base.sum())
    if verbose:
        print(f"  [V  ] certified lbV >= 0    pass {base.sum():6d}/"
              f"{len(idx1)}  ({wall['v_bounds']:.1f}s)")

    # ---- C3 on base cells (staged) ---------------------------------------- #
    t0 = time.time()
    idx3 = np.flatnonzero(base)
    c3_ok, c3_lb, f3 = check_c3_staged(h3_net, boxes, idx3, dyn, cert, N,
                                       verbose)
    wall["c3"] = time.time() - t0
    funnel.update({f"c3_{k}": v for k, v in f3.items()})
    funnel["c3_pass"] = int(c3_ok.sum())

    cand = np.flatnonzero(base & c3_ok)
    funnel["candidates"] = int(len(cand))
    if verbose:
        print(f"  [C3 ] total                 pass {len(cand):6d}/"
              f"{len(idx3)}  ({wall['c3']:.1f}s)")

    # ---- antecedent skip test --------------------------------------------- #
    t0 = time.time()
    skip = antecedent_skip(q_net, boxes, cand, lbV, dyn, cert,
                           cfg.train.gamma, verbose)
    wall["skip"] = time.time() - t0
    funnel["skip_pairs"] = int(skip.sum())
    funnel["total_pairs"] = int(skip.size)

    # ---- successor index ranges ------------------------------------------- #
    t0 = time.time()
    succ_rng, succ_act, succ_dom = _successor_ranges(lat, boxes, cand, cfg,
                                                     verbose, successor_boxes_fn)
    wall["succ"] = time.time() - t0

    return CertPrecompute(lat, boxes, c1_ok, lbV, ubV, c3_ok, c3_lb, cand,
                          skip, succ_rng, succ_act, succ_dom, funnel, wall)


# --------------------------------------------------------------------------- #
def c2_fixed_point(pre: CertPrecompute, c: float,
                   verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Greatest fixed point of the C2-pruning operator inside Cand(c).

    Returns (accepted (N,) bool, stats dict).
    """
    lat = pre.lattice
    t0 = time.time()
    init = np.zeros(lat.n_cells, dtype=bool)
    cand_ok = pre.lbV[pre.cand] >= max(c, 0.0)
    init[pre.cand[cand_ok]] = True

    M = len(pre.cand)
    rng_flat = pre.succ_rng.reshape(-1, 6)
    act_flat = pre.succ_active.reshape(M, -1)
    dom_flat = pre.succ_dom_ok.reshape(M, -1)
    n_uc = pre.skip.shape[1]
    S = pre.succ_active.shape[2]

    # a pair whose ANY active sub-box leaves the domain can never be closed
    dom_pair_ok = (dom_flat | ~act_flat).reshape(M, n_uc, S).all(axis=2)
    pair_required = ~pre.skip                       # (M, n_uc)
    hard_fail = (pair_required & ~dom_pair_ok).any(axis=1)

    accepted = init.copy()
    accepted[pre.cand[hard_fail & cand_ok]] = False
    history = [int(accepted.sum())]
    it = 0
    while True:
        it += 1
        P = acceptance_prefix_sum(lat, accepted)
        ok_box = ranges_fully_accepted(P, rng_flat).reshape(M, n_uc, S)
        ok_box |= ~pre.succ_active                  # inactive boxes vacuous
        ok_pair = ok_box.all(axis=2)                # (M, n_uc)
        ok_cell = (ok_pair | pre.skip).all(axis=1)  # (M,)
        new = accepted.copy()
        new[pre.cand] &= ok_cell
        changed = int(accepted.sum() - new.sum())
        accepted = new
        history.append(int(accepted.sum()))
        if changed == 0:
            break
    stats = {"c": float(c), "init": history[0], "accepted": history[-1],
             "iters": it, "history": history,
             "wall_s": time.time() - t0}
    if verbose:
        print(f"  [FP ] c={c:5.2f}  init {history[0]:6d} -> accepted "
              f"{history[-1]:6d}  in {it} iters ({stats['wall_s']:.1f}s)")
    return accepted, stats


# --------------------------------------------------------------------------- #
# Fallback-only certificate (sub-level-set / CBF route; no antecedent skip).
#
# For a CBF-trained artifact the deployed safety policy is the certified
# fallback u = clip(pi(x)) alone, and the certified claim is that {accepted} is
# robustly invariant *under the fallback*.  C2 then only has to cover the
# fallback's robust successor box -- one action range per cell -- so it does not
# need the antecedent skip (which cannot fire for these plants; see
# DESIGN_REVIEW / dubins_e0_results_and_ablation.md).  Soundness: the runtime
# applies u = clip(pi(x)) in [clip(lbU), clip(ubU)] over the cell, and the
# successor box over that range x D is verified to lie inside {accepted}.
# --------------------------------------------------------------------------- #
def fallback_successor_ranges(lat: CellLattice, boxes: np.ndarray,
                              cand: np.ndarray, pi_net: SeqNet,
                              dyn, cert, successor_boxes_fn=successor_boxes):
    """Robust successor index ranges of the fallback action clip(pi(cell)) x D."""
    cb = boxes[cand]
    M = len(cand)
    lo = cb[:, [0, 2, 4]]
    hi = cb[:, [1, 3, 5]]
    ulb, uub = crown_bounds_chunked(pi_net, lo, hi, True, cert.chunk)
    cm = dyn.control_max
    u_lo = np.clip(ulb[:, 0], -cm, cm)             # sound action range per cell
    u_hi = np.clip(uub[:, 0], -cm, cm)
    sd = cert.c2_d_subsplit
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, sd + 1)
    S = sd * 2                                     # x2 for the heading wrap-split
    rng = np.zeros((M, S, 6), dtype=np.int16)
    active = np.zeros((M, S), dtype=bool)
    dom = np.zeros((M, S), dtype=bool)
    s = 0
    for b in range(sd):
        b1, b2, m2 = successor_boxes_fn(
            dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3], cb[:, 4], cb[:, 5],
            u_lo, u_hi, d_edges[b], d_edges[b + 1])
        r1, d1 = lat.box_index_ranges(b1)
        r2, d2 = lat.box_index_ranges(b2)
        rng[:, s] = r1; active[:, s] = True; dom[:, s] = d1
        rng[:, s + 1] = r2; active[:, s + 1] = m2; dom[:, s + 1] = d2
        s += 2
    return rng, active, dom, (u_lo, u_hi)


def c2_fallback_fixed_point(lat: CellLattice, cand: np.ndarray, lbV: np.ndarray,
                            fb_rng: np.ndarray, fb_active: np.ndarray,
                            fb_dom: np.ndarray, c: float, verbose: bool = True
                            ) -> tuple[np.ndarray, dict]:
    """Greatest subset of Cand(c) closed under the fallback successor map."""
    t0 = time.time()
    init = np.zeros(lat.n_cells, dtype=bool)
    cand_ok = lbV[cand] >= max(c, 0.0)
    init[cand[cand_ok]] = True
    M = len(cand)
    S = fb_active.shape[1]
    rng_flat = fb_rng.reshape(-1, 6)
    dom_ok = (fb_dom | ~fb_active).all(axis=1)     # all active sub-boxes in domain
    hard_fail = ~dom_ok
    accepted = init.copy()
    accepted[cand[hard_fail & cand_ok]] = False
    history = [int(accepted.sum())]
    it = 0
    while True:
        it += 1
        P = acceptance_prefix_sum(lat, accepted)
        ok_box = ranges_fully_accepted(P, rng_flat).reshape(M, S)
        ok_box |= ~fb_active
        ok_cell = ok_box.all(axis=1)
        new = accepted.copy()
        new[cand] &= ok_cell
        changed = int(accepted.sum() - new.sum())
        accepted = new
        history.append(int(accepted.sum()))
        if changed == 0:
            break
    stats = {"c": float(c), "init": history[0], "accepted": history[-1],
             "iters": it, "history": history, "wall_s": time.time() - t0}
    if verbose:
        print(f"  [FB-FP] c={c:5.2f}  init {history[0]:6d} -> accepted "
              f"{history[-1]:6d}  in {it} iters ({stats['wall_s']:.1f}s)")
    return accepted, stats


# --------------------------------------------------------------------------- #
# Direct sub-level-set CBF certificate (no skip, no cascade) -- the cleanest
# route for a CBF-trained artifact (qcbf.nets.mlp.finetune_cbf).
#
# Certifies {V_theta >= c} robustly forward-invariant under the fallback
# u = clip(pi(x)) and safe, by a ONE-PASS check: for every cell that intersects
# {V>=c} (ubV >= c), the box is safe (gmin >= 0) and the CROWN *lower* bound of
# V over the fallback successor box is >= c (the successor stays in {V>=c}).
# No cell-reachability fixed point (so no boundary cascade), no antecedent skip,
# no Q, no C3.  Soundness: for x with V(x)>=c the fallback successor lies in the
# (sound) successor box, on which V is verified >= c, so {V>=c} is invariant;
# runtime membership is the exact test V(x) >= c.
#
# Applicability: this closes only when the plant can *hold or raise* V on the
# invariant set.  A FIXED-SPEED plant's maximal invariant sets are constant-V
# orbits, leaving no margin for the successor-box undershoot at the {V>=c}
# boundary, so it returns best_c=None (the cell+box-vs-fixed-speed obstruction;
# see docs/dubins_e0_results_and_ablation.md).  A variable-speed / brakeable
# plant contracts and is certifiable.
# --------------------------------------------------------------------------- #
def certify_sublevel_invariant(lat: CellLattice, boxes: np.ndarray,
                               v_net: SeqNet, pi_net: SeqNet, dyn, cert,
                               c_grid=None, successor_boxes_fn=successor_boxes,
                               verbose: bool = True):
    """Largest certifiable safe-invariant sub-level set {V_theta >= c}.

    Returns (accepted mask (N,) bool for {lbV>=c}, info dict with best_c,
    accepted count, and the median successor-box undershoot lbV-lbVs).
    """
    N = lat.n_cells
    ch = cert.chunk
    lbV, ubV = v_cell_bounds(v_net, boxes, np.arange(N), N, ch)
    gmin, _ = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1],
                              boxes[:, 2], boxes[:, 3])
    work = np.flatnonzero(ubV >= 0.0)
    cb = boxes[work]
    ulb, uub = crown_bounds_chunked(pi_net, cb[:, [0, 2, 4]],
                                    cb[:, [1, 3, 5]], True, ch)
    cm = dyn.control_max
    u_lo = np.clip(ulb[:, 0], -cm, cm)
    u_hi = np.clip(uub[:, 0], -cm, cm)
    sd = cert.c2_d_subsplit
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, sd + 1)
    lbVs = np.full(N, -np.inf)
    w = np.full(len(work), np.inf)
    for b in range(sd):
        b1, b2, m2 = successor_boxes_fn(
            dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3], cb[:, 4], cb[:, 5],
            u_lo, u_hi, d_edges[b], d_edges[b + 1])
        l1, _ = crown_bounds_chunked(v_net, b1[:, [0, 2, 4]], b1[:, [1, 3, 5]],
                                     True, ch)
        w = np.minimum(w, l1[:, 0])
        l2, _ = crown_bounds_chunked(v_net, b2[:, [0, 2, 4]], b2[:, [1, 3, 5]],
                                     True, ch)
        w = np.minimum(w, np.where(m2, l2[:, 0], np.inf))
    lbVs[work] = w
    if c_grid is None:
        c_grid = np.linspace(0.0, float(np.max(ubV[work])) if len(work) else 1.0, 41)
    best = None
    for c in c_grid:                                   # smallest valid c
        inset = ubV >= c
        if np.all(gmin[inset] >= 0.0) and np.all(lbVs[inset] >= c) \
                and int(np.sum(lbV >= c)) > 0:
            best = float(c)
            break
    mask = (lbV >= best) if best is not None else np.zeros(N, dtype=bool)
    info = {"best_c": best, "accepted": int(mask.sum()),
            "n_sublevel0": int(len(work)),
            "undershoot_med": float(np.median((lbV - lbVs)[work])) if len(work) else float("nan")}
    if verbose:
        print(f"  [SUB] sub-level cert: best c={best}  accepted "
              f"{info['accepted']}  (undershoot med {info['undershoot_med']:.3f})")
    return mask, info
