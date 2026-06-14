"""Strict verifier for the deployed learned Q-CBF specification (Theorem A).

The deployed filter applies, at each state, either a finite-menu action that
passes the robust Q-gate  min_d Q_theta(x,u,d) >= gamma_deploy V_theta(x)
or the frozen fallback witness u = pi_theta(x).  The certificate checks exactly
the conditions Theorem A needs for that loop on the superlevel set
{V_theta >= 0}:

    C1: {x : V_theta(x) >= 0} subset K.                        (safety floor)
    C3: witness liveness with u = pi_theta(x):                 (feasibility)
            min_d Q_theta(x, pi_theta(x), d) >= gamma_deploy V_theta(x) + eps.
    C4: Q-gate soundness on every action the loop can apply
        (the finite menu and the fallback witness):            (gate => decrease)
            min_d Q_theta(x,u,d) <= min_d V_theta(f(x,u,d)).

The robust one-step decrease  V_theta(f(x,u,d)) >= gamma_deploy V_theta(x)  is
NOT a separate proof obligation: any action the runtime can apply has, by its
own gate, min_d Q_theta >= gamma_deploy V_theta, and C4 then gives
min_d V_theta(f) >= min_d Q_theta >= gamma_deploy V_theta >= 0, so {V_theta>=0}
is robustly forward-invariant and (with C1) safe.  This is why the old
existential C2 menu-decrease check is dropped -- C3 (feasibility) and C4
(gate => decrease) subsume it.

All checks are conservative cell proofs.  A failed check means only that this
verifier did not certify the condition; it is not a counterexample by itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import g_bounds_on_box, successor_boxes
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.verify.conditions import check_c3_staged, v_cell_bounds


@dataclass
class StrictSpecResult:
    lattice: CellLattice
    boxes: np.ndarray
    lbV: np.ndarray
    ubV: np.ndarray
    gmin: np.ndarray
    superlevel_possible: np.ndarray
    superlevel_inner: np.ndarray
    c1_bad: np.ndarray
    c3_ok: np.ndarray
    c3_lb: np.ndarray
    q_consistency_ok: np.ndarray
    q_consistency_margin: np.ndarray
    witness_q_consistency_ok: np.ndarray
    witness_q_consistency_margin: np.ndarray
    accepted: np.ndarray
    report: dict


def _state_lo_hi(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]


def _box_in_domain(dyn, box: np.ndarray) -> np.ndarray:
    return ((box[:, 0] >= dyn.p_lo - 1e-12) & (box[:, 1] <= dyn.p_hi + 1e-12)
            & (box[:, 2] >= dyn.p_lo - 1e-12) & (box[:, 3] <= dyn.p_hi + 1e-12))


def _menu(dyn, cert) -> np.ndarray:
    """Finite action menu used for the existential C2 witness."""
    if cert.n_u_cells <= 1:
        return np.array([0.0])
    return np.linspace(-dyn.control_max, dyn.control_max, cert.n_u_cells)


def _control_range(pol_net: SeqNet, cb: np.ndarray, dyn, cert
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell sound range [u_lo, u_hi] of the deployed control u = clip(pi(x))
    over each cell, via CROWN on the compiled (exactly piecewise-linear) policy
    network.  This is the tight enclosure the witness C4 check uses instead of
    the full admissible interval [-omega_max, omega_max] -- the deployed loop
    only ever applies clip(pi(x)), so enclosing its actual per-cell range is
    both sound and far tighter (DEVELOPMENT_LOG W1)."""
    slo, shi = cb[:, [0, 2, 4]], cb[:, [1, 3, 5]]
    ulo, uhi = crown_bounds_chunked(pol_net, slo, shi, True, cert.chunk)
    return (np.clip(ulo[:, 0], -dyn.control_max, dyn.control_max),
            np.clip(uhi[:, 0], -dyn.control_max, dyn.control_max))


def _succ_v_lower(v_net: SeqNet, cb: np.ndarray, dyn, cert,
                  u_lo, u_hi) -> np.ndarray:
    """Sound lower bound of  min_d V_theta(f(C, u, d))  over each state cell C,
    for control u in [u_lo, u_hi] and disturbance d in D.

    u_lo, u_hi may be a scalar (a fixed menu action) or (N,) arrays (the
    witness per-cell control range).  Each cell's heading is sub-split
    ``c4_psi_subsplit`` ways and the disturbance ``c2_d_subsplit`` ways; both
    only tighten the cos/sin and disturbance enclosures, so the result is sound
    and >= the un-split bound (sub-splitting never loosens)."""
    N = len(cb)
    if N == 0:
        return np.zeros(0)
    u_lo = np.broadcast_to(np.asarray(u_lo, float), (N,))
    u_hi = np.broadcast_to(np.asarray(u_hi, float), (N,))
    npsi = max(1, cert.c4_psi_subsplit)
    nd = max(1, cert.c2_d_subsplit)
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, nd + 1)
    psi_lo, psi_hi = cb[:, 4], cb[:, 5]
    worst = np.full(N, np.inf)
    for jp in range(npsi):
        pl = psi_lo + (psi_hi - psi_lo) * (jp / npsi)
        ph = psi_lo + (psi_hi - psi_lo) * ((jp + 1) / npsi)
        for kd in range(nd):
            b1, b2, m2 = successor_boxes(
                dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3], pl, ph,
                u_lo, u_hi, d_edges[kd], d_edges[kd + 1])
            dom1 = _box_in_domain(dyn, b1)
            lo1, hi1 = _state_lo_hi(b1)
            lb1, _ = crown_bounds_chunked(v_net, lo1, hi1, True, cert.chunk)
            val = np.where(dom1, lb1[:, 0], -np.inf)
            if np.any(m2):
                dom2 = _box_in_domain(dyn, b2)
                lo2, hi2 = _state_lo_hi(b2)
                lb2, _ = crown_bounds_chunked(v_net, lo2, hi2, True, cert.chunk)
                val = np.minimum(val, np.where(m2 & dom2, lb2[:, 0],
                                               np.where(m2, -np.inf, np.inf)))
            worst = np.minimum(worst, val)
    return worst


def _q_min_upper(q_net: SeqNet, boxes: np.ndarray, idx: np.ndarray,
                 dyn, cert, u: float) -> np.ndarray:
    """Sound upper bound of min_d Q_theta(C,u,d) using fixed d probes."""
    cb = boxes[idx]
    if len(cb) == 0:
        return np.zeros(0)
    probes = (np.linspace(-dyn.d_max, dyn.d_max, cert.ante_d_probes)
              if cert.ante_d_probes > 1 else np.array([0.0]))
    out = np.full(len(cb), np.inf)
    for d in probes:
        lo = np.column_stack([cb[:, 0], cb[:, 2], cb[:, 4],
                              np.full(len(cb), float(u)),
                              np.full(len(cb), float(d))])
        hi = np.column_stack([cb[:, 1], cb[:, 3], cb[:, 5],
                              np.full(len(cb), float(u)),
                              np.full(len(cb), float(d))])
        _, qhi = crown_bounds_chunked(q_net, lo, hi, True, cert.chunk)
        out = np.minimum(out, qhi[:, 0])
    return out


def _witness_q_upper(h3_net: SeqNet, boxes: np.ndarray, idx: np.ndarray,
                     dyn, cert, gamma: float, ubV: np.ndarray,
                     eps: float) -> np.ndarray:
    """Upper bound min_d Q_theta(C, pi_theta(C), d) for the witness.

    The compiled h3 network is

        Q_theta(x, pi_theta(x), d) - gamma V_theta(x) - eps.

    A fixed d probe gives an upper bound on the disturbance minimum.  Adding
    gamma * ubV(C) + eps yields a sound upper bound on min_d Q for the witness.
    """
    cb = boxes[idx]
    if len(cb) == 0:
        return np.zeros(0)
    probes = (np.linspace(-dyn.d_max, dyn.d_max, cert.ante_d_probes)
              if cert.ante_d_probes > 1 else np.array([0.0]))
    out = np.full(len(cb), np.inf)
    for d in probes:
        lo = np.column_stack([cb[:, 0], cb[:, 2], cb[:, 4],
                              np.full(len(cb), float(d))])
        hi = np.column_stack([cb[:, 1], cb[:, 3], cb[:, 5],
                              np.full(len(cb), float(d))])
        _, hhi = crown_bounds_chunked(h3_net, lo, hi, True, cert.chunk)
        out = np.minimum(out, hhi[:, 0] + gamma * ubV[idx] + eps)
    return out


def _check_q_consistency(v_net: SeqNet, q_net: SeqNet, boxes: np.ndarray,
                         idx: np.ndarray, cfg: ExperimentConfig,
                         menu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C4 on every finite-menu action for each active cell."""
    cb = boxes[idx]
    ok = np.ones(len(idx), dtype=bool)
    worst_margin = np.full(len(idx), np.inf)
    for u in menu:
        ub_qmin = _q_min_upper(q_net, boxes, idx, cfg.dynamics, cfg.cert,
                               float(u))
        lb_vsucc = _succ_v_lower(v_net, cb, cfg.dynamics, cfg.cert,
                                 float(u), float(u))
        margin = lb_vsucc - ub_qmin
        worst_margin = np.minimum(worst_margin, margin)
        ok &= margin >= 0.0
    return ok, worst_margin


def _check_witness_q_consistency(v_net: SeqNet, h3_net: SeqNet, pol_net: SeqNet,
                                 boxes: np.ndarray, idx: np.ndarray,
                                 ubV: np.ndarray, cfg: ExperimentConfig
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """C4 for the deployed fallback witness u = clip(pi_theta(x)).

    The witness action is enclosed by its tight per-cell range [u_lo, u_hi]
    from the compiled policy (not the full control interval): the deployed loop
    only ever applies clip(pi(x))."""
    cb = boxes[idx]
    ub_qmin = _witness_q_upper(h3_net, boxes, idx, cfg.dynamics, cfg.cert,
                               cfg.train.gamma_deploy, ubV, cfg.cert.eps_margin)
    if len(cb) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    u_lo, u_hi = _control_range(pol_net, cb, cfg.dynamics, cfg.cert)
    lb_vsucc = _succ_v_lower(v_net, cb, cfg.dynamics, cfg.cert, u_lo, u_hi)
    margin = lb_vsucc - ub_qmin
    return margin >= 0.0, margin


def run_strict_spec_certificate(cfg: ExperimentConfig, v_net: SeqNet,
                                q_net: SeqNet, h3_net: SeqNet, pol_net: SeqNet,
                                verbose: bool = True) -> StrictSpecResult:
    """Run C1, C3, C4 (Theorem A) on the frozen artifact over {V_theta >= 0}."""
    t_all = time.time()
    dyn, cert = cfg.dynamics, cfg.cert
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    n = lat.n_cells
    wall: dict[str, float] = {}

    t0 = time.time()
    all_idx = np.arange(n)
    lbV, ubV = v_cell_bounds(v_net, boxes, all_idx, n, cert.chunk)
    wall["v_bounds"] = time.time() - t0

    t0 = time.time()
    gmin, _ = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1],
                              boxes[:, 2], boxes[:, 3])
    superlevel_possible = ubV >= 0.0
    superlevel_inner = lbV >= 0.0
    c1_bad = superlevel_possible & (gmin < 0.0)
    wall["c1"] = time.time() - t0
    if verbose:
        print(f"  [C1 ] {{V>=0}}=>K possible cells {int(superlevel_possible.sum())}, "
              f"bad {int(c1_bad.sum())}")

    active_idx = np.flatnonzero(superlevel_possible & ~c1_bad)
    menu = _menu(dyn, cert)

    if np.any(c1_bad):
        if verbose:
            print("  [FAIL] C1 already failed; skipping C3/C4 because the "
                  "strict spec cannot pass.")
        c3_ok = np.zeros(n, dtype=bool)
        c3_lb = np.full(n, -np.inf)
        q_consistency_ok = np.zeros(n, dtype=bool)
        q_consistency_margin = np.full(n, -np.inf)
        witness_q_consistency_ok = np.zeros(n, dtype=bool)
        witness_q_consistency_margin = np.full(n, -np.inf)
        accepted = np.zeros(n, dtype=bool)
        report = {
            "spec_pass": False,
            "fail_fast": "c1",
            "n_cells": int(n),
            "menu": [float(u) for u in menu],
            "superlevel_possible": int(superlevel_possible.sum()),
            "superlevel_inner": int(superlevel_inner.sum()),
            "c1_bad": int(c1_bad.sum()),
            "c3": {
                "checked": 0,
                "stageA_pass": 0,
                "stageB_pass": 0,
                "stageC_pass": 0,
                "pass_on_active": 0,
                "active": int(len(active_idx)),
                "skipped": True,
            },
            "q_consistency": {
                "pass_on_active": 0,
                "active": int(len(active_idx)),
                "min_margin": None,
                "skipped": True,
            },
            "witness_q_consistency": {
                "pass_on_active": 0,
                "active": int(len(active_idx)),
                "min_margin": None,
                "skipped": True,
            },
            "accepted_cells": 0,
            "wall_s_stages": {k: round(v, 1) for k, v in wall.items()},
            "wall_s_total": round(time.time() - t_all, 1),
        }
        return StrictSpecResult(
            lattice=lat, boxes=boxes, lbV=lbV, ubV=ubV, gmin=gmin,
            superlevel_possible=superlevel_possible,
            superlevel_inner=superlevel_inner, c1_bad=c1_bad,
            c3_ok=c3_ok, c3_lb=c3_lb,
            q_consistency_ok=q_consistency_ok,
            q_consistency_margin=q_consistency_margin,
            witness_q_consistency_ok=witness_q_consistency_ok,
            witness_q_consistency_margin=witness_q_consistency_margin,
            accepted=accepted, report=report,
        )

    t0 = time.time()
    c3_ok, c3_lb, c3_report = check_c3_staged(h3_net, boxes, active_idx,
                                              dyn, cert, n, verbose)
    wall["c3"] = time.time() - t0

    t0 = time.time()
    q_local, q_margin_local = _check_q_consistency(
        v_net, q_net, boxes, active_idx, cfg, menu)
    q_consistency_ok = np.zeros(n, dtype=bool)
    q_consistency_margin = np.full(n, -np.inf)
    q_consistency_ok[active_idx] = q_local
    q_consistency_margin[active_idx] = q_margin_local
    wall["q_consistency"] = time.time() - t0
    if verbose:
        print(f"  [C4m] menu Q<=V(f)          pass {int(q_local.sum()):6d}/"
              f"{len(active_idx)}")

    t0 = time.time()
    wq_local, wq_margin_local = _check_witness_q_consistency(
        v_net, h3_net, pol_net, boxes, active_idx, ubV, cfg)
    witness_q_consistency_ok = np.zeros(n, dtype=bool)
    witness_q_consistency_margin = np.full(n, -np.inf)
    witness_q_consistency_ok[active_idx] = wq_local
    witness_q_consistency_margin[active_idx] = wq_margin_local
    wall["witness_q_consistency"] = time.time() - t0
    if verbose:
        print(f"  [C4w] witness Q<=V(f)       pass {int(wq_local.sum()):6d}/"
              f"{len(active_idx)}")

    all_active_ok = (
        not np.any(c1_bad)
        and np.all(c3_ok[active_idx])
        and np.all(q_consistency_ok[active_idx])
        and np.all(witness_q_consistency_ok[active_idx])
    )
    accepted = superlevel_inner.copy() if all_active_ok else np.zeros(n, dtype=bool)

    report = {
        "spec_pass": bool(all_active_ok),
        "n_cells": int(n),
        "menu": [float(u) for u in menu],
        "superlevel_possible": int(superlevel_possible.sum()),
        "superlevel_inner": int(superlevel_inner.sum()),
        "c1_bad": int(c1_bad.sum()),
        "c3": {
            **{k: int(v) for k, v in c3_report.items()},
            "pass_on_active": int(np.sum(c3_ok[active_idx])),
            "active": int(len(active_idx)),
        },
        "q_consistency": {
            "pass_on_active": int(np.sum(q_consistency_ok[active_idx])),
            "active": int(len(active_idx)),
            "min_margin": float(np.min(q_consistency_margin[active_idx])) if len(active_idx) else float("nan"),
        },
        "witness_q_consistency": {
            "pass_on_active": int(np.sum(witness_q_consistency_ok[active_idx])),
            "active": int(len(active_idx)),
            "min_margin": float(np.min(witness_q_consistency_margin[active_idx])) if len(active_idx) else float("nan"),
        },
        "accepted_cells": int(accepted.sum()),
        "wall_s_stages": {k: round(v, 1) for k, v in wall.items()},
        "wall_s_total": round(time.time() - t_all, 1),
    }

    return StrictSpecResult(
        lattice=lat, boxes=boxes, lbV=lbV, ubV=ubV, gmin=gmin,
        superlevel_possible=superlevel_possible,
        superlevel_inner=superlevel_inner, c1_bad=c1_bad,
        c3_ok=c3_ok, c3_lb=c3_lb,
        q_consistency_ok=q_consistency_ok,
        q_consistency_margin=q_consistency_margin,
        witness_q_consistency_ok=witness_q_consistency_ok,
        witness_q_consistency_margin=witness_q_consistency_margin,
        accepted=accepted, report=report,
    )
