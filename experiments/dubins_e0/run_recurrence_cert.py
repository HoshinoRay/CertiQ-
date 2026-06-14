"""Recurrence (CBF-superlevel) certificate for the deployed learned artifact.

This is Theorem A in its *recurrence* form -- a single barrier function and a
single value condition, no Q-gate, no GFP.  Let

    W(x) = min( g(x), V_theta(x) )          (the safety clamp; {W>=0} subset K)

and let the deployed witness policy be  u = clip(pi_theta(x)).  For a level m,

    S_m = { x : W(x) >= m }.

The certificate condition checked here is, for every cell that can contain a
point of S_m (every "active" cell, ub_W(C) >= m):

    lb_W( f(C, [u_lo,u_hi], D) ) >= m,        (*)

a SOUND lower bound on W over the cell-worst successor box under the witness's
own control range [u_lo,u_hi] and the full disturbance D.  Because S_m is the
m-superlevel set of the SAME W, (*) says "the successor's W >= m", i.e. the
successor is back in S_m.  So:

    if (*) holds on ALL active cells  ==>  S_m is forward-invariant under the
    witness, and S_m subset {g >= m} subset K  (for m >= 0).

That is the whole proof -- no fixed-point iteration.  A viability-kernel GFP is
neither needed nor appropriate: it would discard W at the successor and round
the successor box up to whole cells, which is strictly looser.

Soundness of every step: the control range [u_lo,u_hi] is the CROWN enclosure
of clip(pi) over the cell (the deployed loop only applies clip(pi)); the
successor box uses the exact cos/sin and disturbance extrema (the trusted
successor_boxes map); lb_W = min(lb_g, lb_V) with the closed-form g bound and a
CROWN bound on V; the heading seam is split into two boxes.  None of it can
certify a false property -- looseness only shrinks the certified set.

Reporting.  The decisive distinction (this is the point of the script):

  * pass_frac(m)  = (active cells satisfying (*)) / (active cells).  This is a
    PASS RATE.  By itself it certifies NOTHING unless it is 1.0 -- a partial
    pass gives no invariant set, because a passing cell's successor may land in
    a failing cell.
  * rho_inner(m)  = Vol({lb_W >= m} cells) / Vol(Omega*).  This is the certified
    SET SIZE.  It is a valid certificate ONLY at levels m where pass_frac == 1.

We sweep m and report, for each level, both numbers, plus the smallest m at
which pass_frac reaches 1.0 (the largest COMPLETE certified S_m).  That answers
"what does the 0.64 mean": if pass_frac < 1 at m=0, then 0.64 was a pass rate,
not a certified volume, and the complete certificate is the S_{m*} at the level
where the whole superlevel set closes.
"""
from __future__ import annotations

import argparse
import time

from common import REPO, file_hash, save_json

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.spec import _control_range
from qcbf.certify.volume import omega_star_volume
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import g_bounds_on_box, successor_boxes
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.verify.compiler import compile_policy
from qcbf.verify.conditions import v_cell_bounds


def _succ_W_lower(v_net: SeqNet, cb: np.ndarray, dyn, cert,
                  u_lo: np.ndarray, u_hi: np.ndarray) -> np.ndarray:
    """Sound lower bound of  min_d W(f(C, u, d)),  W = min(g, V_theta),  over
    each state cell C, for the witness control range u in [u_lo, u_hi] and
    disturbance d in D.

    Mirrors qcbf.certify.spec._succ_v_lower but bounds W = min(g, V): the
    successor position box gets the exact closed-form g lower bound, the full
    successor box gets the CROWN V lower bound, and we take the min of the two
    (a sound lower bound of min(g, V)).  No domain guard is needed: the world
    disk lies inside the lattice domain, so any successor leaving the domain
    has g < 0 <= m and correctly fails (*).  Heading sub-split c4_psi_subsplit
    and disturbance sub-split c2_d_subsplit only tighten the enclosure.
    """
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
            lbV1, _ = crown_bounds_chunked(
                v_net, b1[:, [0, 2, 4]], b1[:, [1, 3, 5]], True, cert.chunk)
            lbg1, _ = g_bounds_on_box(dyn, b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3])
            val = np.minimum(lbV1[:, 0], lbg1)
            if np.any(m2):
                lbV2, _ = crown_bounds_chunked(
                    v_net, b2[:, [0, 2, 4]], b2[:, [1, 3, 5]], True, cert.chunk)
                lbg2, _ = g_bounds_on_box(dyn, b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3])
                lbW2 = np.minimum(lbV2[:, 0], lbg2)
                val = np.minimum(val, np.where(m2, lbW2, np.inf))
            worst = np.minimum(worst, val)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cells", type=int, default=0,
                    help="Override lattice resolution per axis (verifier lever).")
    ap.add_argument("--m-grid", default="0.0,0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30",
                    help="Comma-separated levels m to sweep.")
    args = ap.parse_args()

    import dataclasses
    cfg = ExperimentConfig.load(args.config)
    if args.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=args.out_dir)
    if args.cells > 0:
        cfg = dataclasses.replace(cfg, cert=dataclasses.replace(
            cfg.cert, n_cells_px=args.cells, n_cells_py=args.cells,
            n_cells_psi=args.cells))
    out = REPO / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def op(name: str):
        return REPO / cfg.out_dir / name

    t_all = time.time()
    dyn, cert = cfg.dynamics, cfg.cert
    m_grid = [float(s) for s in args.m_grid.split(",")]
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    n = lat.n_cells

    v = MLP.load(str(op("v.npz")))
    pi = MLP.load(str(op("pi.npz")))
    v_net = SeqNet.from_mlp(v)
    pol_net = compile_policy(pi, dyn.control_max)

    # ---- W = min(g, V) cell bounds ---------------------------------------- #
    t0 = time.time()
    lbV, ubV = v_cell_bounds(v_net, boxes, np.arange(n), n, cert.chunk)
    gmin, gmax = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    lbW = np.minimum(lbV, gmin)
    ubW = np.minimum(ubV, gmax)
    print(f"[recurrence-cert] cells {n}  (W bounds {time.time()-t0:.1f}s)  "
          f"max m candidate {{ubW>=0}} = {int((ubW >= 0).sum())} cells")

    # ---- successor lb_W on every candidate cell (ub_W >= 0); independent of m
    cand = np.flatnonzero(ubW >= 0.0)
    t0 = time.time()
    u_lo, u_hi = _control_range(pol_net, boxes[cand], dyn, cert)
    succ_lbW = np.full(n, -np.inf)
    succ_lbW[cand] = _succ_W_lower(v_net, boxes[cand], dyn, cert, u_lo, u_hi)
    print(f"[recurrence-cert] witness successor lb_W on {len(cand)} candidate "
          f"cells ({time.time()-t0:.1f}s)")

    # ---- Omega* reference volume ------------------------------------------ #
    oracle = DubinsOracle(dyn, cfg.oracle, gamma=cfg.train.gamma_teach)
    V_star = np.load(op("oracle.npz"))["V"]
    om = omega_star_volume(oracle, V_star, cfg)
    cell_vol = float(lat.cell_volume)

    # ---- sweep the level m ------------------------------------------------- #
    levels = []
    m_star = None
    for m in m_grid:
        active = ubW >= m                       # must all satisfy (*)
        inner = lbW >= m                         # sound interior of S_m
        rec_ok = succ_lbW >= m
        n_active = int(active.sum())
        n_pass = int((active & rec_ok).sum())
        n_fail = n_active - n_pass
        pass_frac = float(n_pass / n_active) if n_active else 0.0
        complete = (n_fail == 0) and n_active > 0
        # certified volume: inner cells, but only meaningful as a certificate
        # when complete.  Report rho_inner regardless, flagged by `complete`.
        vol_inner = int(inner.sum()) * cell_vol
        rho_inner = float(vol_inner / om["volume"]) if om["volume"] > 0 else 0.0
        # worst recurrence margin over the cells we must pass
        marg = (succ_lbW - m)[active]
        rec = {
            "m": m,
            "active_cells": n_active,
            "inner_cells": int(inner.sum()),
            "recurrence_pass": n_pass,
            "recurrence_fail": n_fail,
            "pass_frac": pass_frac,
            "complete_certificate": bool(complete),
            "rho_inner_vs_omega_star": rho_inner,
            "rho_certified": rho_inner if complete else 0.0,
            "min_recurrence_margin": float(marg.min()) if n_active else None,
        }
        levels.append(rec)
        if complete and m_star is None:
            m_star = rec
        flag = "COMPLETE" if complete else f"incomplete ({n_fail} fail)"
        print(f"  m={m:.3f}: active {n_active:6d}  pass {100*pass_frac:5.1f}%  "
              f"inner rho {rho_inner:.4f}  -> {flag}")

    # ---- failure diagnosis at m=0: verifier-slack vs genuine policy leak --- #
    active0 = ubW >= 0.0
    fail0 = active0 & (succ_lbW < 0.0)
    inner0 = lbW >= 0.0                      # truly inside S_0
    fi = fail0 & inner0                       # inner cells that fail (the real problem)
    fb = fail0 & ~inner0                      # straddle/boundary cells that fail
    sl = succ_lbW[fail0]
    margin0_diag = {
        "active_m0": int(active0.sum()),
        "fail_m0": int(fail0.sum()),
        "fail_inner": int(fi.sum()),          # lb_W(cell) >= 0 but successor leaks
        "fail_boundary_straddle": int(fb.sum()),
        "succ_lbW_on_fail_min": float(sl.min()) if sl.size else None,
        "succ_lbW_on_fail_p05": float(np.percentile(sl, 5)) if sl.size else None,
        "succ_lbW_on_fail_p50": float(np.percentile(sl, 50)) if sl.size else None,
        "fail_near_miss_-0.02": int((sl >= -0.02).sum()),   # verifier-slack band
        "fail_near_miss_-0.05": int((sl >= -0.05).sum()),
        "fail_deep_<-0.2": int((sl < -0.2).sum()),          # genuine leak band
    }
    np.savez(op(f"recurrence_cert_arrays{args.tag}.npz"),
             lbW=lbW, ubW=ubW, succ_lbW=succ_lbW)
    print(f"  [diag m=0] fail {int(fail0.sum())} = inner {int(fi.sum())} + "
          f"straddle {int(fb.sum())}; succ_lbW on fail: min {margin0_diag['succ_lbW_on_fail_min']:+.3f} "
          f"p50 {margin0_diag['succ_lbW_on_fail_p50']:+.3f}; near-miss(>=-0.02) "
          f"{margin0_diag['fail_near_miss_-0.02']}, deep(<-0.2) {margin0_diag['fail_deep_<-0.2']}")

    headline = {
        "complete_certificate_exists": m_star is not None,
        "m_star": m_star["m"] if m_star else None,
        "rho_certified": m_star["rho_certified"] if m_star else 0.0,
        "rho_at_m0": levels[0]["rho_inner_vs_omega_star"],
        "pass_frac_at_m0": levels[0]["pass_frac"],
    }
    report = {
        "kind": "recurrence_cbf_superlevel_certificate",
        "config": str(args.config),
        "out_dir": cfg.out_dir,
        "config_hash": cfg.hash(),
        "v_hash": file_hash(op("v.npz")),
        "pi_hash": file_hash(op("pi.npz")),
        "definition": "W = min(g, V_theta); S_m = {W >= m}; witness u = clip(pi_theta). "
                      "Certificate (Theorem A, recurrence form): if lb_W(f(C,[u_lo,u_hi],D)) >= m "
                      "on ALL active cells (ub_W >= m), then S_m is forward-invariant under the "
                      "witness and S_m subset K.  No GFP.  pass_frac is a PASS RATE (certifies "
                      "nothing unless 1.0); rho_inner is the certified set size, valid only where complete.",
        "lattice": {"n_cells": int(n),
                    "shape": [cert.n_cells_px, cert.n_cells_py, cert.n_cells_psi]},
        "omega_star_volume": om["volume"],
        "cell_volume": cell_vol,
        "headline": headline,
        "fail_diagnosis_m0": margin0_diag,
        "levels": levels,
        "wall_s_total": round(time.time() - t_all, 1),
    }
    fname = f"recurrence_cert_report{args.tag}.json"
    save_json(cfg, fname, report)
    print(f"\n[recurrence-cert] headline: "
          f"{'COMPLETE at m*=%.3f, rho=%.4f' % (headline['m_star'], headline['rho_certified']) if m_star else 'NO complete certificate on this grid'};"
          f"  at m=0 pass {100*headline['pass_frac_at_m0']:.1f}%, inner rho {headline['rho_at_m0']:.4f}")
    print(f"  wrote {op(fname)}")


if __name__ == "__main__":
    main()
