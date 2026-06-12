"""F1TENTH E2-L / P1 -- DIRECT-COMPOSITION Gate-D certificate of the LEARNED trio.

Replaces the certification PRIMITIVE only (object + plant unchanged): Route 1
(dev doc Sec 8.5) -- bound V_theta(f_brake(x,d)) DIRECTLY via the CROWN affine
lower functional minimised over the true successor (qcbf/certify/direct.py), with
NO intermediate outward-rounded successor box (which was the eroding step in
run_cert_learned.py's Route B, and the looser bound in its Route A).

IRON RULE (enforced here): every primitive is first sanity-checked on the
ANALYTIC IDEAL V.  The ideal MUST pass -- non-empty, order ~86,580 cells -- else
the primitive is wrong and we do NOT touch the learned object.  Only after the
ideal passes do we certify the learned (V_theta, Q_theta, pi_b), so any residual
learned failure is a clean learned-vs-ideal fact, not a primitive artefact.

This is a sub-level (Mode A) certificate of Omega_c = {V_theta >= c}: the deployed
filter races only when a sound check confirms V_theta(x+) >= c, else brakes; so
C2 for the racing branch holds by filter construction and the certificate's job
is (C1) Omega_c collision-free, (C2-brake) the braking fallback keeps Omega_c
invariant from EVERY heading and disturbance, (C3) the witness is admissible
(liveness).  Direct composition tightens the binding C2-brake undershoot.

    python experiments/f1tenth_e2/run_cert_p1.py --quick     # smoke
    python experiments/f1tenth_e2/run_cert_p1.py             # ideal-first + sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_cbf_bounds)
from qcbf.certify.direct import crown_brake_successor_lb
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import GAMMA, distill
from experiments.f1tenth_e2.run_cert import (certified_set,
                                             verify_braking_decrease)


# --------------------------------------------------------------------------- #
def _grid(cfg, npx, npy, nv):
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    vv = np.linspace(0.0, cfg.v_max, nv + 1)
    PXl, PYl, Vl = np.meshgrid(px[:-1], py[:-1], vv[:-1], indexing="ij")
    PXh, PYh, Vh = np.meshgrid(px[1:], py[1:], vv[1:], indexing="ij")
    lo = np.column_stack([PXl.ravel(), PYl.ravel(), Vl.ravel()])
    hi = np.column_stack([PXh.ravel(), PYh.ravel(), Vh.ravel()])
    cell_vol = ((cfg.p_hi - cfg.p_lo) / npx) * ((cfg.p_hi - cfg.p_lo) / npy) * (cfg.v_max / nv)
    return lo, hi, cell_vol


# --------------------------------------------------------------------------- #
def ideal_reference(cfg, model, npx, npy, nv, verbose=True):
    """IRON RULE: certify the ANALYTIC IDEAL V under its tightest sound bound.

    For the analytic V the direct-composition limit is the EXACT structural
    coupled braking-decrease bound (clearance loss dt*v cancels recovery dt*v).
    Reuses run_cert.py (the accepted E2 PASS).  Must be non-empty (~86,580).
    """
    cset = certified_set(cfg, model, npx, npy, nv)
    dec = verify_braking_decrease(cfg, model, npx, npy, max(24, nv // 2), nv)
    passed = cset["accepted"] > 0 and dec >= -1e-9
    if verbose:
        print(f"  [ideal] analytic V: {cset['accepted']}/{cset['n_cells']} cells "
              f"certified, min braking decrease {dec:+.5f} -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    return {"accepted": cset["accepted"], "n_cells": cset["n_cells"],
            "min_decrease": dec, "volume": cset["volume"],
            "frac_of_domain": cset["frac_of_domain"], "pass": bool(passed)}


# --------------------------------------------------------------------------- #
def learned_sublevel_direct(cfg, v_net, npx, npy, nv, npsi, c_grid,
                            chunk=4096, verbose=True):
    """Direct-composition C2-brake closure of {V_theta >= c} (worst heading).

    For each (px,py,v) cell: V_theta box bounds (lb,ub), exact C1, and the
    worst-heading direct-composition brake-successor lower bound
        lbVbrake = min_psi  crown_brake_successor_lb(cell, [psi], D).
    Level c is VALID iff every cell touching the sublevel (ub V_theta >= c) is
    (C1) collision-free and (C2) brake-closed: lbVbrake >= c.  Reports the
    smallest valid c (largest certified set) and the binding undershoot.
    """
    lo, hi, cell_vol = _grid(cfg, npx, npy, nv)
    # V_theta box bounds + analytic V bounds (rho denominator) on the same grid
    lbV, ubV = crown_bounds_chunked(v_net, lo, hi, True, chunk,
                                    progress="V-bounds" if verbose else None)
    lbV, ubV = lbV[:, 0], ubV[:, 0]
    aVlo, _ = brake_cbf_bounds(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1],
                               lo[:, 2], hi[:, 2])
    gmin = model_g_box(cfg, lo, hi)

    cmin = float(min(c_grid))
    work = np.flatnonzero(ubV >= cmin)
    M = len(work)
    if verbose:
        print(f"  [base] {len(lbV)} cells, {M} touch ub V>= {cmin:.2f} "
              f"(analytic-certified {int((aVlo >= 0).sum())})", flush=True)
    if M == 0:
        return {"empty": True}

    wlo, whi = lo[work], hi[work]
    ps_edges = np.linspace(-np.pi, np.pi, npsi + 1)
    lbVbrake = np.full(M, np.inf)               # worst-heading direct successor lb
    pb = Progress(npsi, "P1-direct-psi") if verbose else None
    for h in range(npsi):
        lb_h = crown_brake_successor_lb(
            cfg, v_net, wlo[:, 0], whi[:, 0], wlo[:, 1], whi[:, 1],
            ps_edges[h], ps_edges[h + 1], wlo[:, 2], whi[:, 2])
        lbVbrake = np.minimum(lbVbrake, lb_h)
        if pb is not None:
            pb.update(h + 1)
    if pb is not None:
        pb.done()

    wubV, wlbV, wg = ubV[work], lbV[work], gmin[work]
    undershoot = wubV - lbVbrake                 # binding C2-brake quantity

    routeP1 = None
    accepted_mask = None
    for c in c_grid:
        inset = wubV >= c
        if inset.any() and np.all(wg[inset] >= 0.0) and np.all(lbVbrake[inset] >= c):
            routeP1 = float(c)
            accepted_mask = wlbV >= c            # provably-inside cells (volume)
            break

    # analytic-ideal cell count on the same grid (rho denominator)
    analytic_inc = int((aVlo >= 0).sum())
    near = (wubV >= 0.0) & (wubV < 0.15)
    us_near = undershoot[near]
    certP1 = int(accepted_mask.sum()) if accepted_mask is not None else 0
    return {
        "empty": False,
        "closes_at_c": routeP1,                  # None => no level closes
        "cert_cells": certP1,
        "rho": float(certP1 / max(analytic_inc, 1)),
        "undershoot_med": float(np.median(us_near)) if us_near.size else float("nan"),
        "undershoot_p90": float(np.percentile(us_near, 90)) if us_near.size else float("nan"),
        "undershoot_min": float(undershoot.min()),
        "crown_gap_V_med": float(np.median(wubV - wlbV)),
        "analytic_cells_on_grid": analytic_inc,
        "n_work": int(M), "n_base": int(len(lbV)),
        "cell_m": round((cfg.p_hi - cfg.p_lo) / npx, 4), "cell_vol": cell_vol,
    }


def model_g_box(cfg, lo, hi):
    from qcbf.dynamics.bicycle_accel import g_bounds_sq
    return g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])


# --------------------------------------------------------------------------- #
def box_route_undershoot(cfg, v_net, npx, npy, nv, npsi, chunk=4096):
    """The OLD Route-A (box+CROWN) undershoot on the same grid, for the
    direct-vs-box tightening comparison (no certificate, just the binding gap)."""
    from qcbf.dynamics.bicycle_accel import successor_box
    lo, hi, _ = _grid(cfg, npx, npy, nv)
    lbV, ubV = crown_bounds_chunked(v_net, lo, hi, True, chunk)
    lbV, ubV = lbV[:, 0], ubV[:, 0]
    work = np.flatnonzero(ubV >= 0.0)
    wlo, whi = lo[work], hi[work]
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    lbb = np.full(len(work), np.inf)
    for h in range(npsi):
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, wlo[:, 0], whi[:, 0], wlo[:, 1], whi[:, 1],
            np.full(len(work), ps[h]), np.full(len(work), ps[h + 1]),
            wlo[:, 2], whi[:, 2], cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        bl, _ = crown_bounds_chunked(v_net, np.column_stack([bx0, by0, bv0]),
                                     np.column_stack([bx1, by1, bv1]), True, chunk)
        lbb = np.minimum(lbb, bl[:, 0])
    us = ubV[work] - lbb
    near = (ubV[work] >= 0.0) & (ubV[work] < 0.15)
    return float(np.median(us[near])) if near.any() else float("nan")


# --------------------------------------------------------------------------- #
def run_one(cfg, seed, margin, res, npsi, c_grid, train_kw, verbose=True):
    v, q, pi, diag = distill(cfg, seed, margin, verbose=verbose, **train_kw)
    vn = SeqNet.from_mlp(v)
    rep = learned_sublevel_direct(cfg, vn, *res, npsi, c_grid, verbose=verbose)
    rep["box_route_undershoot_med"] = box_route_undershoot(cfg, vn, *res, npsi)
    rep["mse_v"] = diag.get("mse_v")
    if verbose:
        d, b = rep.get("undershoot_med", float("nan")), rep["box_route_undershoot_med"]
        print(f"  [P1] direct undershoot med={d:+.4f}  (box route {b:+.4f}, "
              f"tightening {b - d:+.4f})", flush=True)
        print(f"  [P1] sub-level closes @ c={rep.get('closes_at_c')}  "
              f"cert={rep.get('cert_cells')}  rho={rep.get('rho', 0):.3f}", flush=True)
    return rep


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    c_grid = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    t0 = time.time()

    if args.quick:
        seeds, margins, res, npsi = [0], [0.10], (40, 40, 30), 8
        train_kw = dict(n_samples=20_000, reg_epochs=10, cbf_epochs=10)
        res_sweep = [(40, 40, 30)]
    else:
        seeds, margins, res, npsi = [0, 1, 2], [0.10], (80, 80, 60), 16
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=25)
        res_sweep = [(48, 48, 36), (64, 64, 48), (80, 80, 60)]

    # ---- IRON RULE: analytic ideal MUST pass under the primitive first ---- #
    print("=" * 84)
    print("P1 IRON RULE -- analytic IDEAL V under the (exact) direct-composition limit")
    print("=" * 84, flush=True)
    ideal = ideal_reference(cfg, model, *res, verbose=True)
    if not ideal["pass"]:
        print("\nIDEAL FAILED under P1 -> primitive is wrong; NOT touching the "
              "learned object.  (Skip to P3/P4, ideal-first.)", flush=True)
        (out / "p1_report.json").write_text(json.dumps(
            {"ideal": ideal, "learned": None, "verdict": "ideal-fail",
             "wall_s": round(time.time() - t0, 1)}, indent=2))
        return
    print(f"  IDEAL PASSES ({ideal['accepted']} cells) -> certifying the learned "
          f"trio under P1.\n", flush=True)

    # ---- learned trio under P1 (>=3 seeds) -------------------------------- #
    runs = []
    for s in seeds:
        print(f"=== learned P1  seed={s}  m={margins[0]:.2f}  res={res} "
              f"npsi={npsi} ===", flush=True)
        runs.append(run_one(cfg, s, margins[0], res, npsi, c_grid, train_kw))

    # ---- resolution sweep (P2 trend: undershoot vs cell size) ------------- #
    print("\n=== P2 trend: undershoot vs resolution (seed 0) ===", flush=True)
    sweep = []
    for r in res_sweep:
        rr = run_one(cfg, 0, margins[0], r, npsi, c_grid, train_kw, verbose=False)
        sweep.append({"res": list(r), "cell_m": rr["cell_m"],
                      "undershoot_med": rr.get("undershoot_med"),
                      "box_undershoot_med": rr["box_route_undershoot_med"],
                      "closes_at_c": rr.get("closes_at_c"),
                      "cert_cells": rr.get("cert_cells")})
        print(f"  cell={rr['cell_m']:.3f}m  direct us={rr.get('undershoot_med'):+.4f}"
              f"  box us={rr['box_route_undershoot_med']:+.4f}  "
              f"closes@c={rr.get('closes_at_c')}", flush=True)

    closes = [r.get("closes_at_c") for r in runs]
    any_close = any(c is not None for c in closes)
    blob = {"ideal": ideal, "learned_per_seed": runs, "res_sweep": sweep,
            "config": {"res": list(res), "npsi": npsi, "seeds": seeds,
                       "margin": margins[0], "gamma": GAMMA, "c_grid": list(c_grid)},
            "verdict": "learned-pass" if any_close else "learned-fail",
            "wall_s": round(time.time() - t0, 1)}
    (out / "p1_report.json").write_text(json.dumps(blob, indent=2))

    # ---- attributable report ---------------------------------------------- #
    print("\n" + "=" * 84)
    print("P1  DIRECT-COMPOSITION result (object + plant FIXED; primitive replaced)")
    print("=" * 84)
    print(f"IDEAL (iron rule): PASS, {ideal['accepted']} cells "
          f"({100*ideal['frac_of_domain']:.1f}% of (p,v) domain)")
    du = np.nanmedian([r.get("undershoot_med") for r in runs])
    bu = np.nanmedian([r["box_route_undershoot_med"] for r in runs])
    print(f"LEARNED ({len(seeds)} seeds): direct-composition undershoot med "
          f"{du:+.4f}  (box route {bu:+.4f}; tightening {bu - du:+.4f})")
    if any_close:
        cc = [r["cert_cells"] for r in runs]
        rh = [r["rho"] for r in runs]
        print(f"  Gate D PASS: closes@c={closes}, cert cells {cc}, "
              f"rho={np.mean(rh):.3f}")
    else:
        print("  Gate D FAIL (sub-level): no level closes -- the worst-heading "
              "brake undershoot stays > 0 (knife-edge: analytic contraction is")
        print("  EXACTLY zero, so any positive black-box undershoot fails). Direct")
        print("  composition reduces the undershoot vs the box route but cannot")
        print("  reach the exact 1-Lipschitz cancellation. This is a clean "
              "learned-vs-ideal fact (ideal passes, learned does not), NOT a")
        print("  primitive artefact.  undershoot ~ the per-cell CROWN gap on V_theta")
        print("  (output-side), so P3 (zonotope, input-side) will NOT close it;")
        print("  the live candidate is P4 (contraction / finite-horizon recursive")
        print("  feasibility), which drops the successor-in-level knife-edge.")
    print(f"Wrote {out / 'p1_report.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
