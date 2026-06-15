"""Realized collision rate of the PERFECT (HJ-optimal) filter under disturbance.

Motivation
----------
The cell-worst certificate does NOT pass at 100% even on the *ground-truth* HJ
value/Q (cell pass ~0.7 for V, ~0.9 for Q).  NOTE: the GT V_HJ/Q_HJ are GRID
tables from value iteration, NOT neural networks -- there is no CROWN here.
The GT's cell-worst bound is the sound interval enclosure of the (piecewise
multilinear) grid value over each cell, plus the dynamics interval
over-approximation; the residual <100% is pure verifier conservativeness +
value-grid discretization, with ZERO fitting error and ZERO CROWN slack.

But the HJ-optimal controller is, in continuous time/space, a robust
controlled-invariant filter on Omega* = {V_HJ >= 0}: started inside Omega* it
should keep g >= 0 forever against EVERY admissible disturbance.  So the gap
between "cert pass < 100%" and "actually safe" is exactly the **soundness slack**
of the cell-worst verifier (interval over-approximation + grid), NOT real
control failure.  This experiment measures the realized failure directly.

This experiment measures the *realized* failure: deploy the best HJ-greedy filter
we can extract from the ground-truth value, run it for a long horizon from every
Omega* cell under several disturbance signals (including the matched value-greedy
worst-case adversary), and report at what LEVEL and PROBABILITY collisions occur.

The deployed controller is re-evaluated at the continuous state each step:

    u*(x) = argmax_{u in Ugrid} min_{d in Dgrid} V_HJ(f(x, u, d))         (control)
    d*(x) = argmin_{d in Dgrid} V_HJ(f(x, u*, d))      (matched worst-case adv)

with V_HJ the (frozen) oracle value, trilinearly interpolated.  Using grids at
least as fine as the solve grids makes this the best-case ("perfect") filter, so
any collision is an upper bound on the discretization leak.

Outputs: collision-fraction trend vs horizon per regime; converged probability;
stratification by the starting boundary-shell value V_HJ(x0); violation depth
(min g reached); and the time-to-first-collision profile for the adversary.
"""
from __future__ import annotations

import argparse
import json
import time

from common import REPO

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.volume import omega_star_volume
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import DubinsModel
from qcbf.oracle.value_iteration import DubinsOracle


def _fmt_eta(done, total, t0):
    el = time.time() - t0
    rate = el / max(done, 1)
    rem = rate * (total - done)
    return f"{el:5.1f}s elapsed, ETA {rem:5.1f}s ({100*done/total:4.1f}%)"


def _control_and_adv(oracle, V, model, X, Ugrid, Dgrid, want_adv):
    """Vectorized HJ-greedy control (and matched worst-case adv) at states X.

    Successor position is independent of (u,d); only heading changes, so each
    (u,d) is one trilinear value lookup.  Returns (u_star, dadv) where dadv is
    None unless want_adv.
    """
    M = len(X)
    # ---- control: u* = argmax_u min_d V(f(x,u,d)) -------------------------
    best_val = np.full(M, -np.inf)
    u_star = np.zeros(M)
    for u in Ugrid:
        succ = model.step(X[None, :, :], np.full((len(Dgrid), M), u),
                          Dgrid[:, None])           # (n_d, M, 3)
        vals = oracle.interp_V(V, succ)             # (n_d, M)
        worst = vals.min(axis=0)                     # min_d
        sel = worst > best_val
        best_val = np.where(sel, worst, best_val)
        u_star = np.where(sel, u, u_star)
    if not want_adv:
        return u_star, None
    # ---- matched worst-case adversary at u*: argmin_d V(f(x,u*,d)) --------
    succ = model.step(X[None, :, :], u_star[None, :] + np.zeros((len(Dgrid), 1)),
                      Dgrid[:, None])                # (n_d, M, 3)
    vals = oracle.interp_V(V, succ)                  # (n_d, M)
    dadv = Dgrid[vals.argmin(axis=0)]
    return u_star, dadv


def _simulate(oracle, V, model, X0, Ugrid, Dgrid, Hcheck, regime, rng, dmax,
              prog_label=""):
    """Run the perfect filter from X0 for max(Hcheck) steps under `regime`.

    regime in {'d0','cpos','cneg','rand','adv'}.  Returns dict with:
      collided_at[H]  : bool mask, collided by step H
      tcoll           : first-collision step (or -1)
      ming            : min g ever reached
    """
    Hcheck = sorted(Hcheck)
    Hmax = Hcheck[-1]
    c = model.cfg
    ox, oy = c.obs_center
    M = len(X0)
    X = X0.copy()
    collided = np.zeros(M, bool)
    hit_obs = np.zeros(M, bool)   # first violation was the obstacle
    hit_wld = np.zeros(M, bool)   # first violation was the world boundary
    tcoll = np.full(M, -1, dtype=np.int64)
    ming = model.g(X).astype(float)
    want_adv = (regime == "adv")
    snap = {}
    ci = 0
    t0 = time.time()
    for h in range(1, Hmax + 1):
        u_star, dadv = _control_and_adv(oracle, V, model, X, Ugrid, Dgrid, want_adv)
        if regime == "d0":
            d = np.zeros(M)
        elif regime == "cpos":
            d = np.full(M, dmax)
        elif regime == "cneg":
            d = np.full(M, -dmax)
        elif regime == "rand":
            d = rng.uniform(-dmax, dmax, M)
        else:  # adv
            d = dadv
        X = model.step(X, u_star, d)
        g_obs = (X[:, 0] - ox) ** 2 + (X[:, 1] - oy) ** 2 - c.obs_radius ** 2
        g_wld = c.world_radius ** 2 - (X[:, 0] ** 2 + X[:, 1] ** 2)
        gnow = np.minimum(g_obs, g_wld)
        ming = np.minimum(ming, gnow)
        newly = (gnow < 0.0) & ~collided
        tcoll = np.where(newly, h, tcoll)
        hit_obs |= newly & (g_obs < 0.0)
        hit_wld |= newly & (g_obs >= 0.0) & (g_wld < 0.0)
        collided |= (gnow < 0.0)
        while ci < len(Hcheck) and Hcheck[ci] == h:
            snap[Hcheck[ci]] = collided.copy()
            ci += 1
        if h % 25 == 0 or h == Hmax:
            print(f"    [{prog_label}/{regime}] step {h:4d}/{Hmax}  "
                  f"collided {int(collided.sum()):5d}/{M}  {_fmt_eta(h, Hmax, t0)}",
                  flush=True)
    return {"snap": snap, "tcoll": tcoll, "ming": ming, "collided": collided,
            "hit_obs": hit_obs, "hit_wld": hit_wld}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--art-dir", default="results/dubins_e0_pilot_2sided_v012",
                    help="Dir with oracle.npz (key 'V' = ground-truth HJ value).")
    ap.add_argument("--n-u", type=int, default=21, help="Control grid (>= solve n_u for 'perfect').")
    ap.add_argument("--n-d", type=int, default=21, help="Disturbance grid (control inner-min AND adversary).")
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--checkpoints", default="1,2,5,10,20,50,100,200,500,1000")
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--max-starts", type=int, default=0, help="0 = all Omega* cells.")
    ap.add_argument("--out", default="results/dubins_e0_pilot_2sided_v012/perfect_filter_collision.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    dyn, cert = cfg.dynamics, cfg.cert
    art = REPO / args.art_dir
    model = DubinsModel(dyn)
    oracle = DubinsOracle(dyn, cfg.oracle, gamma=cfg.train.gamma_teach)
    V = np.load(art / "oracle.npz")["V"]
    dmax = dyn.d_max
    Ugrid = np.linspace(-dyn.control_max, dyn.control_max, args.n_u)
    Dgrid = np.linspace(-dmax, dmax, args.n_d)
    Hcheck = [int(s) for s in args.checkpoints.split(",") if int(s) <= args.horizon]
    if args.horizon not in Hcheck:
        Hcheck.append(args.horizon)

    # starts = 40^3 cell centres in Omega* = {V_HJ >= 0} and in K = {g >= 0}
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    centres = np.column_stack([0.5 * (boxes[:, 0] + boxes[:, 1]),
                               0.5 * (boxes[:, 2] + boxes[:, 3]),
                               0.5 * (boxes[:, 4] + boxes[:, 5])])
    g0_all = model.g(centres)
    V0_all = oracle.interp_V(V, centres)
    in_om = (V0_all >= 0.0) & (g0_all >= 0.0)
    idx = np.flatnonzero(in_om)
    if args.max_starts and len(idx) > args.max_starts:
        idx = np.random.default_rng(0).choice(idx, args.max_starts, replace=False)
        idx.sort()
    X0 = centres[idx]
    V0 = V0_all[idx]
    M = len(X0)

    om = omega_star_volume(oracle, V, cfg)
    print(f"[perfect-filter] V_HJ from {args.art_dir}  Omega* frac {om['frac_of_domain']:.3f}")
    print(f"[perfect-filter] starts in Omega*: {M} cells  "
          f"(of {lat.n_cells})  Ugrid {args.n_u}  Dgrid {args.n_d}  H={args.horizon}")
    print(f"[perfect-filter] regimes: d0, cpos(+{dmax}), cneg(-{dmax}), "
          f"rand x{args.rand_seeds}, adv(value-greedy worst-case)")

    regimes = ["d0", "cpos", "cneg", "adv"] + [f"rand{i}" for i in range(args.rand_seeds)]
    runs = {}
    t_all = time.time()
    for r in regimes:
        base = "rand" if r.startswith("rand") else r
        seed = 100 + (int(r[-1]) if r.startswith("rand") else 0)
        rng = np.random.default_rng(seed)
        print(f"  -- regime {r} --", flush=True)
        runs[r] = _simulate(oracle, V, model, X0, Ugrid, Dgrid, Hcheck,
                            base, rng, dmax, prog_label=r)
    wall = time.time() - t_all

    # ---- collision trend (fraction of Omega* starts) vs horizon ----------
    def frac(mask):
        return round(100.0 * int(mask.sum()) / M, 3)

    trend = {r: {H: frac(runs[r]["snap"][H]) for H in Hcheck} for r in regimes}
    # aggregate the rand seeds (mean +/- spread of converged prob)
    rand_final = [100.0 * int(runs[f"rand{i}"]["collided"].sum()) / M
                  for i in range(args.rand_seeds)]

    # ---- boundary-shell stratification (collision frac by V0 bin) --------
    v_edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, np.inf]
    shell = {}
    for r in ["adv", "cpos", "cneg", "rand0", "d0"]:
        coll = runs[r]["collided"]
        rows = []
        for lo, hi in zip(v_edges[:-1], v_edges[1:]):
            m = (V0 >= lo) & (V0 < hi)
            n = int(m.sum())
            c = int((coll & m).sum())
            rows.append({"V0_lo": lo, "V0_hi": (None if np.isinf(hi) else hi),
                         "n": n, "collided": c,
                         "frac_pct": (round(100.0 * c / n, 3) if n else None)})
        shell[r] = rows

    # ---- V-threshold sweep: collision prob of {V>=tau} starts ------------
    # {V_HJ>=0} is NOT invariant (the discount lets V decay by gamma/step);
    # raise the level tau to recover a genuine ~invariant sub-level set.
    # rho_tau = Vol({V>=tau})/Vol(Omega*) ~ n(V0>=tau)/M (equal cell volumes).
    taus = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    tau_sweep = {}
    for r in ["d0", "adv", "rand0"]:
        coll = runs[r]["collided"]
        rows = []
        for tau in taus:
            m = V0 >= tau
            n = int(m.sum())
            c = int((coll & m).sum())
            rows.append({"tau": tau, "n_starts": n,
                         "rho_vs_omega_star": round(n / M, 4),
                         "collision_pct": (round(100.0 * c / n, 3) if n else None)})
        tau_sweep[r] = rows

    # ---- obstacle-first vs world-first split (per regime) ----------------
    coll_type = {r: {"obstacle_first": int(runs[r]["hit_obs"].sum()),
                     "world_first": int(runs[r]["hit_wld"].sum())}
                 for r in regimes}

    # ---- violation depth + time-to-collision for the adversary -----------
    adv = runs["adv"]
    coll = adv["collided"]
    depth = adv["ming"][coll]
    tcoll = adv["tcoll"][coll]
    depth_stats = {
        "n_collided": int(coll.sum()),
        "min_g_p50": (round(float(np.median(depth)), 4) if depth.size else None),
        "min_g_p05": (round(float(np.quantile(depth, 0.05)), 4) if depth.size else None),
        "min_g_worst": (round(float(depth.min()), 4) if depth.size else None),
        "frac_grazing_gt_-0.02": (round(float((depth > -0.02).mean()), 3) if depth.size else None),
    }
    tcoll_stats = {
        "tcoll_p50": (int(np.median(tcoll)) if tcoll.size else None),
        "tcoll_p90": (int(np.quantile(tcoll, 0.9)) if tcoll.size else None),
        "collided_by_H1": int((adv["tcoll"] == 1).sum()),
        "collided_after_H50": int(((tcoll > 50)).sum()) if tcoll.size else 0,
    }

    report = {
        "kind": "perfect_filter_realized_collision",
        "art_dir": args.art_dir,
        "value_object": "ground-truth HJ value V_HJ (oracle.npz['V'])",
        "filter": "u* = argmax_u min_d V_HJ(f(x,u,d)), re-evaluated at continuous state",
        "grids": {"n_u": args.n_u, "n_d": args.n_d,
                  "solve_n_u": cfg.oracle.n_u, "solve_n_d": cfg.oracle.n_d},
        "omega_star": {"frac_domain": om["frac_of_domain"]},
        "starts_in_omega_star": M, "lattice_cells": lat.n_cells,
        "horizon": args.horizon, "checkpoints": Hcheck,
        "collision_frac_pct_vs_H": {r: {str(h): trend[r][h] for h in Hcheck} for r in regimes},
        "converged_collision_prob_pct": {
            "d0": trend["d0"][args.horizon],
            "cpos": trend["cpos"][args.horizon],
            "cneg": trend["cneg"][args.horizon],
            "adv": trend["adv"][args.horizon],
            "rand_mean": round(float(np.mean(rand_final)), 3),
            "rand_seeds": [round(x, 3) for x in rand_final],
        },
        "boundary_shell_collision_frac_pct": shell,
        "v_threshold_sweep": tau_sweep,
        "collision_type_first": coll_type,
        "adv_violation_depth": depth_stats,
        "adv_time_to_collision": tcoll_stats,
        "wall_s": round(wall, 1),
    }
    if args.out:
        (REPO / args.out).write_text(json.dumps(report, indent=2))

    # ---------------------------- console ---------------------------------
    print(f"\n[perfect-filter] DONE  wall {wall:.1f}s   starts {M} (Omega*)")
    print(f"\n  collision fraction (% of Omega* starts) vs horizon H:")
    hdr = "    regime      " + "".join(f"{H:>8d}" for H in Hcheck)
    print(hdr)
    for r in regimes:
        row = "".join(f"{trend[r][H]:>8.3f}" for H in Hcheck)
        print(f"    {r:<11s} {row}")
    cv = report["converged_collision_prob_pct"]
    print(f"\n  CONVERGED collision probability (H={args.horizon}):")
    print(f"    d=0 nominal        : {cv['d0']:.3f}%")
    print(f"    d=+{dmax} constant   : {cv['cpos']:.3f}%")
    print(f"    d=-{dmax} constant   : {cv['cneg']:.3f}%")
    print(f"    random U[-d,d]     : {cv['rand_mean']:.3f}%  (seeds {cv['rand_seeds']})")
    print(f"    worst-case adv     : {cv['adv']:.3f}%   <-- matched Isaacs adversary")
    print(f"\n  boundary-shell stratification (adversary), collision % by V_HJ(x0) bin:")
    print(f"    {'V0 bin':<16s}{'n':>8s}{'coll':>8s}{'frac%':>9s}")
    for row in shell["adv"]:
        hi = "inf" if row["V0_hi"] is None else f"{row['V0_hi']:.2f}"
        lab = f"[{row['V0_lo']:.2f},{hi})"
        fr = "-" if row["frac_pct"] is None else f"{row['frac_pct']:.3f}"
        print(f"    {lab:<16s}{row['n']:>8d}{row['collided']:>8d}{fr:>9s}")
    print(f"\n  V-threshold sweep: collision% of {{V_HJ>=tau}} starts "
          f"(recover an invariant sub-level set):")
    print(f"    {'tau':>6s}{'rho':>9s}{'d0 coll%':>11s}{'adv coll%':>11s}{'rand coll%':>12s}")
    for i, tau in enumerate(taus):
        d0r = tau_sweep["d0"][i]; ar = tau_sweep["adv"][i]; rr = tau_sweep["rand0"][i]
        f0 = "-" if d0r["collision_pct"] is None else f"{d0r['collision_pct']:.3f}"
        fa = "-" if ar["collision_pct"] is None else f"{ar['collision_pct']:.3f}"
        fr = "-" if rr["collision_pct"] is None else f"{rr['collision_pct']:.3f}"
        print(f"    {tau:>6.2f}{d0r['rho_vs_omega_star']:>9.4f}{f0:>11s}{fa:>11s}{fr:>12s}")
    print(f"\n  collision type (first violation, obstacle vs world boundary):")
    for r in regimes:
        ct = coll_type[r]
        print(f"    {r:<11s} obstacle-first {ct['obstacle_first']:5d}   "
              f"world-first {ct['world_first']:5d}")
    print(f"\n  adversary violation depth (min g among collided): "
          f"median {depth_stats['min_g_p50']}, worst {depth_stats['min_g_worst']}, "
          f"grazing(>-0.02) {depth_stats['frac_grazing_gt_-0.02']}")
    print(f"  adversary time-to-collision: median {tcoll_stats['tcoll_p50']}, "
          f"by H=1 {tcoll_stats['collided_by_H1']}, after H=50 {tcoll_stats['collided_after_H50']}")
    if args.out:
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
