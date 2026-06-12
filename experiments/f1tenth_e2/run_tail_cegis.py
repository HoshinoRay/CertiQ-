"""Path 2 -- STABILIZED TAIL-CEGIS (single seed, with stop-loss).

Step-0/1/2 localised the failure: ~99% of the {V_theta~0} boundary shell is
already nearly clean (p01(h) ~ -0.022) and only the worst 1-5% TAIL fails
(min h ~ -0.075).  Naive CEGIS oscillated because it chased a MOVING boundary.
This driver adds the three Path-2 stabilisers and runs single-seed with an
explicit stop-loss so we don't burn compute on a non-converging loop:

  1. FIXED level c (here c=0): always probe/repair the same shell {V_theta in [0,d]}.
  2. TRUST REGION: each round anchors V to the PREVIOUS iterate V_{k-1}(X)
     (distill(..., v_ref=V_{k-1}), anchor_w bumped) so V cannot run away.
  3. PERSISTENT replay: counterexamples ACCUMULATE across rounds (capped), never
     replace-only, so old tail pockets stay covered while new ones are added.

The repaired quantity is the EXACT (no-bounds) deployed decrease on the shell:
    h(x,d) = V_theta(f(x,pi_b,d)) - gamma V_theta(x)            (c = 0).

Stop-loss (per the plan):
  round 1 hope:  p01(h) > -0.005  AND  Pr[h<0] clearly down (5% -> <1-3%).
  final goal:    min h > 0.
  abort -> structured V_theta if after K rounds  min h ~ -0.07 and Pr[h<0] ~ 5%
           (then the tail is NOT replay-fixable; see docs note Path A/B).

    python experiments/f1tenth_e2/run_tail_cegis.py            # 4 rounds, seed 0
    python experiments/f1tenth_e2/run_tail_cegis.py --rounds 3 --seed 1
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

from qcbf.dynamics.bicycle_accel import BicycleAccelConfig, BicycleAccelModel
from experiments.f1tenth_e2.distill import GAMMA, distill
from experiments.f1tenth_e2.run_probe_ceiling import sample_band, ceiling_margins

DELTA = 0.10                 # boundary shell thickness {V in [0, DELTA]}
CEX_CAP = 60_000             # persistent replay buffer cap


def probe_shell(cfg, model, v, q, pi, n_band, rng):
    """Exact deployed decrease h = V(f)-gamma V on the in-set shell {V in [0,d]}.
    Returns (stats, counterexample states with h<0)."""
    X = sample_band(cfg, v, n_band, DELTA + 0.02, rng)
    m = ceiling_margins(cfg, model, v, q, pi, X)
    shell = (m["Vx"] >= 0.0) & (m["Vx"] <= DELTA)
    h = m["m2_gamma"][shell]
    Xs = X[shell]
    stats = {"min": float(h.min()), "p01": float(np.percentile(h, 1)),
             "p05": float(np.percentile(h, 5)), "p50": float(np.percentile(h, 50)),
             "frac_neg": float(np.mean(h < 0.0)), "n_shell": int(shell.sum())}
    return stats, Xs[h < 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    t0 = time.time()
    train_kw = (dict(n_samples=20_000, reg_epochs=10, cbf_epochs=12) if args.quick
                else dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30))
    n_band = 120_000 if args.quick else 350_000
    rng = np.random.default_rng(9000 + args.seed)

    print("=" * 92)
    print(f"PATH 2 -- STABILIZED TAIL-CEGIS (seed {args.seed}, fixed level c=0, "
          f"trust region + persistent replay)")
    print("=" * 92, flush=True)

    cex_buf = np.zeros((0, 4))
    v_prev = None
    traj = []
    for rd in range(args.rounds):
        tr = time.time()
        v, q, pi, diag = distill(cfg, args.seed, 0.10, c2_fix=True,
                                 cex=(cex_buf if len(cex_buf) else None),
                                 v_ref=v_prev, verbose=False, **train_kw)
        st, cex_new = probe_shell(cfg, model, v, q, pi, n_band, rng)
        # PERSISTENT replay: accumulate, then cap by random subsample
        cex_buf = np.concatenate([cex_buf, cex_new])
        if len(cex_buf) > CEX_CAP:
            cex_buf = cex_buf[rng.permutation(len(cex_buf))[:CEX_CAP]]
        v_prev = v
        st.update(round=rd, mse_v=diag["mse_V_target"], n_cex_new=int(len(cex_new)),
                  n_cex_buf=int(len(cex_buf)), wall_s=round(time.time() - tr, 1))
        traj.append(st)
        print(f"round {rd}: min {st['min']:+.4f}  p01 {st['p01']:+.4f}  "
              f"p05 {st['p05']:+.4f}  frac<0 {st['frac_neg']:.3f}  "
              f"| MSE(V) {diag['mse_V_target']:.3f}  cex+{len(cex_new)} "
              f"buf={len(cex_buf)} ({time.time()-tr:.0f}s)", flush=True)

    # ---- verdict + stop-loss --------------------------------------------- #
    r0, rl = traj[0], traj[-1]
    improved = (rl["p01"] > r0["p01"] + 1e-3) and (rl["frac_neg"] < r0["frac_neg"] - 5e-3)
    success = rl["min"] > 0.0
    near = (rl["p01"] > -0.005) and (rl["frac_neg"] < 0.02)
    stalled = (rl["min"] < -0.05) and (rl["frac_neg"] > 0.04)
    if success:
        verdict = "SUCCESS -- min h > 0; the object is pointwise C2-clean on the shell"
    elif near:
        verdict = "PROMISING -- p01>~0 and frac<0 collapsed; one or two more rounds may close min"
    elif stalled:
        verdict = ("STALLED -- min ~ -0.07 and frac<0 ~ 5% persist; the tail is NOT "
                   "replay-fixable -> switch to structured V_theta (pairwise temporal "
                   "regression / V=C(p)-D(v)).")
    else:
        verdict = "PARTIAL -- some movement but not closed; inspect trajectory"

    print("\n" + "=" * 92)
    print("TAIL-CEGIS trajectory (exact deployed decrease on {V in [0,0.1]} shell):")
    print(f"  {'round':>5} | {'min h':>8} | {'p01':>8} | {'p05':>8} | {'frac<0':>7} | {'cex buf':>8}")
    for s in traj:
        print(f"  {s['round']:5d} | {s['min']:+8.4f} | {s['p01']:+8.4f} | "
              f"{s['p05']:+8.4f} | {s['frac_neg']:7.3f} | {s['n_cex_buf']:8d}")
    print("-" * 92)
    print(f"baseline (no C2-fix) was: min -0.115, frac<0 5.9%, p01 -0.033")
    print(f"VERDICT: {verdict}")
    print(f"  round-1 stop-loss check: p01>-0.005? {traj[0]['p01']>-0.005}; "
          f"frac<0 dropped vs base? {traj[0]['frac_neg']<0.04}")

    blob = {"seed": args.seed, "rounds": args.rounds, "delta": DELTA,
            "gamma": GAMMA, "trajectory": traj, "verdict": verdict,
            "improved": bool(improved), "success": bool(success),
            "wall_s": round(time.time() - t0, 1)}
    (out / f"tail_cegis_s{args.seed}.json").write_text(json.dumps(blob, indent=2))
    print(f"Wrote {out / f'tail_cegis_s{args.seed}.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
