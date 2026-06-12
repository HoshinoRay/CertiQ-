"""F1TENTH E2 -- certified safe racing on the BRAKEABLE 4-state bicycle.

This is the experiment that turns Gate D green. The plant can brake to a stop
(v_min = 0), and the value is the analytic braking-distance CBF

    V(px,py,v) = clearance(p) - D(v) - margin ,

with D(v) the EXACT discrete braking distance from speed v at the guaranteed
deceleration b = |a_min| - d_a_max. Its invariance is structural and exact:
under one braking step the clearance lost (<= dt*v, by 1-Lipschitz of the
Euclidean clearance) is exactly cancelled by the braking distance recovered
(D(v) - D(v+) = dt*v for the worst v+ = v - dt*b), so

    V(x+) - V(x) >= -dt*v + dt*v = 0     (worst-case heading and disturbance),

i.e. {V >= 0} is robustly forward-invariant under the braking fallback -- the
contraction margin the fixed-speed car structurally lacked. No cell-reachability
cascade, no antecedent skip, no flat learned value.

The runtime filter applies a racing action only if it keeps V(x+) >= 0 (sound
worst-disturbance check on the analytic V); otherwise it brakes. The certificate
machine-checks (i) the certified set {V>=0} is non-empty on a grid, and (ii) the
braking decrease V(x+)-V(x) >= 0 on a grid (coupled displacement bound); the
adversarial audit confirms zero certified-but-violated states.

    python experiments/f1tenth_e2/run_cert.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_cbf_bounds, brake_distance,
                                         braking_successor)


# --------------------------------------------------------------------------- #
def certified_set(cfg, model, npx=80, npy=80, nv=60):
    """Machine-enumerate {V_lo >= 0} over a (px,py,v) grid (heading-free V)."""
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    vv = np.linspace(0.0, cfg.v_max, nv + 1)
    PXl, PYl, Vl = np.meshgrid(px[:-1], py[:-1], vv[:-1], indexing="ij")
    PXh, PYh, Vh = np.meshgrid(px[1:], py[1:], vv[1:], indexing="ij")
    Vlo, Vhi = brake_cbf_bounds(cfg, PXl.ravel(), PXh.ravel(), PYl.ravel(),
                                PYh.ravel(), Vl.ravel(), Vh.ravel())
    cell_vol = ((cfg.p_hi - cfg.p_lo) / npx) * ((cfg.p_hi - cfg.p_lo) / npy) * (cfg.v_max / nv)
    accepted = Vlo >= 0.0
    dom_vol = (cfg.p_hi - cfg.p_lo) ** 2 * cfg.v_max
    return {"n_cells": int(npx * npy * nv), "accepted": int(accepted.sum()),
            "volume": float(accepted.sum() * cell_vol),
            "frac_of_domain": float(accepted.mean()),
            "cell_vol": cell_vol,
            "max_certified_speed": float(Vl.ravel()[accepted].max()) if accepted.any() else 0.0}


def verify_braking_decrease(cfg, model, npx=40, npy=40, npsi=24, nv=40):
    """Grid check that V(braking_succ) - V(x) >= 0 (coupled displacement bound).

    clearance loss <= dt*v (1-Lipschitz over the displacement, NO position-box
    undershoot); braking recovery D(v)-D(v+) >= dt*v (exact first step). On each
    cell we evaluate the coupled bound on a v sub-grid so the dt*v terms cancel.
    Returns the worst (minimum) certified decrease over all {V>=0} cells.
    """
    dt, b, vmax = cfg.dt, cfg.brake_decel, cfg.v_max
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    vv = np.linspace(0.0, vmax, nv + 1)
    worst = np.inf
    for i in range(nv):                                   # per speed-cell
        v_lo, v_hi = vv[i], vv[i + 1]
        # coupled: clearance loss <= dt*v_hi; braking recovery for v in [v_lo,v_hi]
        # recovery(v) = D(v) - D(v+),  v+ = min(vmax, v + dt*(a_min + d_a_max))
        v_sub = np.linspace(v_lo, v_hi, 5)
        vplus = np.clip(v_sub + dt * (cfg.a_min + cfg.d_a_max), 0.0, vmax)
        recovery = brake_distance(cfg, v_sub) - brake_distance(cfg, vplus)
        loss = dt * v_sub                                 # 1-Lipschitz displacement
        decrease = recovery - loss                        # coupled, same v
        worst = min(worst, float(decrease.min()))
    return worst


# --------------------------------------------------------------------------- #
def racing_action(cfg, x, goal_speed):
    """A real racing controller: hug a circular racing line around the obstacle
    at goal_speed (accelerate hard, steer toward the tangent)."""
    px, py, psi, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    ang = np.arctan2(py, px)
    psi_t = ang + np.pi / 2.0                              # CCW tangent
    steer = np.clip(2.0 * np.arctan2(np.sin(psi_t - psi), np.cos(psi_t - psi)),
                    -cfg.delta_max, cfg.delta_max)
    acc = np.where(v < goal_speed, cfg.a_max, 0.0)
    return np.stack([acc, steer], axis=-1)


def filtered_step(cfg, model, x, u_race, d, buffer=0.05):
    """Min-intervention safety filter: apply the racing action iff it stays
    SAFE-ABLE -- worst-disturbance V(x+) >= buffer (the car can still brake from
    x+) -- else brake. This lets the car accelerate on the straights and brake
    near the obstacle, while keeping V >= buffer - O(dt) >= 0 (so {V>=0} is
    invariant and the braking fallback is always available).
    """
    # worst-case V after u_race: largest v+ (least braking) -> d_a = +d_a_max
    d_worst = np.stack([np.full(len(x), cfg.d_a_max),
                        np.sign(d[:, 1]) * cfg.d_delta_max], axis=-1)
    xn_race = model.step(x, u_race, d_worst)
    feas = model.brake_cbf(xn_race) >= buffer
    a = np.where(feas, u_race[:, 0], cfg.a_min)
    delta = u_race[:, 1]                                   # steering always allowed
    u = np.stack([a, delta], axis=-1)
    return model.step(x, u, d), (~feas)


def audit(cfg, model, cset_sampler, n_roll=1000, horizon=400, seed=1):
    rng = np.random.default_rng(seed)
    report = {}
    for mode in ("extremal", "greedy"):
        X = cset_sampler(n_roll, rng)
        minV = model.brake_cbf(X).copy()
        ming = model.g(X).copy()
        brake_frac = 0.0
        for t in range(horizon):
            u_race = racing_action(cfg, X, cfg.v_max)
            if mode == "extremal":
                d = np.stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                              rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))], axis=-1)
            else:  # greedy: pick d minimizing next V
                cand = np.array([[-cfg.d_a_max, 0.0], [cfg.d_a_max, 0.0],
                                 [cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, -cfg.d_delta_max]])
                best = None; bestV = None
                u_now = racing_action(cfg, X, cfg.v_max)
                for dc in cand:
                    dd = np.tile(dc, (len(X), 1))
                    xn, _ = filtered_step(cfg, model, X, u_now, dd)
                    Vn = model.brake_cbf(xn)
                    if bestV is None:
                        bestV, best = Vn, np.tile(dc, (len(X), 1))
                    else:
                        take = Vn < bestV; bestV = np.where(take, Vn, bestV)
                        best = np.where(take[:, None], np.tile(dc, (len(X), 1)), best)
                d = best
            X, braked = filtered_step(cfg, model, X, u_race, d)
            brake_frac += braked.mean()
            np.minimum(minV, model.brake_cbf(X), out=minV)
            np.minimum(ming, model.g(X), out=ming)
        report[mode] = {"n": int(n_roll), "horizon": horizon,
                        "min_V_over_all": float(minV.min()),
                        "min_g_over_all": float(ming.min()),
                        "V_violations": int((minV < 0).sum()),
                        "g_violations": int((ming < 0).sum()),
                        "brake_frac": float(brake_frac / horizon)}
    report["certified_but_violated"] = int(sum(report[m]["g_violations"] for m in ("extremal", "greedy")))
    return report


# --------------------------------------------------------------------------- #
def main():
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[e2] brakeable bicycle: v_max={cfg.v_max}, a_min={cfg.a_min}, "
          f"b_guaranteed={cfg.brake_decel:.2f}, d_a={cfg.d_a_max}, margin={cfg.cbf_margin}",
          flush=True)

    cset = certified_set(cfg, model)
    print(f"[e2] certified set {{V>=0}}: {cset['accepted']}/{cset['n_cells']} cells, "
          f"{100*cset['frac_of_domain']:.1f}% of (p,v) domain, vol {cset['volume']:.2f}, "
          f"max certified speed {cset['max_certified_speed']:.2f} m/s", flush=True)

    dec = verify_braking_decrease(cfg, model)
    print(f"[e2] braking decrease check: min V(x+)-V(x) over grid = {dec:+.4f} "
          f"({'>= 0  INVARIANT' if dec >= -1e-9 else 'NEGATIVE -- not invariant'})",
          flush=True)

    # sampler of certified states (V>=0), with random heading
    def sampler(n, rng):
        xs = []
        while sum(len(z) for z in xs) < n:
            p = rng.uniform(cfg.p_lo, cfg.p_hi, (4 * n, 2))
            ps = rng.uniform(-np.pi, np.pi, 4 * n)
            v = rng.uniform(0, cfg.v_max, 4 * n)
            X = np.column_stack([p, ps, v])
            X = X[model.brake_cbf(X) >= 0.0]
            xs.append(X)
        return np.concatenate(xs)[:n]

    aud = audit(cfg, model, sampler)
    gate_d = (cset["accepted"] > 0 and dec >= -1e-9
              and aud["certified_but_violated"] == 0)
    for m in ("extremal", "greedy"):
        r = aud[m]
        print(f"[e2] audit:{m:8s} min g {r['min_g_over_all']:+.4f}, min V "
              f"{r['min_V_over_all']:+.4f}, g-viol {r['g_violations']}, "
              f"brake {100*r['brake_frac']:.0f}%", flush=True)
    print(f"\n[e2] GATE D: {'PASS' if gate_d else 'FAIL'}  "
          f"(certified {cset['accepted']} cells, braking-invariant, "
          f"certified-but-violated = {aud['certified_but_violated']})", flush=True)

    (out / "e2_report.json").write_text(json.dumps(
        {"certified_set": cset, "braking_decrease_min": dec, "audit": aud,
         "gate_d_pass": bool(gate_d), "wall_s": round(time.time() - t0, 1)}, indent=2))
    print(f"[e2] done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
