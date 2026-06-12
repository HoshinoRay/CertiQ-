"""F1TENTH E1 -- certified-robust Q-CBF on the kinematic bicycle.

End-to-end demonstration that the *entire* certifier ports to a new plant by
swapping only the dynamics (qcbf/dynamics/bicycle.py): the grid oracle (teacher),
the hand-written MLPs, the CROWN/IBP verifier, the h3 compile, the lattice C2
closure, the runtime filter and the adversarial audit are all reused unchanged.
The only plant-specific injections are ``model=BicycleModel(...)`` into the
oracle and ``successor_boxes_fn=bicycle.successor_boxes`` into the certificate.

The fixed-speed bicycle shares the Dubins position update, so the oracle's
bilinear-xy stencil applies; the heading rate is (v/L)*tan(delta+d). Realistic
F1TENTH steering authority (v/L)*tan(delta_max) is large relative to the
disturbance, so the robust set is *fat* -- and we carry over the Dubins lesson
that the value must be sharpened (TrainConfig.value_sharpen) for the antecedent
skip, hence C2 closure, to be non-vacuous.

    python experiments/f1tenth_e1/run_cert.py --scale smoke   # ~minutes, plumbing
    python experiments/f1tenth_e1/run_cert.py --scale pilot   # Gate-D attempt
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

from qcbf.config import (AuditConfig, CertConfig, ExperimentConfig, NetConfig,
                         OracleConfig, TrainConfig)
from qcbf.dynamics.bicycle import BicycleConfig, BicycleModel, successor_boxes
from qcbf.nets.mlp import (MLP, finetune_witness_margin, policy_forward,
                           train_regression)
from qcbf.oracle.value_iteration import GridOracle
from qcbf.verify.bounds import SeqNet
from qcbf.verify.compiler import compile_h3
from qcbf.certify.refine import precompute_certificate
from qcbf.certify.csearch import omega_star_volume, run_c_sweep
from qcbf.runtime.filter import CertifiedFilter
from qcbf.audit.falsify import run_audit


# --------------------------------------------------------------------------- #
def make_cfg(scale: str) -> ExperimentConfig:
    """Bicycle ExperimentConfig at the requested scale (configs built in code so
    no JSON plant-dispatch is needed; dynamics is a BicycleConfig)."""
    dyn = BicycleConfig()                       # F1TENTH-scale defaults
    nets = NetConfig()
    if scale == "smoke":
        # deliberately tiny: a fast end-to-end plumbing check (Gate D FAIL by
        # design at this resolution / sub-split=1), not a Gate-D demo.
        oc = OracleConfig(n_px=21, n_py=21, n_psi=20, n_u=11, n_d=5,
                          max_iters=120, tol=1e-5, backup="interp", discount=0.92)
        tr = TrainConfig(n_q_samples=60_000, epochs_v=20, epochs_q=15,
                         epochs_pi=20, epochs_margin=15, gamma=0.5,
                         value_sharpen=3.0, margin_target=0.06)
        ce = CertConfig(n_cells_px=20, n_cells_py=20, n_cells_psi=20,
                        n_u_cells=8, ante_d_probes=1,
                        c3_state_subsplit=1, c3_d_subsplit=1,
                        c_sweep=(0.0, 0.05, 0.1, 0.2))
        au = AuditConfig(n_rollouts=150, horizon=120)
    else:  # pilot -- a genuine Gate-D attempt on the bicycle
        oc = OracleConfig(n_px=61, n_py=61, n_psi=60, n_u=21, n_d=7,
                          max_iters=400, tol=1e-6, backup="interp", discount=0.92)
        tr = TrainConfig(n_q_samples=400_000, epochs_v=100, epochs_q=70,
                         epochs_pi=100, epochs_margin=70, gamma=0.7,
                         value_sharpen=3.0, margin_target=0.06)
        ce = CertConfig(n_cells_px=56, n_cells_py=56, n_cells_psi=56,
                        n_u_cells=16, ante_d_probes=3,
                        c_sweep=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3))
        au = AuditConfig(n_rollouts=600, horizon=300)
    return ExperimentConfig(name=f"f1tenth_e1_{scale}",
                            out_dir=f"results/f1tenth_e1_{scale}",
                            dynamics=dyn, oracle=oc, nets=nets, train=tr,
                            cert=ce, audit=au)


# --------------------------------------------------------------------------- #
def fit_networks(cfg, oracle, V_star, rng):
    """V_theta, Q_theta, pi_phi (+ witness-margin fine-tune), bicycle plant.

    Mirrors the Dubins run_train glue but for the injected bicycle oracle;
    value_sharpen is applied to the regression targets (monotone, zero-set
    preserving) so the gate Q>=gamma V is steep enough for the skip test.
    """
    tr, dyn, nets = cfg.train, cfg.dynamics, cfg.nets
    g = oracle.grid
    PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
    X_grid = np.stack([PX, PY, PSI], axis=-1).reshape(-1, 3)

    def sharpen(y):
        return np.tanh(tr.value_sharpen * y) if tr.value_sharpen > 0 else y

    # V regression (grid nodes + random interpolated states)
    n_extra = len(X_grid) // 2
    X_extra = np.column_stack([rng.uniform(dyn.p_lo, dyn.p_hi, n_extra),
                               rng.uniform(dyn.p_lo, dyn.p_hi, n_extra),
                               rng.uniform(-np.pi, np.pi, n_extra)])
    X_v = np.vstack([X_grid, X_extra])
    v = MLP([3, *nets.v_hidden, 1], seed=tr.seed)
    mse_v = train_regression(v, X_v, sharpen(oracle.interp_V(V_star, X_v)),
                             tr.epochs_v, tr.batch, tr.lr, seed=tr.seed, tag="V")

    # Q regression on random (x, delta, d) triples
    n = tr.n_q_samples
    Xq = np.column_stack([rng.uniform(dyn.p_lo, dyn.p_hi, n),
                          rng.uniform(dyn.p_lo, dyn.p_hi, n),
                          rng.uniform(-np.pi, np.pi, n),
                          rng.uniform(-dyn.control_max, dyn.control_max, n),
                          rng.uniform(-dyn.d_max, dyn.d_max, n)])
    Yq = sharpen(oracle.q_star(V_star, Xq[:, :3], Xq[:, 3], Xq[:, 4]))
    q = MLP([5, *nets.q_hidden, 1], seed=tr.seed + 1)
    mse_q = train_regression(q, Xq, Yq, tr.epochs_q, tr.batch, tr.lr,
                             seed=tr.seed + 1, tag="Q")

    # pi regression on robust-greedy fallback labels, then witness-margin tune
    u_flat, _ = oracle.fallback_labels(V_star)
    Y_pi = u_flat.reshape(-1).astype(np.float64)
    pi = MLP([3, *nets.pi_hidden, 1], seed=tr.seed + 2)
    train_regression(pi, X_grid, Y_pi, tr.epochs_pi, tr.batch, tr.lr,
                     seed=tr.seed + 2, tag="pi")
    inside = oracle.interp_V(V_star, X_grid) >= 0.0
    X_m, anchor = X_grid[inside], Y_pi[inside]
    d_grid = np.linspace(-dyn.d_max, dyn.d_max, tr.margin_d_grid)
    finetune_witness_margin(pi, q, v, X_m, d_grid, tr.gamma, dyn.control_max,
                            tr.margin_target, tr.epochs_margin, tr.batch,
                            tr.lr * 0.5, seed=tr.seed + 3, label_anchor=anchor)
    return v, q, pi, {"mse_v": mse_v, "mse_q": mse_q, "n_margin": int(len(X_m))}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="smoke", choices=["smoke", "pilot"])
    args = ap.parse_args()
    cfg = make_cfg(args.scale)
    out = REPO / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dyn = cfg.dynamics
    t_all = time.time()
    print(f"[f1tenth] {cfg.name}: bicycle v={dyn.v} L={dyn.wheelbase} "
          f"delta_max={dyn.delta_max} d_max={dyn.d_max} "
          f"(yaw_rate_max={dyn.yaw_rate_max:.2f} rad/s)", flush=True)

    # --- teacher: bicycle grid oracle (model injected) --------------------- #
    oracle = GridOracle(dyn, cfg.oracle, model=BicycleModel(dyn))
    sol = oracle.solve(verbose=True)
    V_star = sol["V"]
    print(f"[f1tenth] oracle done: Omega* grid frac {np.mean(V_star >= 0):.3f}",
          flush=True)

    # --- train + freeze the artifact --------------------------------------- #
    rng = np.random.default_rng(cfg.train.seed)
    v, q, pi, diag = fit_networks(cfg, oracle, V_star, rng)
    for name, net in (("v", v), ("q", q), ("pi", pi)):
        net.save(str(out / f"{name}.npz"))

    # --- certificate: compile h3, C1/C2/C3, c-sweep, Gate D ---------------- #
    h3 = compile_h3(pi, q, v, cfg.train.gamma, cfg.cert.eps_margin,
                    dyn.control_max)
    v_net, q_net = SeqNet.from_mlp(v), SeqNet.from_mlp(q)
    print(f"[f1tenth] certify lattice {cfg.cert.n_cells_px}^3, "
          f"{cfg.cert.n_u_cells} u-cells, gamma={cfg.train.gamma}, "
          f"sharpen={cfg.train.value_sharpen}", flush=True)
    pre = precompute_certificate(cfg, v_net, q_net, h3, verbose=True,
                                 successor_boxes_fn=successor_boxes)
    om = omega_star_volume(oracle, V_star, cfg)
    sweep, best_mask = run_c_sweep(pre, cfg, om["volume"], verbose=True)
    n_best = int(best_mask.sum())
    rho_best = max(e["rho"] for e in sweep["entries"])
    gate_d = sweep["gate_d_pass"]
    print(f"[f1tenth] GATE D: {'PASS' if gate_d else 'FAIL'}  "
          f"(best certified cells {n_best}, rho={rho_best:.3f})", flush=True)

    # --- adversarial audit (only if something was certified) --------------- #
    audit = {"skipped": True}
    if n_best > 0:
        filt = CertifiedFilter(pre.lattice, best_mask, v, q, pi, q_net,
                               cfg.train.gamma, dyn.control_max, dyn.d_max,
                               cfg.audit.n_u_candidates, cfg.cert.tighten_intermediate)
        audit = run_audit(cfg, BicycleModel(dyn), filt, verbose=True)
        print(f"[f1tenth] AUDIT: certified-but-violated = "
              f"{audit['certified_but_violated']}, invariance viol = "
              f"{audit['invariance_violations']}  -> "
              f"{'PASS' if audit['pass'] else 'FAIL'}", flush=True)

    report = {"config_hash": cfg.hash(), "scale": args.scale,
              "omega_star": om, "funnel": pre.funnel, "train": diag,
              "best_cells": n_best, "best_rho": rho_best,
              "gate_d_pass": gate_d, "sweep": sweep["entries"],
              "audit": audit, "wall_s": round(time.time() - t_all, 1)}
    (out / "f1tenth_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "certificate.npz", accepted=best_mask,
                        cand=pre.cand, lbV=pre.lbV, c3_lb=pre.c3_lb)
    print(f"[f1tenth] done in {report['wall_s']:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
