"""Numerical C1/C3/C4 checks on the oracle ground-truth object.

This stage mirrors the learned strict-spec structure, but replaces

    (V_theta, Q_theta, pi_theta)

with the oracle objects

    V_HJ,
    Q_HJ(x,u,d) = V_HJ(f(x,u,d)),
    pi_HJ(x) = argmax_u min_d Q_HJ(x,u,d).

It intentionally does NOT check an old C2 condition.  The checks are numerical
grid / Monte-Carlo diagnostics on the oracle interpolation, not a CROWN
continuous-cell proof.
"""
from __future__ import annotations

import argparse
import time

from common import REPO, out_path, save_json, update_manifest

import numpy as np

from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import DubinsModel
from qcbf.oracle.value_iteration import DubinsOracle


def _menu(dyn, cert) -> np.ndarray:
    if cert.n_u_cells <= 1:
        return np.array([0.0])
    return np.linspace(-dyn.control_max, dyn.control_max, cert.n_u_cells)


def _margin_summary(margin: np.ndarray, active: np.ndarray) -> dict:
    vals = margin[active]
    if len(vals) == 0:
        return {
            "active": 0,
            "pass": 0,
            "fail": 0,
            "pass_frac": None,
            "margin_min": None,
            "margin_p01": None,
            "margin_p05": None,
            "margin_mean": None,
        }
    ok = vals >= 0.0
    return {
        "active": int(active.sum()),
        "pass": int(ok.sum()),
        "fail": int((~ok).sum()),
        "pass_frac": float(ok.mean()),
        "margin_min": float(np.min(vals)),
        "margin_p01": float(np.percentile(vals, 1)),
        "margin_p05": float(np.percentile(vals, 5)),
        "margin_mean": float(np.mean(vals)),
    }


def _oracle_witness(oracle: DubinsOracle, V: np.ndarray, X: np.ndarray,
                    u_grid: np.ndarray, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return min_d Q_HJ(x, pi_HJ(x), d) and pi_HJ on a finite oracle grid."""
    best = np.full(len(X), -np.inf)
    best_u = np.zeros(len(X))
    for u in u_grid:
        worst = np.full(len(X), np.inf)
        for d in d_grid:
            q = oracle.q_star(V, X, np.full(len(X), u), np.full(len(X), d))
            np.minimum(worst, q, out=worst)
        take = worst > best
        best = np.where(take, worst, best)
        best_u = np.where(take, u, best_u)
    return best, best_u


def _c4_menu_report(oracle: DubinsOracle, model: DubinsModel, V: np.ndarray,
                    X: np.ndarray, active: np.ndarray,
                    menu: np.ndarray, d_grid: np.ndarray,
                    tol: float = 1e-12) -> dict:
    """Check C4 on all runtime menu actions:

        min_d Q_HJ(x,u,d) <= min_d V_HJ(f(x,u,d)).

    Since Q_HJ is defined as V_HJ o f, this should hold as equality up to
    floating-point roundoff.
    """
    Xa = X[active]
    if len(Xa) == 0:
        return {"active": 0, "actions": int(len(menu)), "pass": True,
                "min_margin": None, "max_abs_identity_error": None}
    worst_margin = np.full(len(Xa), np.inf)
    max_abs = 0.0
    for u in menu:
        qmin = np.full(len(Xa), np.inf)
        vmin = np.full(len(Xa), np.inf)
        for d in d_grid:
            q = oracle.q_star(V, Xa, np.full(len(Xa), u), np.full(len(Xa), d))
            vf = oracle.interp_V(V, model.step(Xa, np.full(len(Xa), u), np.full(len(Xa), d)))
            max_abs = max(max_abs, float(np.max(np.abs(q - vf))))
            np.minimum(qmin, q, out=qmin)
            np.minimum(vmin, vf, out=vmin)
        np.minimum(worst_margin, vmin - qmin, out=worst_margin)
    return {
        "active": int(len(Xa)),
        "actions": int(len(menu)),
        "pass": bool(np.all(worst_margin >= -tol)),
        "min_margin": float(np.min(worst_margin)),
        "max_abs_identity_error": max_abs,
    }


def _c4_witness_report(oracle: DubinsOracle, model: DubinsModel, V: np.ndarray,
                       X: np.ndarray, active: np.ndarray, witness_u: np.ndarray,
                       d_grid: np.ndarray, tol: float = 1e-12) -> dict:
    """Check C4 on the oracle witness action pi_HJ."""
    Xa = X[active]
    ua = witness_u[active]
    if len(Xa) == 0:
        return {"active": 0, "pass": True, "min_margin": None,
                "max_abs_identity_error": None}
    qmin = np.full(len(Xa), np.inf)
    vmin = np.full(len(Xa), np.inf)
    max_abs = 0.0
    for d in d_grid:
        q = oracle.q_star(V, Xa, ua, np.full(len(Xa), d))
        vf = oracle.interp_V(V, model.step(Xa, ua, np.full(len(Xa), d)))
        max_abs = max(max_abs, float(np.max(np.abs(q - vf))))
        np.minimum(qmin, q, out=qmin)
        np.minimum(vmin, vf, out=vmin)
    margin = vmin - qmin
    return {
        "active": int(len(Xa)),
        "pass": bool(np.all(margin >= -tol)),
        "min_margin": float(np.min(margin)),
        "max_abs_identity_error": max_abs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--n-mc", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    (REPO / cfg.out_dir).mkdir(parents=True, exist_ok=True)
    dyn, cert = cfg.dynamics, cfg.cert
    gamma = cfg.train.gamma_deploy
    eps = cert.eps_margin
    t0 = time.time()

    z = np.load(out_path(cfg, "oracle.npz"))
    if "gamma_teach" in z and abs(float(z["gamma_teach"]) - cfg.train.gamma_teach) > 1e-12:
        raise ValueError("oracle.npz gamma_teach does not match config train.gamma_teach")
    V = z["V"]
    G = z["g"]
    u_grid = z["u"]
    d_grid = z["d"]
    u_hj_grid = z["u_hj"]
    q_witness_grid = z["q_robust"]

    oracle = DubinsOracle(dyn, cfg.oracle, gamma=cfg.train.gamma_teach)
    model = DubinsModel(dyn)
    menu = _menu(dyn, cert)

    # Grid-node strict-spec checks.
    PX, PY, PSI = np.meshgrid(oracle.grid.px, oracle.grid.py, oracle.grid.psi,
                              indexing="ij")
    X_grid = np.stack([PX, PY, PSI], axis=-1).reshape(-1, 3)
    V_grid = V.reshape(-1)
    G_grid = G.reshape(-1)
    u_hj_flat = u_hj_grid.reshape(-1)
    q_witness_flat = q_witness_grid.reshape(-1)
    active_grid = V_grid >= 0.0

    c1_bad_grid = active_grid & (G_grid < 0.0)
    c3_margin_grid = q_witness_flat - gamma * V_grid - eps

    # Monte-Carlo strict-spec checks on the oracle interpolation.
    rng = np.random.default_rng(args.seed)
    X_mc = np.column_stack([
        rng.uniform(dyn.p_lo, dyn.p_hi, args.n_mc),
        rng.uniform(dyn.p_lo, dyn.p_hi, args.n_mc),
        rng.uniform(-np.pi, np.pi, args.n_mc),
    ])
    V_mc = oracle.interp_V(V, X_mc)
    G_mc = model.g(X_mc)
    active_mc = V_mc >= 0.0
    c1_bad_mc = active_mc & (G_mc < 0.0)
    q_witness_mc, u_hj_mc = _oracle_witness(oracle, V, X_mc, u_grid, d_grid)
    c3_margin_mc = q_witness_mc - gamma * V_mc - eps

    report = {
        "kind": "oracle_ground_truth_strict_spec_numerical_test",
        "config": str(args.config),
        "config_hash": cfg.hash(),
        "gamma_deploy": gamma,
        "gamma_teach": cfg.train.gamma_teach,
        "eps_margin": eps,
        "oracle_definition": "Q_HJ(x,u,d) = V_HJ(f(x,u,d))",
        "witness_definition": "pi_HJ(x) = argmax_u min_d Q_HJ(x,u,d) over oracle.u",
        "note": "Numerical grid/Monte-Carlo diagnostic matching neural C1/C3/C4 semantics; not a CROWN continuous-cell proof.",
        "grid": {
            "n_states": int(V_grid.size),
            "active_v_ge_0": int(active_grid.sum()),
            "c1": {
                "pass": bool(c1_bad_grid.sum() == 0),
                "bad": int(c1_bad_grid.sum()),
                "bad_frac_active": float(c1_bad_grid.sum() / max(active_grid.sum(), 1)),
            },
            "c3_witness_gate": _margin_summary(c3_margin_grid, active_grid),
            "c4_menu_q_le_vsucc": _c4_menu_report(
                oracle, model, V, X_grid, active_grid, menu, d_grid),
            "c4_witness_q_le_vsucc": _c4_witness_report(
                oracle, model, V, X_grid, active_grid, u_hj_flat, d_grid),
        },
        "monte_carlo": {
            "n": int(args.n_mc),
            "seed": int(args.seed),
            "active_v_ge_0": int(active_mc.sum()),
            "c1": {
                "pass": bool(c1_bad_mc.sum() == 0),
                "bad": int(c1_bad_mc.sum()),
                "bad_frac_active": float(c1_bad_mc.sum() / max(active_mc.sum(), 1)),
            },
            "c3_witness_gate": _margin_summary(c3_margin_mc, active_mc),
            "c4_menu_q_le_vsucc": _c4_menu_report(
                oracle, model, V, X_mc, active_mc, menu, d_grid),
            "c4_witness_q_le_vsucc": _c4_witness_report(
                oracle, model, V, X_mc, active_mc, u_hj_mc, d_grid),
        },
        "overall": {},
        "wall_s": None,
    }

    report["overall"] = {
        "grid_c1_c3_c4_pass": bool(
            report["grid"]["c1"]["pass"]
            and report["grid"]["c3_witness_gate"]["fail"] == 0
            and report["grid"]["c4_menu_q_le_vsucc"]["pass"]
            and report["grid"]["c4_witness_q_le_vsucc"]["pass"]
        ),
        "mc_c1_c3_c4_pass": bool(
            report["monte_carlo"]["c1"]["pass"]
            and report["monte_carlo"]["c3_witness_gate"]["fail"] == 0
            and report["monte_carlo"]["c4_menu_q_le_vsucc"]["pass"]
            and report["monte_carlo"]["c4_witness_q_le_vsucc"]["pass"]
        ),
    }
    report["wall_s"] = time.time() - t0

    save_json(cfg, "oracle_spec_report.json", report)
    update_manifest(cfg, "oracle_spec", {
        "grid_pass": report["overall"]["grid_c1_c3_c4_pass"],
        "mc_pass": report["overall"]["mc_c1_c3_c4_pass"],
        "grid_c1_bad": report["grid"]["c1"]["bad"],
        "grid_c3_fail": report["grid"]["c3_witness_gate"]["fail"],
        "mc_c1_bad": report["monte_carlo"]["c1"]["bad"],
        "mc_c3_fail": report["monte_carlo"]["c3_witness_gate"]["fail"],
        "wall_s": round(report["wall_s"], 1),
    })

    print("[oracle-spec] ground-truth numerical C1/C3/C4")
    print(f"  grid active V>=0: {report['grid']['active_v_ge_0']}/{report['grid']['n_states']}")
    print(f"  grid C1 bad: {report['grid']['c1']['bad']}")
    print(f"  grid C3 fail: {report['grid']['c3_witness_gate']['fail']}")
    print(f"  grid C3 min margin: {report['grid']['c3_witness_gate']['margin_min']:+.6f}")
    print(f"  grid C4 menu pass: {report['grid']['c4_menu_q_le_vsucc']['pass']}")
    print(f"  grid C4 witness pass: {report['grid']['c4_witness_q_le_vsucc']['pass']}")
    print(f"  MC active V>=0: {report['monte_carlo']['active_v_ge_0']}/{args.n_mc}")
    print(f"  MC C1 bad: {report['monte_carlo']['c1']['bad']}")
    print(f"  MC C3 fail: {report['monte_carlo']['c3_witness_gate']['fail']}")
    print(f"  MC C3 min margin: {report['monte_carlo']['c3_witness_gate']['margin_min']:+.6f}")
    print(f"  MC C4 menu pass: {report['monte_carlo']['c4_menu_q_le_vsucc']['pass']}")
    print(f"  MC C4 witness pass: {report['monte_carlo']['c4_witness_q_le_vsucc']['pass']}")
    print(f"  overall grid pass: {report['overall']['grid_c1_c3_c4_pass']}")
    print(f"  overall MC pass: {report['overall']['mc_c1_c3_c4_pass']}")
    print(f"  wrote {out_path(cfg, 'oracle_spec_report.json')}")


if __name__ == "__main__":
    main()
