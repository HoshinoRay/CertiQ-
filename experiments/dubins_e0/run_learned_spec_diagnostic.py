"""Diagnostic C1/C3/C4 checks on the learned artifact.

This is a diagnostic companion to ``run_certify.py``.  The official certificate
may fail fast when C1 fails, because the strict theorem is already false.  This
script keeps going on a deterministic subset of active cells so we can see how
C3 and C4 behave for the learned

    (V_theta, Q_theta, pi_theta)

artifact.  It uses the same bound helpers and the same C1/C3/C4 semantics as
the neural strict verifier; it does not introduce an old C2 check.
"""
from __future__ import annotations

import argparse
import time

from common import REPO, file_hash, out_path, save_json, update_manifest

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.spec import (
    _check_q_consistency,
    _check_witness_q_consistency,
    _menu,
)
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import g_bounds_on_box
from qcbf.nets.mlp import MLP
from qcbf.verify.bounds import SeqNet
from qcbf.verify.compiler import compile_h3
from qcbf.verify.conditions import check_c3_staged, v_cell_bounds


def _bool_summary(mask: np.ndarray, n_total: int) -> dict:
    return {
        "pass": int(mask.sum()),
        "fail": int(n_total - mask.sum()),
        "pass_frac": float(mask.sum() / max(n_total, 1)),
    }


def _margin_summary(margin: np.ndarray, ok: np.ndarray) -> dict:
    if len(margin) == 0:
        return {
            "pass": 0,
            "fail": 0,
            "pass_frac": None,
            "margin_min": None,
            "margin_p01": None,
            "margin_p05": None,
            "margin_mean": None,
        }
    return {
        "pass": int(ok.sum()),
        "fail": int((~ok).sum()),
        "pass_frac": float(ok.mean()),
        "margin_min": float(np.min(margin)),
        "margin_p01": float(np.percentile(margin, 1)),
        "margin_p05": float(np.percentile(margin, 5)),
        "margin_mean": float(np.mean(margin)),
    }


def _sample_idx(idx: np.ndarray, max_cells: int | None,
                rng: np.random.Generator) -> np.ndarray:
    if max_cells is None or max_cells <= 0 or len(idx) <= max_cells:
        return idx
    pick = rng.choice(len(idx), size=max_cells, replace=False)
    return np.sort(idx[pick])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--max-c3-cells", type=int, default=2048,
                    help="Deterministic sample size for expensive C3 diagnostics; <=0 means all active cells.")
    ap.add_argument("--max-c4-cells", type=int, default=2048,
                    help="Deterministic sample size for expensive C4 diagnostics; <=0 means all active cells.")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    (REPO / cfg.out_dir).mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rng = np.random.default_rng(args.seed)

    v = MLP.load(str(out_path(cfg, "v.npz")))
    q = MLP.load(str(out_path(cfg, "q.npz")))
    pi = MLP.load(str(out_path(cfg, "pi.npz")))
    v_net = SeqNet.from_mlp(v)
    q_net = SeqNet.from_mlp(q)
    h3_net = compile_h3(pi, q, v, cfg.train.gamma_deploy, cfg.cert.eps_margin,
                        cfg.dynamics.control_max)

    dyn, cert = cfg.dynamics, cfg.cert
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    n = lat.n_cells
    all_idx = np.arange(n)
    menu = _menu(dyn, cert)

    wall: dict[str, float] = {}
    t0 = time.time()
    lbV, ubV = v_cell_bounds(v_net, boxes, all_idx, n, cert.chunk)
    wall["v_bounds"] = time.time() - t0

    t0 = time.time()
    gmin, _ = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1],
                              boxes[:, 2], boxes[:, 3])
    superlevel_possible = ubV >= 0.0
    superlevel_inner = lbV >= 0.0
    c1_bad = superlevel_possible & (gmin < 0.0)
    active_idx_all = np.flatnonzero(superlevel_possible & ~c1_bad)
    wall["c1"] = time.time() - t0

    c3_idx = _sample_idx(active_idx_all, args.max_c3_cells, rng)
    c4_idx = _sample_idx(active_idx_all, args.max_c4_cells, rng)

    t0 = time.time()
    c3_ok, c3_lb, c3_funnel = check_c3_staged(
        h3_net, boxes, c3_idx, dyn, cert, n, verbose=True)
    wall["c3_sample"] = time.time() - t0

    t0 = time.time()
    q_ok_local, q_margin_local = _check_q_consistency(
        v_net, q_net, boxes, c4_idx, cfg, menu)
    wall["c4_menu_sample"] = time.time() - t0

    t0 = time.time()
    w_ok_local, w_margin_local = _check_witness_q_consistency(
        v_net, h3_net, boxes, c4_idx, ubV, cfg)
    wall["c4_witness_sample"] = time.time() - t0

    c3_local_ok = c3_ok[c3_idx]
    c3_local_margin = c3_lb[c3_idx]

    report = {
        "kind": "learned_strict_spec_diagnostic",
        "config": str(args.config),
        "config_hash": cfg.hash(),
        "v_hash": file_hash(out_path(cfg, "v.npz")),
        "q_hash": file_hash(out_path(cfg, "q.npz")),
        "pi_hash": file_hash(out_path(cfg, "pi.npz")),
        "gamma_deploy": cfg.train.gamma_deploy,
        "eps_margin": cert.eps_margin,
        "note": "Same neural C1/C3/C4 semantics as strict verifier; C3/C4 are sampled diagnostics when max cells is positive. No C2 check.",
        "lattice": {
            "n_cells": int(n),
            "shape": [cert.n_cells_px, cert.n_cells_py, cert.n_cells_psi],
            "menu": [float(u) for u in menu],
        },
        "c1": {
            "superlevel_possible": int(superlevel_possible.sum()),
            "superlevel_inner": int(superlevel_inner.sum()),
            "bad": int(c1_bad.sum()),
            "bad_frac_possible": float(c1_bad.sum() / max(superlevel_possible.sum(), 1)),
            "pass": bool(c1_bad.sum() == 0),
        },
        "active_after_c1": int(len(active_idx_all)),
        "c3_witness_gate_sample": {
            "sampled": int(len(c3_idx)),
            "sample_mode": "all" if len(c3_idx) == len(active_idx_all) else "random_without_replacement",
            "funnel": {k: int(v) for k, v in c3_funnel.items()},
            **_margin_summary(c3_local_margin, c3_local_ok),
        },
        "c4_menu_q_le_vsucc_sample": {
            "sampled": int(len(c4_idx)),
            "sample_mode": "all" if len(c4_idx) == len(active_idx_all) else "random_without_replacement",
            **_margin_summary(q_margin_local, q_ok_local),
        },
        "c4_witness_q_le_vsucc_sample": {
            "sampled": int(len(c4_idx)),
            "sample_mode": "all" if len(c4_idx) == len(active_idx_all) else "random_without_replacement",
            **_margin_summary(w_margin_local, w_ok_local),
        },
        "overall": {
            "strict_spec_pass": bool(
                c1_bad.sum() == 0
                and len(c3_idx) == len(active_idx_all)
                and np.all(c3_local_ok)
                and len(c4_idx) == len(active_idx_all)
                and np.all(q_ok_local)
                and np.all(w_ok_local)
            ),
            "diagnostic_full_active_coverage": bool(
                len(c3_idx) == len(active_idx_all)
                and len(c4_idx) == len(active_idx_all)
            ),
        },
        "wall_s_stages": {k: round(v, 1) for k, v in wall.items()},
        "wall_s_total": round(time.time() - t_all, 1),
    }

    save_json(cfg, "learned_spec_diagnostic_report.json", report)
    update_manifest(cfg, "learned_spec_diagnostic", {
        "c1_bad": report["c1"]["bad"],
        "c3_sample_fail": report["c3_witness_gate_sample"]["fail"],
        "c4_menu_sample_fail": report["c4_menu_q_le_vsucc_sample"]["fail"],
        "c4_witness_sample_fail": report["c4_witness_q_le_vsucc_sample"]["fail"],
        "wall_s": report["wall_s_total"],
    })

    print("[learned-diagnostic] neural C1/C3/C4")
    print(f"  C1 possible cells: {report['c1']['superlevel_possible']}/{n}")
    print(f"  C1 bad: {report['c1']['bad']}")
    print(f"  active after C1: {report['active_after_c1']}")
    print(f"  C3 sampled: {report['c3_witness_gate_sample']['sampled']}, "
          f"fail {report['c3_witness_gate_sample']['fail']}, "
          f"min margin {report['c3_witness_gate_sample']['margin_min']:+.6f}")
    print(f"  C4 menu sampled: {report['c4_menu_q_le_vsucc_sample']['sampled']}, "
          f"fail {report['c4_menu_q_le_vsucc_sample']['fail']}, "
          f"min margin {report['c4_menu_q_le_vsucc_sample']['margin_min']:+.6f}")
    print(f"  C4 witness sampled: {report['c4_witness_q_le_vsucc_sample']['sampled']}, "
          f"fail {report['c4_witness_q_le_vsucc_sample']['fail']}, "
          f"min margin {report['c4_witness_q_le_vsucc_sample']['margin_min']:+.6f}")
    print(f"  wrote {out_path(cfg, 'learned_spec_diagnostic_report.json')}")


if __name__ == "__main__":
    main()
