"""Shared utilities for the E0 stage scripts.

Every stage writes a manifest entry containing the config hash, so any
certificate on disk can be traced to the exact configuration (and hence
dynamics, networks and verifier settings) that produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.config import ExperimentConfig  # noqa: E402


def load_cfg() -> ExperimentConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/"
                                            "config_pilot.json"))
    args = ap.parse_args()
    cfg = ExperimentConfig.load(args.config)
    out = REPO / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    return cfg


def out_path(cfg: ExperimentConfig, name: str) -> Path:
    return REPO / cfg.out_dir / name


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def update_manifest(cfg: ExperimentConfig, stage: str, info: dict) -> None:
    p = out_path(cfg, "manifest.json")
    man = json.loads(p.read_text()) if p.exists() else {
        "experiment": cfg.name, "config_hash": cfg.hash(), "stages": {}}
    assert man["config_hash"] == cfg.hash(), \
        "manifest belongs to a different config; clear the output directory"
    man["stages"][stage] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            **info}
    p.write_text(json.dumps(man, indent=2))


def save_json(cfg: ExperimentConfig, name: str, obj: dict) -> None:
    out_path(cfg, name).write_text(json.dumps(obj, indent=2))
