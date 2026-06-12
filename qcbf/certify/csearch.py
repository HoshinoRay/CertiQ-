"""c-sweep: certified sets for a family of initialization levels c.

Because C1, C3, lbV, the skip mask and the successor ranges are independent
of the accepted set, the entire sweep reuses one `CertPrecompute`; each c
costs only the (prefix-sum) fixed-point iterations.  The paper metric is

    rho(c) = Vol(Omega_cert(c)) / Vol(Omega*) ,

with Vol(Omega*) estimated by Monte-Carlo over the oracle's interpolated
value function (the oracle is a REFERENCE for tightness, never part of the
soundness argument).
"""
from __future__ import annotations

import numpy as np

from qcbf.config import ExperimentConfig
from qcbf.certify.refine import CertPrecompute, c2_fixed_point


# --------------------------------------------------------------------------- #
def omega_star_volume(oracle, V_star: np.ndarray, cfg: ExperimentConfig,
                      n_mc: int = 400_000, seed: int = 7) -> dict:
    """MC estimate of Vol(Omega*) = Vol({V* >= 0}) and the domain volume."""
    dyn = cfg.dynamics
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(dyn.p_lo, dyn.p_hi, n_mc),
        rng.uniform(dyn.p_lo, dyn.p_hi, n_mc),
        rng.uniform(-np.pi, np.pi, n_mc),
    ])
    vals = oracle.interp_V(V_star, X)
    frac = float(np.mean(vals >= 0.0))
    vol_dom = (dyn.p_hi - dyn.p_lo) ** 2 * 2.0 * np.pi
    return {"frac_of_domain": frac, "volume": frac * vol_dom,
            "domain_volume": vol_dom, "n_mc": n_mc}


# --------------------------------------------------------------------------- #
def run_c_sweep(pre: CertPrecompute, cfg: ExperimentConfig,
                omega_star_vol: float, verbose: bool = True
                ) -> tuple[dict, np.ndarray]:
    """Fixed points for every c in cfg.cert.c_sweep.

    Returns (sweep dict, accepted mask of the LARGEST certified set --
    by monotonicity that is the smallest c).
    """
    lat = pre.lattice
    cell_vol = lat.cell_volume
    entries = []
    best_mask = np.zeros(lat.n_cells, dtype=bool)
    for c in cfg.cert.c_sweep:
        acc, st = c2_fixed_point(pre, float(c), verbose)
        vol = st["accepted"] * cell_vol
        st["volume"] = vol
        st["rho"] = vol / omega_star_vol if omega_star_vol > 0 else 0.0
        st["frac_of_init"] = (st["accepted"] / st["init"]) if st["init"] else 0.0
        entries.append(st)
        if st["accepted"] > int(best_mask.sum()):
            best_mask = acc
    sweep = {"entries": entries,
             "gate_d_pass": bool(max(e["accepted"] for e in entries) > 0)}
    return sweep, best_mask
