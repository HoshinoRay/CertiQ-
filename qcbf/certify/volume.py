"""Reference-volume utilities for the Dubins CBVF/Q-CBF experiment."""
from __future__ import annotations

import numpy as np

from qcbf.config import ExperimentConfig


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
