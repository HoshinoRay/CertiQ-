"""Falsification audit (dev guide Sec. 12; Gate D evidence).

The audit attacks the *deployed closed loop* (certified filter + dynamics)
from initial states sampled inside the certified set, under three
disturbance models of increasing hostility:

  iid       d_t ~ Unif(D)                       (nominal stochastic)
  extremal  d_t in {-d_max, +d_max} random sign (bang-bang)
  greedy    d_t = argmin_{d in grid} V_theta(f(x_t, u_t, d))
            (a best-response adversary against the learned certificate,
             evaluated AFTER the filter commits to u_t)

Pass criterion: ZERO certified-but-violated events, where a violation is
  * any g(x_t) < 0 along the horizon (safety), or
  * any x_t leaving the certified cell union (invariance).

All rollouts of a mode run in lockstep (vectorized over episodes).
"""
from __future__ import annotations

import time

import numpy as np

from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import DubinsModel
from qcbf.runtime.filter import CertifiedFilter


# --------------------------------------------------------------------------- #
def sample_certified_states(lat, accepted: np.ndarray, n: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Uniform states inside the certified cell union."""
    ids = np.flatnonzero(accepted)
    pick = rng.choice(ids, size=n, replace=True)
    ip = pick % lat.npsi
    iy = (pick // lat.npsi) % lat.ny
    ix = pick // (lat.npsi * lat.ny)
    u3 = rng.uniform(size=(n, 3))
    return np.column_stack([
        lat.p_lo + (ix + u3[:, 0]) * lat.hx,
        lat.p_lo + (iy + u3[:, 1]) * lat.hy,
        -np.pi + (ip + u3[:, 2]) * lat.hp,
    ])


# --------------------------------------------------------------------------- #
def _adversary_greedy(model: DubinsModel, v_fn, x: np.ndarray,
                      u: np.ndarray, d_grid: np.ndarray) -> np.ndarray:
    """d = argmin_d V_theta(f(x, u, d)) per rollout (vectorized)."""
    B = len(x)
    K = len(d_grid)
    xr = np.repeat(x, K, axis=0)
    ur = np.repeat(u, K)
    dr = np.tile(d_grid, B)
    nxt = model.step(xr, ur, dr)
    vals = v_fn(nxt).reshape(B, K)
    return d_grid[np.argmin(vals, axis=1)]


def run_audit(cfg: ExperimentConfig, model: DubinsModel,
              filt: CertifiedFilter, verbose: bool = True) -> dict:
    aud = cfg.audit
    rng = np.random.default_rng(aud.seed)
    lat = filt.lat
    d_grid = np.linspace(-cfg.dynamics.d_max, cfg.dynamics.d_max,
                         aud.adversary_d_grid)
    v_fn = lambda X: filt.v(X).ravel()

    report: dict = {"modes": {}}
    for mode in ("iid", "extremal", "greedy"):
        t0 = time.time()
        X = sample_certified_states(lat, filt.accepted, aud.n_rollouts, rng)
        B = len(X)
        alive_safe = np.ones(B, dtype=bool)      # g >= 0 so far
        alive_inv = np.ones(B, dtype=bool)       # in certified set so far
        min_g = model.g(X).copy()
        fb_steps = 0
        margins = []
        for t in range(aud.horizon):
            u, used_fb, margin = filt.batch_select(X, np.zeros(B))
            fb_steps += int(used_fb.sum())
            margins.append(margin)
            if mode == "iid":
                d = rng.uniform(-cfg.dynamics.d_max, cfg.dynamics.d_max, B)
            elif mode == "extremal":
                d = rng.choice([-cfg.dynamics.d_max, cfg.dynamics.d_max], B)
            else:
                d = _adversary_greedy(model, v_fn, X, u, d_grid)
            X = model.step(X, u, d)
            g = model.g(X)
            np.minimum(min_g, g, out=min_g)
            alive_safe &= g >= 0.0
            alive_inv &= filt.is_certified(X)
        n_unsafe = int((~alive_safe).sum())
        n_escape = int((~alive_inv).sum())
        report["modes"][mode] = {
            "n_rollouts": B,
            "horizon": aud.horizon,
            "violations_safety": n_unsafe,
            "violations_invariance": n_escape,
            "min_g_over_all": float(min_g.min()),
            "fallback_step_frac": fb_steps / (B * aud.horizon),
            "mean_best_margin": float(np.mean(margins)),
            "wall_s": time.time() - t0,
        }
        if verbose:
            r = report["modes"][mode]
            print(f"  [audit:{mode:8s}] safety viol {n_unsafe}, invariance "
                  f"viol {n_escape}, min g {r['min_g_over_all']:+.4f}, "
                  f"fallback {100*r['fallback_step_frac']:.2f}% "
                  f"({r['wall_s']:.1f}s)")
    report["certified_but_violated"] = int(sum(
        m["violations_safety"] for m in report["modes"].values()))
    report["invariance_violations"] = int(sum(
        m["violations_invariance"] for m in report["modes"].values()))
    report["pass"] = (report["certified_but_violated"] == 0
                      and report["invariance_violations"] == 0)
    return report
