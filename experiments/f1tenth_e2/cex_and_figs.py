"""Experiment C (false-feasibility counterexample bank) + the closing figures.

All EXACT pointwise (no CROWN) on the frozen baseline trio -- this is diagnostic
+ illustration, not the certificate (the certificate is run_cert_rfc.py).

  python experiments/f1tenth_e2/cex_and_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.dynamics.bicycle_accel import BicycleAccelConfig, BicycleAccelModel
from experiments.f1tenth_e2.distill import GAMMA, distill, clip_action, d_probe_grid, racing_steer

OUT = REPO / "results" / "f1tenth_e2"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)


def racing_action(cfg, X):
    v = X[:, 3]
    return np.column_stack([np.where(v < cfg.v_max, cfg.a_max, 0.0), racing_steer(cfg, X)])


def brake_traj_safe_points(cfg, model, v_net, X, n_stop=22):
    """EXACT pointwise: braking from each state keeps V_theta>=0 & g>=0 to stop."""
    Xc = X.copy()
    ok = np.ones(len(X), bool)
    for _ in range(n_stop):
        ok &= (v_net.forward(Xc[:, [0, 1, 3]])[:, 0] >= 0.0) & (model.g(Xc) >= 0.0)
        u = np.column_stack([np.full(len(Xc), cfg.a_min), racing_steer(cfg, Xc)])
        d = np.column_stack([np.full(len(Xc), cfg.d_a_max), np.zeros(len(Xc))])  # worst v+
        Xc = model.step(Xc, u, d)
        if (Xc[:, 3] <= 1e-9).all():
            break
    ok &= (v_net.forward(Xc[:, [0, 1, 3]])[:, 0] >= 0.0) & (model.g(Xc) >= 0.0)
    return ok


def min_d_Q(cfg, q_net, X, U):
    dprobe = d_probe_grid(cfg)
    qm = np.full(len(X), np.inf)
    for dk in dprobe:
        q = q_net.forward(np.concatenate([X, U, np.tile(dk, (len(X), 1))], 1))[:, 0]
        qm = np.minimum(qm, q)
    return qm


# --------------------------------------------------------------------------- #
def counterexample_bank(cfg, model, v, q, pi, n=400_000, seed=3):
    """Experiment C: states where the racing predicate Phi holds but the racing
    successor is NOT brake-safe (leaves S_brake^Q or collides)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(cfg.p_lo, cfg.p_hi, (n, 2)); ps = rng.uniform(-np.pi, np.pi, n)
    vv = rng.uniform(0.0, cfg.v_max, n)
    X = np.column_stack([p[:, 0], p[:, 1], ps, vv])
    Vx = v.forward(X[:, [0, 1, 3]])[:, 0]
    inset = Vx >= 0.0
    X, Vx = X[inset], Vx[inset]
    U = racing_action(cfg, X)
    qmin = min_d_Q(cfg, q, X, U)
    phi = qmin >= GAMMA * Vx                                  # racing Q-feasible
    # worst-case racing successor: d_a=+d_a_max (max v+), d_delta worst for g
    bad = np.zeros(len(X), bool); Vf_w = np.full(len(X), np.inf); gf_w = np.full(len(X), np.inf)
    for dd in (-cfg.d_delta_max, 0.0, cfg.d_delta_max):
        d = np.column_stack([np.full(len(X), cfg.d_a_max), np.full(len(X), dd)])
        Xn = model.step(X, U, d)
        Vf = v.forward(Xn[:, [0, 1, 3]])[:, 0]; gf = model.g(Xn)
        bs = brake_traj_safe_points(cfg, model, v, Xn)        # successor brake-safe?
        unsafe = (gf < 0.0) | ~bs
        bad |= unsafe
        Vf_w = np.minimum(Vf_w, Vf); gf_w = np.minimum(gf_w, gf)
    cex = phi & bad                                           # Phi true but successor unsafe
    margin = qmin - GAMMA * Vx                                # how feasible Phi looks
    stats = {
        "n_inset": int(len(X)), "n_phi": int(phi.sum()),
        "N_false": int(cex.sum()),
        "false_feas_rate_among_phi": float(cex.sum() / max(phi.sum(), 1)),
        "min_g_successor_on_cex": float(gf_w[cex].min()) if cex.any() else None,
        "min_Q_minus_gammaV_on_cex": float(margin[cex].min()) if cex.any() else None,
        "median_Q_minus_gammaV_on_cex": float(np.median(margin[cex])) if cex.any() else None,
    }
    k = min(int(cex.sum()), 20000)
    ci = np.flatnonzero(cex)[:k]
    np.savez(OUT / "cex_bank.npz", x=X[ci], u=U[ci], Vx=Vx[ci], qmin=qmin[ci],
             Vf=Vf_w[ci], gf=gf_w[ci])
    (OUT / "cex_stats.json").write_text(json.dumps(stats, indent=2))
    print("[cex]", json.dumps(stats, indent=2), flush=True)
    return X, U, Vx, phi, cex, Vf_w, gf_w


# --------------------------------------------------------------------------- #
def fig_safe_slices(cfg, model, v, q, pi, v_levels=(0.5, 1.5, 2.0), npx=240):
    """Ω* (gray) vs S_brake (blue) vs S_brake^Q (green) on (px,py) at fixed v."""
    gx = np.linspace(cfg.p_lo, cfg.p_hi, npx)
    PX, PY = np.meshgrid(gx, gx, indexing="ij")
    psis = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    fig, axes = plt.subplots(1, len(v_levels), figsize=(4.2 * len(v_levels), 4))
    for ax, vl in zip(np.atleast_1d(axes), v_levels):
        flat = np.column_stack([PX.ravel(), PY.ravel(), np.zeros(PX.size), np.full(PX.size, vl)])
        Va = model.brake_cbf(flat)                            # analytic Omega*
        # worst over heading: S_brake (brake-safe all psi), Q-live (some psi feasible)
        brake_ok = np.ones(PX.size, bool); qlive = np.zeros(PX.size, bool)
        Vth = v.forward(flat[:, [0, 1, 3]])[:, 0]
        for psi in psis:
            X = flat.copy(); X[:, 2] = psi
            brake_ok &= brake_traj_safe_points(cfg, model, v, X)
            U = np.column_stack([np.full(PX.size, cfg.a_min), racing_steer(cfg, X)])
            qlive |= (min_d_Q(cfg, q, X, U) >= GAMMA * Vth)
        img = np.zeros(PX.size)
        img[Va >= 0] = 1                                      # Omega* gray
        img[brake_ok] = 2                                     # S_brake blue
        img[brake_ok & qlive] = 3                             # S_brake^Q green
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["white", "0.8", "#5b8def", "#2ca02c"])
        ax.imshow(img.reshape(npx, npx).T, origin="lower", cmap=cmap, vmin=0, vmax=3,
                  extent=[cfg.p_lo, cfg.p_hi, cfg.p_lo, cfg.p_hi])
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cfg.obs_radius * np.cos(th), cfg.obs_radius * np.sin(th), "k", lw=1)
        ax.plot(cfg.world_radius * np.cos(th), cfg.world_radius * np.sin(th), "k", lw=1)
        ax.set_title(f"v = {vl:.1f} m/s"); ax.set_xlabel("px"); ax.set_aspect("equal")
    axes[0].set_ylabel("py")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="0.8", label="Ω* (analytic)"),
                        Patch(color="#5b8def", label="S_brake (verified)"),
                        Patch(color="#2ca02c", label="S_brake^Q (Q-live)")],
               loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "safe_slices.png", dpi=130)
    plt.close(fig); print("[fig] safe_slices.png", flush=True)


def fig_false_feasibility(cfg, X, U, phi, cex, Vf_w, gf_w):
    """Racing transitions x -> f(x,u,d): Phi-feasible but successor collides."""
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(cfg.obs_radius * np.cos(th), cfg.obs_radius * np.sin(th), "k", lw=1)
    ax.plot(cfg.world_radius * np.cos(th), cfg.world_radius * np.sin(th), "k", lw=1)
    k = np.flatnonzero(cex)[:400]
    model = BicycleAccelModel(cfg)
    d = np.column_stack([np.full(len(k), cfg.d_a_max), np.full(len(k), cfg.d_delta_max)])
    Xn = model.step(X[k], U[k], d)
    for i in range(len(k)):
        ax.plot([X[k[i], 0], Xn[i, 0]], [X[k[i], 1], Xn[i, 1]], "-", color="r", alpha=0.25, lw=0.6)
    ax.scatter(X[k, 0], X[k, 1], s=4, c="orange", label="Φ-feasible state (Q_θ says race OK)")
    ax.scatter(Xn[:, 0], Xn[:, 1], s=4, c="r", label="racing successor (collides / unsafe)")
    ax.set_title("False feasibility: learned Q_θ admits unsafe racing")
    ax.set_xlabel("px"); ax.set_ylabel("py"); ax.set_aspect("equal"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "false_feasibility.png", dpi=130)
    plt.close(fig); print("[fig] false_feasibility.png", flush=True)


def fig_resolution_and_audit():
    res = [44, 56]
    rho_brake = [0.802, 0.809]; rho_brakeQ = [0.779, 0.803]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(res, rho_brake, "o-", label="ρ_brake", color="#5b8def")
    a1.plot(res, rho_brakeQ, "s-", label="ρ_brake^Q", color="#2ca02c")
    a1.axhline(0.75, ls=":", color="0.5"); a1.set_ylim(0.6, 1.0)
    a1.set_xlabel("grid edge"); a1.set_ylabel("ρ (of Ω*)"); a1.set_title("Resolution stability")
    a1.legend(); a1.grid(alpha=0.3)
    labels = ["fallback-pinned\n(S_brake^Q)", "naive racing\n(baseline Q_θ)", "conservative Q_θ\n(c2fix)"]
    cbv = [0, 205, 3144]
    a2.bar(labels, cbv, color=["#2ca02c", "#d62728", "#d62728"])
    a2.set_ylabel("certified-but-violated"); a2.set_title("Audit (cbv must be 0 to certify)")
    for i, c in enumerate(cbv):
        a2.text(i, c + 40, str(c), ha="center", fontsize=9)
    a2.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "resolution_and_audit.png", dpi=130)
    plt.close(fig); print("[fig] resolution_and_audit.png", flush=True)


# --------------------------------------------------------------------------- #
def main():
    cfg = BicycleAccelConfig(); model = BicycleAccelModel(cfg)
    print("[distill] baseline trio (seed 0) ...", flush=True)
    v, q, pi, _ = distill(cfg, 0, margin=0.10, n_samples=80_000, reg_epochs=30,
                          cbf_epochs=30, verbose=False)
    X, U, Vx, phi, cex, Vf_w, gf_w = counterexample_bank(cfg, model, v, q, pi)
    fig_safe_slices(cfg, model, v, q, pi)
    fig_false_feasibility(cfg, X, U, phi, cex, Vf_w, gf_w)
    fig_resolution_and_audit()
    print("done ->", FIG, flush=True)


if __name__ == "__main__":
    main()
