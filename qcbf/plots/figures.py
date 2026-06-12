"""Paper figures (matplotlib / Agg).

  fig_certified_slices : psi-slices of the certified set vs the oracle
                         maximal robust invariant set Omega* = {V* >= 0}
  fig_c_sweep          : rho(c) and certified-cell counts over the c-sweep
  fig_margins          : certified C3 lower-bound histogram + lbV histogram
  fig_fixed_point      : |A_k| vs pruning iteration per c
  fig_oracle           : VI residual curve + V* slice
  fig_audit_traj       : closed-loop trajectories under the greedy adversary
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import DubinsModel

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "legend.fontsize": 8,
})

SLICES = (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi)
SLICE_NAMES = (r"$\psi = 0$", r"$\psi = \pi/2$", r"$\psi = \pi$",
               r"$\psi = -\pi/2$")


def _geometry(ax, cfg: ExperimentConfig):
    dyn = cfg.dynamics
    ax.add_patch(Circle(dyn.obs_center, dyn.obs_radius, fc="0.25", ec="k",
                        zorder=5))
    ax.add_patch(Circle((0, 0), dyn.world_radius, fc="none", ec="k",
                        ls="--", lw=1.0, zorder=5))
    ax.set_xlim(dyn.p_lo, dyn.p_hi)
    ax.set_ylim(dyn.p_lo, dyn.p_hi)
    ax.set_aspect("equal")


# --------------------------------------------------------------------------- #
def fig_certified_slices(cfg, lat, accepted, oracle, V_star, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    A = accepted.reshape(lat.nx, lat.ny, lat.npsi)
    n_fine = 220
    xs = np.linspace(cfg.dynamics.p_lo, cfg.dynamics.p_hi, n_fine)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    for ax, psi, name in zip(axes.ravel(), SLICES, SLICE_NAMES):
        ip = int(np.floor((psi + np.pi) / lat.hp)) % lat.npsi
        ext = (lat.p_lo, lat.p_hi, lat.p_lo, lat.p_hi)
        ax.imshow(A[:, :, ip].T, origin="lower", extent=ext,
                  cmap=matplotlib.colors.ListedColormap(["white", "#9fd4a3"]),
                  vmin=0, vmax=1, interpolation="nearest", zorder=1)
        P = np.stack([XX, YY, np.full_like(XX, psi)], axis=-1)
        VV = oracle.interp_V(V_star, P.reshape(-1, 3)).reshape(n_fine, n_fine)
        ax.contour(XX, YY, VV, levels=[0.0], colors="tab:blue",
                   linewidths=1.6, zorder=6)
        _geometry(ax, cfg)
        ax.set_title(f"{name}   (cells: {int(A[:, :, ip].sum())})")
    handles = [plt.Line2D([], [], color="tab:blue", lw=1.6,
                          label=r"oracle $\partial\Omega^*$ ($V^*=0$)"),
               plt.Rectangle((0, 0), 1, 1, fc="#9fd4a3",
                             label=r"certified cells $\Omega_{\rm cert}$"),
               plt.Line2D([], [], color="k", ls="--",
                          label="workspace boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Certified invariant set vs. oracle maximal RCI set", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_c_sweep(sweep: dict, path: str):
    cs = [e["c"] for e in sweep["entries"]]
    rho = [e["rho"] for e in sweep["entries"]]
    acc = [e["accepted"] for e in sweep["entries"]]
    init = [e["init"] for e in sweep["entries"]]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax[0].plot(cs, rho, "o-", color="tab:green")
    ax[0].set_xlabel("initialization level $c$")
    ax[0].set_ylabel(r"$\rho(c) = \mathrm{Vol}(\Omega_{\rm cert})/"
                     r"\mathrm{Vol}(\Omega^*)$")
    ax[0].set_ylim(0, 1)
    ax[1].plot(cs, init, "s--", color="0.5", label="initial cells")
    ax[1].plot(cs, acc, "o-", color="tab:green", label="certified cells")
    ax[1].set_xlabel("initialization level $c$")
    ax[1].set_ylabel("# cells")
    ax[1].legend()
    fig.suptitle("c-sweep (single precompute, fixed point per c)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_margins(pre, accepted, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    m = pre.c3_lb[accepted]
    m = m[np.isfinite(m)]
    ax[0].hist(m, bins=50, color="tab:green", alpha=0.8)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel(r"certified lower bound of $h_3$ on cell$\times D$")
    ax[0].set_ylabel("# certified cells")
    ax[0].set_title("C3 margin (already net of $\\varepsilon$)")
    lv = pre.lbV[accepted]
    ax[1].hist(lv[np.isfinite(lv)], bins=50, color="tab:blue", alpha=0.8)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel(r"certified lower bound $\hat{V}_\theta$ per cell")
    ax[1].set_title("value lower bounds")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_fixed_point(sweep: dict, path: str):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for e in sweep["entries"]:
        ax.plot(e["history"], "-o", ms=3, label=f"c={e['c']:.2f}")
    ax.set_xlabel("pruning iteration $k$")
    ax.set_ylabel(r"$|A_k|$")
    ax.legend(ncol=2)
    ax.set_title("C2 fixed-point convergence")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_oracle(cfg, oracle, sol: dict, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].semilogy(sol["history"])
    ax[0].set_xlabel("VI sweep")
    ax[0].set_ylabel(r"$\|V_{k+1} - V_k\|_\infty$")
    ax[0].set_title(f"Bellman-Isaacs VI ({sol['iters']} sweeps, "
                    f"{sol['wall_s']:.0f}s)")
    g = oracle.grid
    ip = int(np.argmin(np.abs(g.psi - 0.0)))
    im = ax[1].pcolormesh(g.px, g.py, sol["V"][:, :, ip].T, cmap="RdYlGn",
                          shading="nearest")
    ax[1].contour(g.px, g.py, sol["V"][:, :, ip].T, levels=[0.0],
                  colors="k", linewidths=1.4)
    _geometry(ax[1], cfg)
    ax[1].grid(False)
    ax[1].set_title(r"$V^*$ at $\psi = 0$")
    fig.colorbar(im, ax=ax[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_audit_traj(cfg, lat, accepted, filt, oracle, V_star, path: str,
                   n_traj: int = 14, horizon: int = 150, seed: int = 11):
    from qcbf.audit.falsify import sample_certified_states, _adversary_greedy
    model = DubinsModel(cfg.dynamics)
    rng = np.random.default_rng(seed)
    X = sample_certified_states(lat, accepted, n_traj, rng)
    d_grid = np.linspace(-cfg.dynamics.d_max, cfg.dynamics.d_max,
                         cfg.audit.adversary_d_grid)
    traj = [X.copy()]
    for _ in range(horizon):
        u, _, _ = filt.batch_select(X, np.zeros(len(X)))
        d = _adversary_greedy(model, lambda Z: filt.v(Z).ravel(), X, u, d_grid)
        X = model.step(X, u, d)
        traj.append(X.copy())
    T = np.stack(traj)                    # (H+1, n, 3)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    psi0 = 0.0
    ip = int(np.floor((psi0 + np.pi) / lat.hp)) % lat.npsi
    A = accepted.reshape(lat.nx, lat.ny, lat.npsi)
    ax.imshow(A[:, :, ip].T, origin="lower",
              extent=(lat.p_lo, lat.p_hi, lat.p_lo, lat.p_hi),
              cmap=matplotlib.colors.ListedColormap(["white", "#dff0e0"]),
              vmin=0, vmax=1, interpolation="nearest")
    for i in range(T.shape[1]):
        ax.plot(T[:, i, 0], T[:, i, 1], lw=0.9, alpha=0.9)
        ax.plot(T[0, i, 0], T[0, i, 1], "k.", ms=4)
    _geometry(ax, cfg)
    ax.grid(False)
    ax.set_title(f"closed loop under greedy adversary "
                 f"({n_traj} rollouts, {horizon} steps)\n"
                 r"background: certified cells at $\psi=0$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def make_all(cfg, lat, pre, accepted, sweep, oracle, V_star, sol, filt,
             out_dir: str) -> list[str]:
    paths = []
    p = f"{out_dir}/fig_certified_slices.png"
    fig_certified_slices(cfg, lat, accepted, oracle, V_star, p); paths.append(p)
    p = f"{out_dir}/fig_c_sweep.png"
    fig_c_sweep(sweep, p); paths.append(p)
    p = f"{out_dir}/fig_margins.png"
    fig_margins(pre, accepted, p); paths.append(p)
    p = f"{out_dir}/fig_fixed_point.png"
    fig_fixed_point(sweep, p); paths.append(p)
    p = f"{out_dir}/fig_oracle.png"
    fig_oracle(cfg, oracle, sol, p); paths.append(p)
    p = f"{out_dir}/fig_audit_traj.png"
    if int(np.count_nonzero(accepted)) > 0:
        fig_audit_traj(cfg, lat, accepted, filt, oracle, V_star, p)
        paths.append(p)
    return paths
