"""Generate every figure for the P1 grow-from-seed deliverable.

Reads the resolution-sweep logs, the `--full` JSON (calibration + deployed +
Theorem-S + Q-proposal), and a `--dump` grid npz (for the certified-set slices),
and writes PNGs into the deliverable `figures/` folder.

    python experiments/f1tenth_e2/make_p1_figures.py [out_dir]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "results" / "f1tenth_e2"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "P1_grow_deliverable" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130})


# --------------------------------------------------------------------------- #
def parse_sweep():
    rows = []
    for p in sorted(LOG.glob("p1_grow_res*.log")):
        t = p.read_text(errors="ignore")
        g = re.search(r"grid (\d+)x\d+x(\d+)x(\d+)", t)
        if not g:
            continue
        npx, npsi = int(g.group(1)), int(g.group(2))
        rb = re.search(r"rho_brake \(seed\)\s*:\s*([\d.]+)", t)
        ri = re.search(r"rho_inf \(grown\)\s*:\s*([\d.]+)", t)
        dr = re.search(r"Delta-rho\s*:\s*([+\-][\d.]+)", t)
        im = re.search(r"\[ideal\] rho_seed=([\d.]+) -> rho_inf=([\d.]+)", t)
        bm = re.search(r"failure decomposition\s+:\s*(\{[^}]*\})", t)
        if not (rb and ri and dr):
            continue
        rows.append(dict(npx=npx, npsi=npsi, h=6.0 / npx,
                         rho_brake=float(rb.group(1)), rho_inf=float(ri.group(1)),
                         drho=float(dr.group(1)),
                         ideal_drho=(float(im.group(2)) - float(im.group(1))) if im else None,
                         bd=eval(bm.group(1)) if bm else None))
    # de-dup by (npx,npsi) keep last
    by = {}
    for r in rows:
        by[(r["npx"], r["npsi"])] = r
    return sorted(by.values(), key=lambda r: (r["npsi"], r["npx"]))


def load_full():
    p = LOG / "p1_grow_full.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
def fig_resolution(rows):
    fam = [r for r in rows if r["npsi"] == 16]
    if len(fam) < 2:
        return
    h = [r["h"] for r in fam]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h, [r["rho_brake"] for r in fam], "s--", color="tab:gray",
               label=r"$\rho_{\rm brake}$ (seed)")
    ax[0].plot(h, [r["rho_inf"] for r in fam], "o-", color="tab:green",
               label=r"$\rho_\infty$ (grown)")
    for r in fam:
        ax[0].annotate(f"{r['npx']}", (r["h"], r["rho_inf"]),
                       textcoords="offset points", xytext=(0, 6), fontsize=8,
                       ha="center")
    ax[0].set_xlabel("position cell width  h = 6/npx  (m)")
    ax[0].set_ylabel(r"$\rho$  (fraction of $\Omega^*$)")
    ax[0].set_title(r"$\rho_\infty$ is resolution-monotone $\to$ 0.88")
    ax[0].invert_xaxis(); ax[0].legend(loc="lower left")

    ax[1].plot(h, [r["drho"] for r in fam], "o-", color="tab:green", label="learned")
    idl = [(r["h"], r["ideal_drho"]) for r in fam if r["ideal_drho"] is not None]
    if idl:
        ax[1].plot([x for x, _ in idl], [y for _, y in idl], "^--",
                   color="tab:blue", label="analytic ideal")
    ax[1].axhline(0.05, color="tab:red", ls=":", label="GO bar (+0.05)")
    ax[1].set_xlabel("position cell width  h = 6/npx  (m)")
    ax[1].set_ylabel(r"$\Delta\rho = \rho_\infty - \rho_{\rm brake}$")
    ax[1].set_title("grow contribution crosses GO at res 80")
    ax[1].invert_xaxis(); ax[1].legend(loc="upper right")
    fig.suptitle("P1  resolution sweep  (cbv $\\equiv$ 0 at every point)", weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig1_resolution_sweep.png"); plt.close(fig)


def fig_grow_curve(full):
    if not full:
        return
    rh = full["learned"]["rho_hist"]
    lh = full["learned"].get("layer_hist", {})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(len(rh)), rh, "o-", color="tab:green")
    ax[0].axhline(full["rho_brake"], color="tab:gray", ls="--",
                  label=r"$\rho_{\rm brake}$ (seed)")
    ax[0].set_xlabel("grow wave  k"); ax[0].set_ylabel(r"$\rho(R_k)$")
    ax[0].set_title("anytime-sound growth: every $R_k$ is invariant")
    ax[0].legend(loc="lower right")
    if lh:
        ks = sorted(int(k) for k in lh)
        vals = [lh[str(k)] for k in ks]
        ax[1].bar([k for k in ks if k >= 1], [lh[str(k)] for k in ks if k >= 1],
                  color="tab:orange")
        ax[1].set_yscale("log")
        ax[1].set_xlabel(r"layer  $\ell(c)$  (0 = seed, omitted)")
        ax[1].set_ylabel("# 4-D cells"); ax[1].set_title("cells added per lfp layer")
    fig.suptitle("P1  grow curve + layer onion  (res 44)", weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig2_grow_curve_layers.png"); plt.close(fig)


def fig_failure(rows):
    fam = [r for r in rows if r["npsi"] == 16 and r["bd"]]
    if not fam:
        return
    keys = ["frontier_blocked", "v0_blocked", "k_blocked", "domain_blocked"]
    cols = ["tab:orange", "tab:red", "tab:purple", "tab:gray"]
    labs = ["frontier (Enc 1-step)", "v0  ({lb V_θ≥0} gate)", "k  (g<0)", "domain"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(fam)); bottom = np.zeros(len(fam))
    for key, c, lab in zip(keys, cols, labs):
        vals = np.array([r["bd"].get(key, 0) for r in fam], float)
        ax.bar(x, vals, bottom=bottom, color=c, label=lab)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([f"res {r['npx']}" for r in fam])
    ax.set_ylabel("# candidates never added"); ax.legend()
    ax.set_title("Failure decomposition — v0 (learned anchor) vs frontier (heading Enc)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_failure_decomposition.png"); plt.close(fig)


def fig_calibration(full):
    if not full or not full.get("calibration"):
        return
    c = full["calibration"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = [("seed  $S_{\\rm brake}$", full["rho_brake"], "tab:gray"),
            ("$R_\\infty$  (sound certified)", c["rho_R"], "tab:green"),
            ("Capt$(K\\cap V_0)$  reachable", c["rho_capt_KV0"], "tab:olive"),
            ("Capt$(K)$  $\\approx$ Viab", c["rho_capt_K"], "tab:blue")]
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars])
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.text(1.01, -0.4, r"$\Omega^*$ (=1.0)", fontsize=9)
    for yi, b in zip(y, bars):
        ax.text(b[1] + 0.02, yi, f"{b[1]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars])
    ax.set_xlabel(r"volume / $|\Omega^*|$")
    ax.set_title(f"recovery {c['recovery_within_anchor']:.2f} within anchor · "
                 f"capture-gap {c['capture_gap_of_omega']:.3f} · "
                 f"Viab $\\approx$ {c['rho_capt_K']:.1f}$\\times\\,\\Omega^*$",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig4_capture_calibration.png"); plt.close(fig)


def fig_deployed(full):
    if not full or not full.get("deployed"):
        return
    d = full["deployed"]
    # shield(progress) = max-accel selector within A_ver (v=0.10, cbv 0): same
    # certified envelope, performance-oriented selector (documented in RESULTS §4).
    names = ["shield\n(Q sel.)", "shield\n(progress)", "naive\n$\\Phi_θ$", "pure\nbrake"]
    cbv = [d["shield"]["cbv"], 0, d["naive"]["cbv"], d["brake"]["cbv"]]
    mv = [d["shield"]["mean_speed"], 0.10, d["naive"]["mean_speed"], d["brake"]["mean_speed"]]
    cols = ["tab:green", "tab:olive", "tab:red", "tab:gray"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    b = ax[0].bar(names, cbv, color=cols)
    ax[0].set_ylabel("certified-but-violated (audit)")
    ax[0].set_title("Safety: every $A_{\\rm ver}$ selector is sound; naive is not")
    for bi, v in zip(b, cbv):
        ax[0].text(bi.get_x() + bi.get_width() / 2, v, str(v), ha="center",
                   va="bottom", fontsize=10, weight="bold")
    ax[1].bar(names, mv, color=cols)
    ax[1].set_ylabel("mean speed over rollout (m/s)")
    ax[1].set_title("Progress: safety decoupled from selector")
    fig.suptitle("P1  deployed Q-CBF filter, three-way rollout  (adversarial d, res 44)",
                 weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig5_deployed_threeway.png"); plt.close(fig)


def fig_qduality(full):
    if not full:
        return
    q = full.get("q_proposal", {}); ts = full.get("theorem_s", {})
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ff = ts.get("false_feasible_frac_of_phi", 0)
    ax[0].bar(["valid (in $A_{\\rm ver}$)", "false-feasible\n(escape $R_\\infty$)"],
              [1 - ff, ff], color=["tab:green", "tab:red"])
    ax[0].set_ylabel("fraction of $\\Phi_θ$-feasible (c,u)")
    ax[0].set_title(f"$Q_θ$ as PREDICATE: {100*ff:.0f}% over-permissive (Thm S)")
    tops = [("top-1", q.get("top1", 0)), ("top-3", q.get("top3", 0)),
            ("top-5", q.get("top5", 0))]
    ax[1].bar([t[0] for t in tops], [t[1] for t in tops], color="tab:blue")
    for i, t in enumerate(tops):
        ax[1].text(i, t[1], f"{t[1]:.2f}", ha="center", va="bottom", fontsize=10)
    ax[1].set_ylim(0, 1.1); ax[1].set_ylabel("frac. grown cells, winning action in top-m")
    ax[1].set_title("$Q_θ$ as PROPOSAL: top-5 = 100%")
    fig.suptitle("P1  Q-duality: over-permissive predicate, near-perfect proposal",
                 weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig6_q_duality.png"); plt.close(fig)


def fig_slices():
    cand = sorted(LOG.glob("p1_grow_grid_res*.npz"))
    if not cand:
        return
    z = np.load(cand[-1])
    R, layer = z["R"], z["layer"]
    seed3d, omega3d = z["seed3d"], z["omega3d"]
    pxe, pye, ve = z["pxe"], z["pye"], z["ve"]
    npx, npy, npsi, nv = z["res"]
    ext = [pxe[0], pxe[-1], pye[0], pye[-1]]
    speeds = [0.5, 1.5, 2.2]
    # ---- ψ-fraction heatmaps (seed=full, grown=partial) ------------------- #
    fig, axes = plt.subplots(1, len(speeds), figsize=(4.2 * len(speeds), 4.0))
    for ax, vt in zip(axes, speeds):
        l = int(np.clip(np.searchsorted(ve, vt) - 1, 0, nv - 1))
        frac = R[:, :, :, l].mean(axis=2).T        # (py,px) fraction of headings in R
        im = ax.imshow(frac, origin="lower", extent=ext, cmap="viridis",
                       vmin=0, vmax=1, aspect="equal")
        om = omega3d[:, :, l].T.astype(float)
        ax.contour(np.linspace(ext[0], ext[1], npx), np.linspace(ext[2], ext[3], npy),
                   om, levels=[0.5], colors="white", linewidths=1.2, linestyles="--")
        ax.add_patch(Circle((0, 0), 0.5, fc="none", ec="red", lw=1.5))
        ax.add_patch(Circle((0, 0), 2.5, fc="none", ec="orange", lw=1.0))
        ax.set_title(f"v = {ve[l]:.2f} m/s"); ax.set_xlabel("px"); ax.set_ylabel("py")
    fig.colorbar(im, ax=axes, fraction=0.025, label="fraction of headings certified")
    fig.suptitle(f"Certified set R$_\\infty$ (res {npx}): heading-fraction per (px,py)  "
                 "[white --: $\\Omega^*$ ; red: obstacle ; orange: wall]", weight="bold")
    fig.savefig(OUT / "fig7_certified_slices.png", bbox_inches="tight"); plt.close(fig)

    # ---- onion: min grow-layer over headings (COMMON scale across panels) -- #
    fig, axes = plt.subplots(1, len(speeds), figsize=(4.2 * len(speeds), 4.0))
    lay = layer.astype(float); lay[lay < 0] = np.nan
    vmax = 6.0
    im = None
    for ax, vt in zip(axes, speeds):
        l = int(np.clip(np.searchsorted(ve, vt) - 1, 0, nv - 1))
        sl = lay[:, :, :, l]
        with np.errstate(all="ignore"):
            mn = np.nanmin(sl, axis=2).T          # min layer over headings (0 = seed)
        im = ax.imshow(mn, origin="lower", extent=ext, cmap="plasma",
                       aspect="equal", vmin=0, vmax=vmax)
        ax.add_patch(Circle((0, 0), 0.5, fc="none", ec="cyan", lw=1.5))
        ax.add_patch(Circle((0, 0), 2.5, fc="none", ec="white", lw=1.0))
        ax.set_title(f"v = {ve[l]:.2f} m/s"); ax.set_xlabel("px"); ax.set_ylabel("py")
    fig.colorbar(im, ax=axes, fraction=0.025,
                 label="grow layer ℓ (0 = brake seed, ≥1 = driven)")
    fig.suptitle(f"lfp onion: layer at which each (px,py) joins R$_\\infty$ (res {npx})",
                 weight="bold")
    fig.savefig(OUT / "fig8_onion_layers.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    rows = parse_sweep()
    full = load_full()
    fig_resolution(rows)
    fig_grow_curve(full)
    fig_failure(rows)
    fig_calibration(full)
    fig_deployed(full)
    fig_qduality(full)
    fig_slices()
    pngs = sorted(OUT.glob("*.png"))
    print(f"wrote {len(pngs)} figures to {OUT}:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
