"""Attribution figures for the E2-L learned Gate-D result (reads the report JSON).

  fig1  brake-undershoot vs cell size (route-A binding quantity) -- shows it
        shrinks with resolution but stays > 0, while the analytic ideal V's
        structural undershoot is ~0 (the exact 1-Lipschitz cancellation).
  fig2  route-B cell-reachability erosion: certified cells vs GFP iteration for
        BOTH the learned and the analytic-IDEAL candidate -- both collapse to 0,
        proving it is the cell+box obstruction, not the learned net.
  fig3  per-margin: undershoot + C3 hole rate vs m -- m moves C3 (liveness) only,
        never the route-A/B safety boundary.

    python experiments/f1tenth_e2/figures_learned.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "f1tenth_e2"
FIG = OUT / "figures"


def main():
    rep = json.loads((OUT / "e2_learned_report.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    margins = rep["config"]["margins"]
    seeds = rep["config"]["seeds"]

    # ---- fig1: undershoot vs cell size ---------------------------------- #
    ra = rep.get("res_attribution", [])
    if ra:
        cell = [a["cell_m"] for a in ra]
        us = [a["undershoot_med"] for a in ra]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(cell, us, "o-", color="C3", lw=2, label="learned $V_\\theta$ (CROWN box)")
        ax.axhline(0.0, color="C0", ls="--", lw=1.5,
                   label="analytic CBF (exact 1-Lipschitz) $\\approx 0$")
        ax.set_xlabel("position cell size (m)")
        ax.set_ylabel("brake-successor undershoot  $ub\\,V_\\theta-lb\\,V_\\theta(x^+)$")
        ax.set_title("Route A binding quantity vs resolution\n(>0 always -> heading-free sub-level never closes)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "fig1_undershoot_vs_resolution.png", dpi=140)
        plt.close(fig)

    # ---- fig2: route-B reachability erosion ----------------------------- #
    key = f"m{margins[len(margins)//2]:.2f}_s{seeds[0]}"
    cert = rep["results"][key]["cert"]
    hL = cert.get("routeB_hist_learned", [])
    hA = cert.get("routeB_hist_analytic", [])
    if hL and hA:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(range(len(hL)), hL, "s-", color="C3", lw=2, label="learned candidate")
        ax.plot(range(len(hA)), hA, "o-", color="C0", lw=2, label="analytic IDEAL candidate")
        ax.set_xlabel("Tarski GFP iteration")
        ax.set_ylabel("accepted 4-D cells")
        ax.set_title("Route B cell-reachability erodes to 0\n(even for the analytic ideal V -> cell+box obstruction)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "fig2_reachability_erosion.png", dpi=140)
        plt.close(fig)

    # ---- fig3: undershoot + C3 hole vs margin --------------------------- #
    us_m = [rep["summary"][f"m{m:.2f}"]["undershoot_med_mean"] for m in margins]
    c3_m = [rep["summary"][f"m{m:.2f}"]["c3_hole_rate_mean"] for m in margins]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(margins, us_m, "o-", color="C3", lw=2, label="brake undershoot (route A/B)")
    ax.set_xlabel("contraction margin $m$ (finetune decrease hinge)")
    ax.set_ylabel("brake undershoot", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax2 = ax.twinx()
    ax2.plot(margins, c3_m, "s--", color="C2", lw=2, label="C3 hole rate (liveness)")
    ax2.set_ylabel("C3 hole rate", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title("m moves only C3 (liveness), not the safety boundary")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_margin_decoupling.png", dpi=140)
    plt.close(fig)

    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    sys.exit(main())
