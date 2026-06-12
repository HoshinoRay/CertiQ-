"""P1 -- direct-composition C2 primitive (no intermediate outward-rounded box).

The eroding route (Route 2 / run_cert_learned.py) computes an outward successor
box and then either CROWN-bounds V_theta over it or rounds it to a cell lattice.
Both inject the box over-approximation between the dynamics and V_theta.  Route 1
(dev doc Sec 8.5) instead bounds  V_theta(f(x,u,d))  *directly*.

For the heading-independent learned value V_theta(px, py, v) and the braking
fallback f_brake, this module realises Route 1 soundly and cheaply:

  1. Build a sound successor box B  ⊇  f_brake(cell × Psi × D).  B is used ONLY
     to pick CROWN's ReLU relaxation slopes -- it is never rounded to cells.
  2. CROWN gives an affine lower functional valid over all of B:
         V_theta(y)  >=  A·y + beta      ∀ y ∈ B          (crown_lower_affine).
  3. Substitute the TRUE nonlinear successor y = f_brake(.) into that functional
     and minimise  A·f_brake + beta  over cell × Psi × D analytically.  Because
     f_brake(cell) ⊆ B, the functional is valid there, and
         min_{cell} A·f_brake + beta   >=   min_{y∈B} A·y + beta
     so this is sound AND tighter-or-equal than the box route.  The gain is the
     (px+, py+) heading correlation: a_px·v·cos ψ + a_py·v·sin ψ is one sinusoid
     in ψ, whose worst case dominates the sum of the two axis-wise worst cases.

The lower bound returned is  inf_{x∈cell, ψ∈Psi, d∈D} V_theta(f_brake(x,u=a_min,d)).
"""
from __future__ import annotations

import numpy as np

from qcbf.dynamics.bicycle_accel import successor_box, brake_distance
from qcbf.dynamics.dubins import cos_interval
from qcbf.verify.bounds import (SeqNet, crown_bounds, crown_lower_affine,
                                crown_upper_affine)


def crown_brake_successor_lb(cfg, v_net: SeqNet,
                             px_lo, px_hi, py_lo, py_hi,
                             psi_lo, psi_hi, v_lo, v_hi,
                             tighten: bool = True) -> np.ndarray:
    """Sound lower bound of V_theta over the braking successor of each
    (px,py,v) cell crossed with heading box [psi_lo,psi_hi] and full D.

    All inputs broadcast to a common shape (M,).  Returns (M,) lower bounds.
    """
    px_lo = np.asarray(px_lo, float); px_hi = np.asarray(px_hi, float)
    py_lo = np.asarray(py_lo, float); py_hi = np.asarray(py_hi, float)
    v_lo = np.asarray(v_lo, float);   v_hi = np.asarray(v_hi, float)
    psi_lo = np.broadcast_to(np.asarray(psi_lo, float), px_lo.shape)
    psi_hi = np.broadcast_to(np.asarray(psi_hi, float), px_lo.shape)

    # ---- (1) sound successor box (braking fallback, full D) -- slopes only -- #
    bx0, bx1, by0, by1, bv0, bv1 = successor_box(
        cfg, px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi, v_lo, v_hi,
        cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)

    # ---- (2) CROWN affine lower functional  V_theta(y) >= A·y + beta -------- #
    A, beta = crown_lower_affine(v_net,
                                 np.column_stack([bx0, by0, bv0]),
                                 np.column_stack([bx1, by1, bv1]), tighten)
    a_px, a_py, a_v = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    beta = beta[:, 0]

    # ---- (3) minimise A·f_brake + beta over the TRUE successor ------------- #
    # position contributes a_px·px + a_py·py (linear, exact corner)
    t_px = np.where(a_px >= 0.0, a_px * px_lo, a_px * px_hi)
    t_py = np.where(a_py >= 0.0, a_py * py_lo, a_py * py_hi)

    # heading-coupled displacement: dt · min_{v,ψ} v·(a_px cos ψ + a_py sin ψ)
    #   a_px cos ψ + a_py sin ψ = R cos(ψ - φ),  R = |A_p|,  range over Psi:
    R = np.hypot(a_px, a_py)
    phi = np.arctan2(a_py, a_px)
    cmin, cmax = cos_interval(psi_lo - phi, psi_hi - phi)
    g_lo, g_hi = R * cmin, R * cmax                  # range of the sinusoid (R>=0)
    # bilinear v·g over [v_lo,v_hi]≥0 × [g_lo,g_hi] -> minimum at a corner
    corners = np.stack([v_lo * g_lo, v_lo * g_hi, v_hi * g_lo, v_hi * g_hi], 0)
    t_disp = cfg.dt * corners.min(axis=0)

    # v+ range under braking (monotone in v) -> exact corner for a_v·v+
    t_v = np.where(a_v >= 0.0, a_v * bv0, a_v * bv1)

    return t_px + t_py + t_disp + t_v + beta


# --------------------------------------------------------------------------- #
# RELATIONAL decrease bound (guide 5.1) -- the lynchpin of the recursive-
# feasibility certificate.  Lower-bounds the DECREASE
#
#     G(x,d) = V_theta(f_brake(x,d)) - rho_level * V_theta(x)
#
# over (px,py,v) cell x heading-box x D as ONE expression sharing the cell input
# x between the two V_theta copies.  The successor copy uses the CROWN lower
# functional (A_f over the successor box); the current copy uses the CROWN UPPER
# functional (A0 over the cell, rho_level >= 0).  Combining the POSITION terms
# BEFORE the per-axis minimisation, the cell-oscillation of V_theta cancels:
# the px coefficient is (a_px - rho*b_px), whose cell variation
# |a_px - rho*b_px|*width is small when the two functionals' slopes align across
# one displacement -- instead of the decoupled (|a_px| + rho*|b_px|)*width that
# carries osc_C(V_theta) twice.  rho_level = 1 => hard non-decrease G >= 0.
#
# The Mode-A relational successor lower bound is then
#     L_rel(cell) = lbV(cell) + G_lb(cell, rho=1)  <=  V_theta(f_brake)
# (sum of two box-universal lower bounds), used to certify the terminal core
# {V_theta >= c_term} braking-invariant: L_rel >= c_term.
# --------------------------------------------------------------------------- #
def crown_relational_decrease_lb(cfg, v_net: SeqNet,
                                 px_lo, px_hi, py_lo, py_hi,
                                 psi_lo, psi_hi, v_lo, v_hi,
                                 rho_level: float = 1.0,
                                 tighten: bool = True,
                                 chunk: int = 8192) -> np.ndarray:
    """Sound lower bound of G = V_theta(f_brake(x,d)) - rho_level*V_theta(x) over
    each (px,py,v) cell crossed with heading box [psi_lo,psi_hi] and full D, under
    the braking fallback (a = a_min, worst d_a).  rho_level >= 0.  Returns (M,).
    """
    px_lo = np.asarray(px_lo, float); px_hi = np.asarray(px_hi, float)
    py_lo = np.asarray(py_lo, float); py_hi = np.asarray(py_hi, float)
    v_lo = np.asarray(v_lo, float);   v_hi = np.asarray(v_hi, float)
    psi_lo = np.broadcast_to(np.asarray(psi_lo, float), px_lo.shape)
    psi_hi = np.broadcast_to(np.asarray(psi_hi, float), px_lo.shape)
    n = len(px_lo)
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sl = slice(s, e)
        # ---- (1) successor box (braking fallback, full D) -- slopes only ---- #
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, px_lo[sl], px_hi[sl], py_lo[sl], py_hi[sl],
            psi_lo[sl], psi_hi[sl], v_lo[sl], v_hi[sl],
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)

        # ---- (2) lower functional of V at successor, UPPER at the cell ------ #
        Af, bf = crown_lower_affine(v_net, np.column_stack([bx0, by0, bv0]),
                                    np.column_stack([bx1, by1, bv1]), tighten)
        A0, b0 = crown_upper_affine(v_net, np.column_stack([px_lo[sl], py_lo[sl], v_lo[sl]]),
                                    np.column_stack([px_hi[sl], py_hi[sl], v_hi[sl]]), tighten)
        a_px, a_py, a_v = Af[:, 0, 0], Af[:, 0, 1], Af[:, 0, 2]
        b_px, b_py, b_v = A0[:, 0, 0], A0[:, 0, 1], A0[:, 0, 2]
        bf = bf[:, 0]; b0 = b0[:, 0]

        # ---- (3) minimise A_f.f(x) - rho*(A0.x + b0) over cell x Psi x D ----- #
        # position: combined coefficient (cancellation happens here)
        cpx = a_px - rho_level * b_px
        cpy = a_py - rho_level * b_py
        t_px = np.where(cpx >= 0.0, cpx * px_lo[sl], cpx * px_hi[sl])
        t_py = np.where(cpy >= 0.0, cpy * py_lo[sl], cpy * py_hi[sl])
        # heading-coupled displacement dt * min_{v,psi} v*(a_px cos psi + a_py sin psi)
        R = np.hypot(a_px, a_py)
        phi = np.arctan2(a_py, a_px)
        cmin, cmax = cos_interval(psi_lo[sl] - phi, psi_hi[sl] - phi)
        g_lo, g_hi = R * cmin, R * cmax
        corners = np.stack([v_lo[sl] * g_lo, v_lo[sl] * g_hi,
                            v_hi[sl] * g_lo, v_hi[sl] * g_hi], 0)
        t_disp = cfg.dt * corners.min(axis=0)
        # successor speed a_v * v+  (v+ monotone -> corner by sign)
        t_vp = np.where(a_v >= 0.0, a_v * bv0, a_v * bv1)
        # current speed -rho*b_v * v  (corner by sign of the coefficient)
        cv = -rho_level * b_v
        t_vc = np.where(cv >= 0.0, cv * v_lo[sl], cv * v_hi[sl])
        out[sl] = t_px + t_py + t_disp + t_vp + t_vc + (bf - rho_level * b0)
    return out
# (analytic braking distance D; learned, ~1-Lipschitz clearance net C_theta).
# The speed axis is EXACT, so only C_theta's spatial CROWN gap enters -- and the
# braking cancellation is structural, so the EXACT contraction is ~0 (not the
# black-box -0.076 floor), hence a fine-enough sound certificate can close.
# --------------------------------------------------------------------------- #
def vtheta_box_bounds_structured(cfg, c_net: SeqNet, px_lo, px_hi, py_lo, py_hi,
                                 v_lo, v_hi, chunk=4096, progress=None):
    """Sound (lb, ub) of V_theta = C_theta(p) - D(v) - margin over (px,py,v) boxes.

    C_theta by CROWN (2-D net); D(v) exact (monotone increasing in v>=0)."""
    from qcbf.verify.bounds import crown_bounds_chunked
    Clo, Chi = crown_bounds_chunked(c_net, np.column_stack([px_lo, py_lo]),
                                    np.column_stack([px_hi, py_hi]), True, chunk,
                                    progress=progress)
    Clo, Chi = Clo[:, 0], Chi[:, 0]
    D_lo = brake_distance(cfg, np.minimum(v_lo, v_hi))
    D_hi = brake_distance(cfg, np.maximum(v_lo, v_hi))
    return Clo - D_hi - cfg.cbf_margin, Chi - D_lo - cfg.cbf_margin


def crown_brake_successor_lb_structured(cfg, c_net: SeqNet,
                                        px_lo, px_hi, py_lo, py_hi,
                                        psi_lo, psi_hi, v_lo, v_hi,
                                        chunk: int = 8192) -> np.ndarray:
    """Sound lower bound of V_theta(f_brake) for the STRUCTURED value, direct
    composition (no outward-rounded box): lower(C_theta(p+)) - D(max v+) - margin.

    p+ = p + dt v [cos psi, sin psi];  v+ in [.,nv_hi] (max v+ = least braking =
    worst for D).  C_theta(p+) lower-bounded by its CROWN affine functional
    minimised over the TRUE successor (heading-coupled, as crown_brake_successor_lb).
    Chunked over cells (the CROWN affine tensor is O(chunk * width^2)).
    """
    px_lo = np.asarray(px_lo, float); px_hi = np.asarray(px_hi, float)
    py_lo = np.asarray(py_lo, float); py_hi = np.asarray(py_hi, float)
    v_lo = np.asarray(v_lo, float); v_hi = np.asarray(v_hi, float)
    psi_lo = np.broadcast_to(np.asarray(psi_lo, float), px_lo.shape)
    psi_hi = np.broadcast_to(np.asarray(psi_hi, float), px_lo.shape)
    n = len(px_lo)
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sl = slice(s, e)
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, px_lo[sl], px_hi[sl], py_lo[sl], py_hi[sl],
            psi_lo[sl], psi_hi[sl], v_lo[sl], v_hi[sl],
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        A, beta = crown_lower_affine(c_net, np.column_stack([bx0, by0]),
                                     np.column_stack([bx1, by1]), True)
        a_px, a_py, beta = A[:, 0, 0], A[:, 0, 1], beta[:, 0]
        t_px = np.where(a_px >= 0.0, a_px * px_lo[sl], a_px * px_hi[sl])
        t_py = np.where(a_py >= 0.0, a_py * py_lo[sl], a_py * py_hi[sl])
        R = np.hypot(a_px, a_py)
        phi = np.arctan2(a_py, a_px)
        cmin, cmax = cos_interval(psi_lo[sl] - phi, psi_hi[sl] - phi)
        g_lo, g_hi = R * cmin, R * cmax
        corners = np.stack([v_lo[sl] * g_lo, v_lo[sl] * g_hi,
                            v_hi[sl] * g_lo, v_hi[sl] * g_hi], 0)
        t_disp = cfg.dt * corners.min(axis=0)
        C_lb = t_px + t_py + t_disp + beta        # lower bound of C_theta(p+)
        # max v+ over the cell (least braking) -> largest D -> smallest V_theta(f)
        out[sl] = C_lb - brake_distance(cfg, bv1) - cfg.cbf_margin
    return out
