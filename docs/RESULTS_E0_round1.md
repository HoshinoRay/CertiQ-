# Dubins E0 — Verifier-in-the-loop certified-training round (2026-06-14)

Consolidated result of the certified-training round on the learned Q-CBF artifact
`(V_theta, Q_theta, pi_theta)` for the fixed-speed Dubins car, certified with the
sound cell-worst CROWN/IBP verifier (Theorem A: C1 `{V>=0} ⊆ K`, C3 witness gate,
C4 `min_d Q ≤ min_d V(f)`).  All training is verifier-in-the-loop on the SAME
sound interval bounds the verifier checks (IBP-pass ⇒ deployed-CROWN-pass); none
of it enters the trusted computing base, and none of it uses the forbidden
Lipschitz / weight-decay / spectral flattening.

## Metrics

Per-gate pass rates are the cell-worst CROWN pass fractions on a sampled active
set.  The **headline coverage metric is the JOINT certified (safe) set** — the
cells that pass **C1 ∧ C3 ∧ C4 together** (on one COMMON sample) — reported as a
cell count and as `rho = certified_volume / Vol(Omega*)`.  Per-gate rates can look
healthy while the joint set is empty, so the joint size is the honest bottom line.
(Computed by `run_learned_spec_diagnostic.py`, section `joint_certified_set`.)

**Which C4?**  There are two C4 conditions: C4_menu (every menu action) and
C4_witness (the witness `pi`).  The SOUND safety set is `C1 ∧ C3 ∧ C4_witness`:
deploy the witness (which C3 guarantees is always gate-feasible) and
`min_d V(f) >= min_d Q(x,pi,d) >= gamma V >= 0`.  C4_menu is NOT required for
safety — it only gives minimum-intervention, and menu actions are not feasible
everywhere, so the pure `C4_menu ∨ C4_witness` UNION over-counts (menu-only cells
are unsafe at states where no menu action is feasible).  The earlier intersection
`C3 ∧ C4_menu ∧ C4_witness` was over-strict.  Empirically it is moot: the
C3-passing and C4-passing cells are essentially DISJOINT, so on the learned runs
`rho = 0` under EVERY rule (intersection / union / witness-only / menu-only),
while C3-only `rho` is 0.28 (V0.12) / 0.43 (C1-floor).  The binding problem is the
cell-level C3 ⊥ C4 anti-correlation, not the menu/witness combination.

## Results table

| config (lever)                     | C1 bad | C3 % | C4 menu % | C4 witn % | JOINT cells | rho |
|------------------------------------|-------:|-----:|----------:|----------:|------------:|----:|
| OFF (no certified lever)           |  3762  |  7.6 |   0.0     |   0.0     |     0       | 0.0 |
| one-sided C4 only                  |  3762  |  0.0 |  99.0     |  86.6     |     0       | 0.0 |
| two-sided C4+C3 (V0.12)            |  3762  | 27   |  21       |   7.4     |     0       | 0.0 |
| two-sided + v_dec_margin 0.25      |  2293  | 33.6 |  16.6     |   3.2     |     0       | 0.0 |
| two-sided + cell-worst V (decrease)| 30079  | 90.8 |   0.3     |   1.5     |     0       | 0.0 |
| two-sided + C1-floor-only V-cert   |   *23* | 45.7 |  11.0     |   2.0     |     0       | 0.0 |

(OFF = matched `cert_*_w=0` baseline.  All pass rates are cell-worst CROWN on
~1.5–2k sampled active cells; JOINT on a common sample.)

## What works (positive signals)

1. **Two-sided Q lever is sound and effective.** Verifier-in-the-loop CROWN-IBP
   eps-schedule training takes a previously-0% gate to high pass on the REAL
   CROWN verifier: one-sided drives C4 menu/witness 0% → 99%/87%; the two-sided
   variant (adding the C3 lower-gate push) is the only config where BOTH gates are
   simultaneously alive, and it resolves the C3/C4 tension as a mechanism (Q is
   squeezed into the band `[gamma*ubV+eps, lb V(f)-m]`).

2. **C1-floor-only V-cert is sound and effective.** The cell-worst C1 floor
   (`ub_IBP V(C) < -m` on every `g<0` cell, full coverage, teacher anchor on safe
   cells only, decrease push OFF) drives **C1 bad 3762 → 23** and sharpens
   `{V>=0}` strictly inside K — the cleanest C1 yet, and it lifts C3 (27 → 46%).

## The walls (why JOINT is still 0)

* **C3/C4 tension (Q side):** C4 wants Q low (`ub Q ≤ lb V(f)`), C3 wants Q high
  (`min_d Q(x,pi,d) ≥ gamma*V+eps`).  The two-sided lever balances them but at a
  level where neither is ~100% on the same cells.
* **Band-vs-slack / cannot widen by training (V side):** the CBF band cannot be
  widened by training — a bigger *pointwise* decrease margin roughens V and
  inflates its cell slack (`v_dec_margin 0.25` backfired); a cell-worst decrease
  push inflates V globally and destroys C1 (the `30079`-bad row).  The residual
  C4 slack is a *verifier* quantity (reducible only by finer cells).
* **C1-vs-C4 tension (V boundary):** pushing `{V>=0}` inside K (C1) lowers the
  boundary successor values C4 leans on, so fixing C1 (3762 → 23) *worsened* C4
  (menu 21 → 11%, witness 7.4 → 2.0%).
* **All-or-nothing strict spec:** the certificate accepts the inner set only if
  EVERY active cell passes C3 ∧ C4, so the joint must reach ~100% on active —
  far from the current per-gate rates.

## Ground-truth reference (V_HJ, Q_HJ = V_HJ o f, pi_HJ)

Running the SAME conditions on the oracle ground-truth object, to separate "spec
hardness" from "learned-model quality":

|                                   | C1 bad | C3 % | C4 %        | JOINT rho vs Omega* |
|-----------------------------------|-------:|-----:|------------:|--------------------:|
| ground truth -- POINTWISE (grid)  |    0   | 72.2 | 100 (exact) |        0.722        |
| ground truth -- CELL-WORST        |  small | low  |   ~0        |        ~0           |

Two structural facts:

1. **Pointwise ceiling rho ~ 0.72.**  Even the perfect teacher only certifies 72%
   of Omega* pointwise -- C3 fails on 28% of active grid nodes near the {V>=0}
   boundary, where the deployed gamma=0.90 gate is tight (min margin -0.135).  So
   ~0.72 is the pointwise ceiling for ANY value function at these settings; it is
   not a learned-model artifact.

2. **Cell-worst rho ~ 0 for the ideal Q, by construction.**  Because
   Q_HJ ≡ V_HJ o f EXACTLY, the cell-worst C4 check
   `ub Q(cell) <= lb V(f)(cell)` becomes `ub(V o f) <= lb(V o f)`, which fails by
   the V_HJ cell spread (mean ~0.48, p95 ~0.94 over the 40^3 cert lattice).  So
   the ideal Q has cell-worst C4 margin = -(spread) < 0 STRUCTURALLY -> C4 ~0%,
   joint rho ~0.  The ideal teacher is un-certifiable under the cell-worst verifier.

Implication: **the learned one-sided Q (trained Q <= V(f) - m) is the RIGHT object,
not a degraded one** -- the below-V(f) gap is exactly what cell-worst C4 needs and
what Q_HJ = V o f lacks (this is why the C4-only lever reached 99% where the ideal
Q gets 0%).  The certificate's walls -- the ~0.48 cell-worst slack and the
deployed-gamma C3 boundary (rho ceiling 0.72) -- bind the teacher too, so they are
inherent to the spec/verifier at this resolution, not learned imperfection.

## Bottom line

Every lever advances its own gate soundly and without flattening, but **no config
yet yields a single jointly-certified cell (JOINT = 0, rho = 0 under intersection,
union, witness-only AND menu-only)** — the C3- and C4-passing cells are essentially
disjoint.  For reference, **C3-only rho is 0.284 (V0.12) / 0.433 (C1-floor)** and
the pointwise ground-truth ceiling is 0.722, so the whole gap is the cell-level
C3 ⊥ C4 anti-correlation under cell-worst slack.  The closest state is the
C1-floor-only run: C1 essentially solved (23), C3 ~46%, with **C4-witness (~2%) the
binding wall**.

Next candidates (not yet run): finer verifier cells to shrink the C4 cell slack;
a softer / band-aware C1 floor to relieve the C1-vs-C4 boundary tension; and/or an
inner sub-region certificate (accept the passing sub-mask) to report a non-zero
`rho` instead of the strict all-or-nothing pass.
