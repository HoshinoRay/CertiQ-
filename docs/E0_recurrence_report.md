# Dubins-E0 — Recurrence Certificate: Clean Main-Line Report

*Fixed-speed Dubins car, sound cell-worst CROWN/IBP verifier, 40³ certificate
lattice (64000 cells) unless noted.  This document keeps only the clear main
line and drops the exploratory dead-ends.  All numbers are reproducible from the
scripts named per section.*

## 0. Object, set, and denominators

- **Plant.** `x=(px,py,ψ)`, `px⁺=px+Δt·v·cos ψ`, `py⁺=py+Δt·v·sin ψ`,
  `ψ⁺=wrap(ψ+Δt·(u+d))`; `Δt=0.1, v=1, u∈[−1,1], d∈[−0.3,0.3]`.
  Safe set `K={g≥0}`, `g=min(‖p−o‖²−r²_obs, R²_world−‖p‖²)`,
  `o=(0,0), r_obs=0.45, R_world=1.8`, domain `[−2,2]²×S¹`.
- **Witness (deployed fallback).** `u=clip(π_θ(x))`, frozen.
- **Reference set.** `Ω*={V*≥0}` from the oracle HJ value (the maximal robust
  controlled-invariant set); `Vol(Ω*)/Vol(domain)=0.44`. All `ρ = Vol(·)/Vol(Ω*)`.
- **Certified set semantics (Theorem A, the correct framing).** The certified set is
  `Σ = ⋃{Cᵢ : C1ᵢ ∧ C3ᵢ ∧ C4ᵢ}` with the **post-closure** reading of C4
  (`Reach(Cᵢ,π_θ,D) ⊆ Σ`). With `C1: Σ⊆K`, `C3: π_θ feasible on Σ`, `C4: successor
  back into Σ`, a one-line induction gives robust forward invariance + safety.
  The shrink-refinement T4 (`Σ_{k+1}=Σ_k∩Pre(Σ_k)`) is **the algorithm** that finds
  the largest such Σ; it is not a separate theorem.

---

## 1. Stage 1 — discounted Q-CBF certificate: per-gate rates nonzero, joint set = 0

The original spec verifies, on `{V_θ≥0}`, `C1 {V_θ≥0}⊆K`, `C3` witness gate
`min_d Q_θ(x,π,d) ≥ γ_deploy V_θ + ε`, and `C4 min_d Q_θ ≤ min_d V_θ(f)`
(γ_deploy=0.90). Strict cert is all-or-nothing over active cells. Per-gate
cell-worst CROWN pass rates (`run_learned_spec_diagnostic.py`):

| config (lever)                       | C1 bad | C3 % | C4 menu % | C4 witn % | JOINT ρ |
|--------------------------------------|-------:|-----:|----------:|----------:|--------:|
| OFF (no certified lever)             |  3762  |  7.6 |   0.0     |   0.0     |  0.00   |
| one-sided C4 only                    |  3762  |  0.0 |  99.0     |  86.6     |  0.00   |
| two-sided C4+C3 (V0.12)              |  3762  | 27   |  21       |   7.4     |  0.00   |
| two-sided + v_dec 0.25               |  2293  | 33.6 |  16.6     |   3.2     |  0.00   |
| two-sided + cell-worst-V (decrease)  | 30079  | 90.8 |   0.3     |   1.5     |  0.00   |
| two-sided + C1-floor-only V-cert     |  *23*  | 45.7 |  11.0     |   2.0     |  0.00   |

**Result: the JOINT certified set is 0 (ρ=0) in every config.** The C3-passing
and C4-passing cells are essentially disjoint (C3 wants Q high, `Q≥γV+ε`; C4 wants
Q low, `Q≤V(f)`), so no cell passes both, and the strict spec accepts nothing.

---

## 2. Stage 2 — recurrence (drop the discount, clamp the value): one-step pass ≈ 0.6

Reformulate to a single barrier with **no discount**: `W=min(g, value)`,
`S_m={W≥m}`, one condition `min_d W(f(x,π_θ,d)) ≥ m` (witness u=clip(π_θ)). This
removes the C3⊥C4 tension (one one-sided lower-bound condition) and makes
`{W≥0}⊆K` structural via the clamp. `run_recurrence_cert.py`.

**The reported number is the one-step recurrence PASS RATE** = (active cells
satisfying the condition)/(active cells). This is *not* a certified set by itself
(see §3). At m=0:

| value object (W=min(g,·))                 | resolution | recurrence pass % | note |
|-------------------------------------------|:----------:|------------------:|------|
| frozen V0.12                              |    40³     |   **59.2**        | inner-vol ρ({W≥0})=0.74 |
| frozen C1-floor V                         |    40³     |   ~54             | |
| **trained W̃ (route-1, §below)**          |    40³     |   **82.9**        | wleak→0 |
| **trained W̃ (route-1)**                  |    80³     |   **90.0**        | finer cells tighten slack |
| ground truth V_HJ — *pointwise* (grid)    |   grid     |   **96.4**        | pointwise (optimistic), not cell-worst |
| ground truth V_HJ — *cell-worst*          |    40³     |   ~58             | sound; ad-hoc estimate |

Notes on the two ground-truth rows: the **pointwise 0.964** is the fraction of Ω*
grid nodes where `max_u min_d V_HJ(f) ≥ 0` holds; it is a *pointwise* pass-fraction
of Ω* (no cell-worst bounds), so it is an optimistic ceiling, near-tautological for
the oracle (the HJ fixed point gives `q_robust ≥ V_HJ ≥ 0` on Ω* by construction).
The apples-to-apples **sound cell-worst** ground-truth number is ~0.58.

**Route-1 training (`train_w_recurrence`, branch `clamped-w-recurrence`).** Trains
`W̃` (V frozen-as-warm-start, π_θ frozen) with a bounded directional hinge on the
verifier's own IBP quantity (UP: raise `lb_IBP W̃(f)` to m where the successor is
g-feasible; DOWN: exclude physical g-leaks), ε-scheduled point→cell. It took the
CROWN recurrence pass **59%→82.9% (40³) / 90% (80³)** with the W̃-fixable leaks
driven to ~0; the residual ~17% (40³) / ~10% (80³) are exactly the **g-leaks**
(cells whose witness successor's `g<m`), which `W̃` provably cannot fix.

---

## 3. Stage 3 — the gap: the local pass is **not closed**; the certified set is 0

The §2 number is a **one-step** pass (successor lands in the over-set `{W≥0}`),
not the post-closure C4 (successor lands back in the certified set Σ). Checking
the closure directly on the 0.6 set (V0.12, m=0):

```
active |{W≥0}|            = 32515
one-step pass |P| (=0.6)  = 19245   (59.2%)
of P, successor stays in P:  13824   →  P loses 5421 cells in ONE closure step
```

So the 0.6 set is **not self-closed**. Iterating the closure (T4,
`run_recurrence_cert.py --t4`) erodes it to **empty**:

| object         | resolution | T4 greatest fixed point (certified Σ) |
|----------------|:----------:|:-------------------------------------:|
| frozen V0.12   |  40³, 80³  |  **0 cells** (ρ=0) at every m=0..0.5   |
| trained W̃     |  40³, 80³  |  **0 cells** (ρ=0)                     |

Finer cells (80³) lift the *value pass* (82.9→90%) but do **not** save T4 — the
closed certified set stays 0, resolution- and value-version-independent.

**So: a decent LOCAL (one-step) certified pass exists (~0.6–0.9), but the CLOSED /
robust certified set is 0.** The next sections measure *why*, by going to the
witness itself.

### 3.1 Witness viability (forward simulation): nominal set is large, robust set is ~1%

`run_witness_viability.py` forward-simulates `u=clip(π_θ(x))` from every lattice-cell
centre that starts in `{V_θ≥0}` (26696 of 64000 cells; Ω* is 0.439 of the domain),
and counts how many stay safe (`g≥0`) for a horizon `H`, under each disturbance
signal. This is the *pointwise* (no cell-worst) invariant set — the empirical
ceiling for any witness-based certificate, and a property of π_θ alone.

**Surviving count (starts in `{V≥0}`, of 26696) vs horizon H:**

| disturbance signal      |  H=1  |  H=5  | H=20  | H=50  | H=100 | H=200 | H=500 |
|-------------------------|------:|------:|------:|------:|------:|------:|------:|
| **d=0 (nominal)**       | 26686 | 25848 | 12111 |  9470 |  9470 |  9470 | **9470** |
| d=−0.3 (const)          | 26686 | 25758 | 11425 |  9758 |  9758 |  9758 |  9758 |
| d ~ U[−0.3,0.3] (rand)  | 26686 | 25845 | 12091 |  9398 |  9398 |  9398 |  9398 |
| **d=+0.3 (const, adversarial)** | 26686 | 25852 | 12746 |  2458 |  1304 |  1134 | **793** |
| **robust = ∩ all signals** | 26686 | 25603 |  7792 |  2395 |  1303 |  1133 | **792** |

**Converged invariant sets (H=500):**

| set                         | cells | % domain | % of {V≥0} | ρ vs Ω* |
|-----------------------------|------:|---------:|-----------:|--------:|
| **nominal (d=0)**           | 9470  |  14.80   |   35.47    | **0.337** |
| adversarial d=−0.3          | 9758  |  15.25   |   36.55    |  0.347  |
| **adversarial d=+0.3**      |  793  |   1.24   |    2.97    | **0.028** |
| **robust (∩ all signals)**  |  792  |   1.24   |    2.97    | **0.028** |

**Observations.**
1. **Nominal / typical disturbances converge to a large, stable invariant set.**
   d=0, d=−0.3 and random all plateau by `H≈50` and hold flat to `H=500`
   (9470 → 9470 → 9470): a genuine non-empty invariant set, **ρ≈0.34** (≈80% of Ω*,
   35% of {V≥0}). So the witness is *not* a bad policy — it really does circle
   safely from a large region. (Your intuition was right: it is far from empty.)
2. **The worst-case constant disturbance d=+0.3 keeps eroding.** 2458 (H=50) → 1304
   (100) → 1134 (200) → 793 (500), still slowly falling; the robust intersection
   tracks it (792). The robust invariant set converges to **≈1.2% of the domain /
   ≈3% of {V≥0} / ρ≈0.028** — small but non-zero. There is a clear left/right
   **asymmetry** (d=−0.3 *helps*, ρ 0.347; d=+0.3 *destroys*, ρ 0.028): the policy
   has a directional bias the adversary exploits.
3. **Chain to the certified set:** nominal pointwise **ρ≈0.34** → robust pointwise
   **ρ≈0.028** → robust *cell-worst* certified (T4) **= 0**. The certificate is
   robust (∀d) and cell-worst, so it lands at the bottom of this chain.



---

## 4. Diagnosis (one root cause)

The binding wall is the **witness π_θ's lack of robustness**, not the verifier,
the value function, or the cell resolution:

- The value object can be made good (route-1 W̃: value pass 59→90%, g-leak-limited).
- The verifier is fine (finer cells help the value pass, leave T4 unchanged).
- The witness has a large **nominal** invariant set but it **collapses under the
  worst-case disturbance** (see §3 viability table); since the certificate is
  robust (∀d∈D, worst case), it certifies ≈ the robust set ≈ 0.

The witness was trained by **regression to the oracle robust labels**
`u_HJ=argmax_u min_d Q_HJ` (bang-bang, MSE-smoothed) **+ a γ-discounted
witness-margin fine-tune against the learned Q_θ** — it never trains directly on
worst-case safety through the true dynamics `f`. So its robustness is inherited
indirectly and degrades at the safety boundary, exactly where the adversary bites.

**Next step (not in this report):** robust/adversarial retrain of the witness —
`max_π min_d W_θ(f(x,π(x),d))` through the true `f` (or a co-trained adversary
`d_ψ`) — to grow the robust invariant set back toward the nominal one.

---

## 5. `γ_teach` sweep — a tighter (less-optimistic) training target (added)

The §3 reference set `Ω*={V_HJ≥0}` uses `γ_teach=0.92`, a fairly heavy discount;
the perfect-filter study (`docs/E0_perfect_filter_collision.md`) showed this makes
`{V≥0}` ~15% LARGER than the true viability kernel (the `(1−γ)g` optimistic shell).
**Hypothesis:** retrain the SAME pipeline against a tighter, genuinely-safe target
(`γ_teach∈{0.97,0.99}`, everything else identical — configs `config_gt097/gt099.json`,
artifacts `results/dubins_e0_gt097/`, `gt099/`) and the witness should inherit a more
robustly-invariant set. (`Ω*` shrinks with `γ`: 0.439 → 0.290 → 0.074.)

**Result A — witness viability (the §3.1 metric), converged H=500, % of domain:**

| `γ_teach` | nominal d=0 | adv d=+0.3 | adv d=−0.3 | **robust ∩** (ρ vs Ω*) | symmetric? |
|----------:|------------:|-----------:|-----------:|----------------------:|:----------:|
| **0.92** (v012) | 14.8 | **1.24** | 15.25 | **1.24%** (ρ 0.028) | ✗ (+0.3 collapses) |
| **0.97** (gt097)| 37.7 | 29.4 | 28.9 | **17.1%** (ρ 0.589) | **✓** |
| **0.99** (gt099)| 15.5 | 18.5 | **0.0** | **0%** (ρ 0) | ✗ (−0.3 collapses) |

**`γ=0.97` is a clear sweet spot:** it FIXES the chiral collapse (now symmetric,
±0.3 ≈ 29%) and grows the **robust invariant set ~14×** (1.24% → 17.1% of domain;
ρ 0.028 → 0.589), and it **converges** (stable from H≈100, vs v012 still eroding).
So the §4 "witness collapses under worst-case d" wall is **substantially relieved by
a less-optimistic teacher target** — the witness inherits the tighter kernel's
robustness. **But `γ=0.99` over-tightens:** the witness re-collapses chirally in the
OPPOSITE direction (−0.3 now dies). The chirality flipping sign across γ confirms it
is a **witness-training instability** (the bang-bang label tie-break amplified by the
smooth fit) that γ only *modulates*; the proper fix is still the robust/symmetric
witness retrain (§4). `γ=0.97` balances it; `0.92`/`0.99` do not.

**Result B — cell-worst recurrence pass (the §2 certified metric), m=0:**

| `γ_teach` | recurrence pass | inner ρ({W≥0}) |
|----------:|----------------:|---------------:|
| 0.92 (v012) | **59.2%** | 0.74 |
| 0.97 (gt097)| 46.9% | 0.68 |
| 0.99 (gt099)| **9.3%** | 0.46 |

**Opposing trend:** the certified recurrence pass falls **monotonically** as γ
tightens. A tighter γ makes `V_HJ` (and the distilled `V_θ`) **sharper** near the
now-genuine safety boundary → higher Lipschitz → larger cell-worst CROWN slack →
fewer cells clear the sound bound. So the model gets **more robust pointwise** while
becoming **harder to verify cell-worst** — the binding constraint shifts from the
witness to verifier slack, which is the *finer-cells / verifier-tightening* lever
(orthogonal to γ), not a model problem.

**Takeaways.** (1) Training against a tighter, non-optimistic target (`γ_teach≈0.97`)
is a cheap, real improvement: **robust viable set 14× larger and symmetric**, the
single best lever found so far for the §4 wall. (2) It trades against cell-worst
certifiability (recurrence 59→47%, recoverable with finer cells); `γ=0.99` over-does
both (recurrence 9%, witness re-collapses). (3) The chirality is a witness-training
instability, not solved by γ — the robust/symmetric witness retrain remains the
principled fix, now with `γ_teach=0.97` as the recommended target.
