# Dubins E0 — Results & Ablation Study

This document records the empirical results of the certified-robust Q-CBF
pipeline on the Dubins-car experiment (E0): what the verifier proves, where
Gate D fails, **why** it fails (a precise causal chain, established by cheap
reuse-the-candidates ablations), and the fix. It is written to be lifted into
the paper's experiments section.

> **One-line summary.** The verifier is sound and the learned filter has a large
> healthy candidate set, but **Gate D fails because the sound antecedent-skip
> test fires almost never** (3 / 726 960 pairs at 64³). With no skips the C2
> robust-invariance closure must cover *every* control action and collapses. The
> root cause is a **flat value function** (regression to the smooth discounted
> HJ teacher), which makes the runtime gate `Q ≥ γV` unselective. The fix is to
> **sharpen the value** so the gate — and hence the skip — discriminates
> set-leaving actions at a moderate γ.

---

## 1. Setup

Fixed-speed Dubins car, `dt=0.1`, `v=1`, `ω_max=1`, disturbance `d∈[−0.3,0.3]`,
single obstacle `r_obs=0.45` inside world radius `1.8`, position domain
`[−2,2]²`. Three frozen ReLU-MLP networks `V_θ, Q_θ, π♭_φ` (hardtanh policy
head). Certificate = Theorem A conditions **C1 ∧ C2 ∧ C3** discharged by
IBP/CROWN over a cell lattice, pure NumPy.

**Soundness floor (T1–T8): all pass.** Every CROWN/IBP bound encloses the true
network (min slack > 0 over random trials), the compiled `h3` reproduces
`Q(x,clip(π),d)−γV−ε` to ~1e-15, the interval `cos/sin/g` bounds enclose their
functions, and successor boxes contain sampled successors (wrap-split active).
The soundness of the verifier is **not** in question anywhere below.

---

## 2. Headline Gate-D runs

| profile | oracle | lattice | C1 pass | lbV≥0 | **C3 (candidates)** | sound skips | **C2 closure** | Gate D |
|---------|--------|---------|--------:|------:|--------------------:|------------:|---------------:|:------:|
| pilot   | 51³    | 40³     | 33 920  | 22 485| **16 813**          | 0           | **→ 0**        | FAIL |
| paper   | 81³    | 64³     | 146 944 | 99 124| **90 870**          | 3 / 726 960 | **→ 0**        | FAIL |

The candidate funnel is *healthy and grows with resolution* (16 813 → 90 870).
The failure is entirely at the **C2 fixed point**, which prunes the candidate
set to empty in ~6 iterations for every initialiser `c`. Oracle `Ω*` is ~42% of
the grid in both profiles, so the denominator is well-defined; ρ = 0 only
because the certified set is empty.

Wall time (paper 64³, 1 CPU, with the new live progress bars): oracle 7 min,
train 22 min, **certify 135 min** (C3 sub-split stage dominates at 76 min, skip
48 min), audit/figures < 1 min.

---

## 3. Ablation: *why* C2 collapses

All ablations below **reuse the saved 64³ candidate set** (the expensive C3
output) and recompute only the seconds-fast successor-ranges + C2, so each row
is cheap. They isolate the cause cleanly.

### 3.1 It is not the lattice, not the disturbance, not the sub-splits

| knob varied | values tried | C2 accepted | C2 history |
|-------------|--------------|------------:|-----------|
| `n_u_cells` | 8, 16, 32    | 0 (all)     | identical: 90870→63796→37923→13749→1208→34→0 |
| `c2_u_subsplit`, `c2_d_subsplit` | 2, 4, 8 | 0 (all) | identical |
| **`d_max`** (successor side) | 0.30, 0.20, 0.10, **0.00** | 0 (all) | identical |

The collapse is **bit-identical** across all of these — even with **zero
disturbance**. So the looseness is the **state-cell extent** propagated through
the drift, not the discretisation of `u`/`d` and not the disturbance magnitude.

### 3.2 Decomposition of the iteration-1 failure (64³, d=0)

```
candidates                 : 90 870
survive iter-1 (all u cov) : 63 796 (70.2%)
fail: successor LEAVES domain:      0 (0.0%)
fail: coverage (in-domain) : 27 074 (29.8%)
successor box extent (cells): x~2.1  y~2.1  psi~2.5   (~11 cells / pair)
```

Each cell's interval successor **box spans ~11 lattice cells**, all of which
must be accepted. Fixed-speed Dubins drifts ~0.1/step (≈ 1–2 cells), so a cell's
successor box never overlaps the cell itself and always straddles the set
boundary. 30% of candidates fail coverage at the boundary, and pruning them
un-covers their neighbours → cascade to empty. This extent is essentially
**resolution-independent** (position ≈ `1 + dt·v·(2π)/4 ≈ 1.16` cells before
outward rounding, heading grows slowly with `n`), which is why §3.1 finer
lattices/sub-splits do not help.

### 3.3 The abstraction needs the skip to do the work

Feeding the interval C2 a **genuinely one-step-invariant set** (the nominal
`d=0` point-invariant set, 85 825 cells, built from the deployed gate) still
collapses:

```
interval C2 on the point-invariant set (d=0): 80 184 → 56 547 → 33 675 → 12 319 → 109 → 0
```

So box-reachability C2 *cannot* close on this drift-dominated flow **when it must
cover every gate-feasible action**. The design intends the **skip** test to
remove the set-leaving actions; the whole question reduces to whether skip fires.

### 3.4 Skip fires ~never — and `γ` is the lever

Skip removes `(cell,u-cell)` pairs whose antecedent `a = min_d Q − γV` is
certified `< 0`. Two measurements on the 64³ candidates:

*Point-skip proxy* (optimistic, antecedent at cell/u-cell centres) → interval C2:

| γ | 0.50 | 0.70 | 0.80 | 0.90 | 0.99 |
|---|-----:|-----:|-----:|-----:|-----:|
| point-skips | 16 977 | 56 425 | 107 677 | 200 412 | 335 356 |
| C2 accepted | 0 | 294 | 2 265 | **9 207** | 25 607 |

*Sound CROWN skip* (the real one, `n_u_cells=16`, `ante_d_probes=3`):

| γ | 0.50 | 0.70 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|-----:|-----:|-----:|-----:|-----:|-----:|
| **sound skips** | 11 | 114 | 581 | 1 350 | 2 911 | 6 447 |
| C2 accepted | 0 | 0 | 0 | 0 | 0 | **8** |

Two facts: (i) **γ is the right direction** — more selectivity → more skips → a
non-empty C2; (ii) the **sound** CROWN upper bound is ~**70× more conservative**
than the point proxy (3–6 k vs 200 k skips), so γ *alone* only squeaks to 8
cells, and only at γ=0.95 where C3 (which shares γ) becomes hostile and shrinks
the candidate set. **γ cannot serve both C3 (wants low γ) and skip (wants high
γ) with a flat value.**

### 3.5 Root cause and fix

The discounted HJ teacher `V*` is **flat** (≈1.05 across the safe region; it
*correctly* labels the invariant orbit `V*>0`, verified, so it is not the
anti-spec trap). Because `Q = V*∘f` is flat in `u`, the gate `Q ≥ γV` admits
~all actions and the sound skip ub stays above `γ·lbV`. **The fix is to sharpen
the value** with a monotone `φ(v)=tanh(k·v)` (`φ(0)=0`, so `{V≥0}` is unchanged):
the gate steepens near the boundary, the skip becomes selective at a *moderate*
γ, and C3 stays easy. Implemented as `TrainConfig.value_sharpen` (opt-in).

> A robustly-invariant set **exists** (a hand-built orbit holds `g≥0` for 2000
> disturbed steps at R=1.0/1.2/1.4; robust circular orbits need
> `R ≥ 1/(ω_max−d_max)=1.43`). The gap is purely what the verifier can *prove*
> with this value function, not what is physically true.

---

## 4. The fixes that *don't* work — and why (the core finding)

We tried every in-method lever to make the **sound** skip fire (the sole
mechanism that lets C2 close on a drift-dominated plant). None do.

| attempt | config | sound skips | C2 / Gate D |
|---------|--------|------------:|-------------|
| baseline | paper-64³, γ=0.5, n_uc=8 | 3 / 726 960 | 0 — FAIL |
| de-risk (γ sweep, sound skip) | 64³, n_uc=16, probes=3 | 11→6 447 (γ 0.5→0.95) | 0, except **8 cells** at γ=0.95 (with the *optimistic* γ=0.5 candidate set) |
| **value-sharpening** (Option 4) | Dubins 64³, `tanh(3·)`, γ=0.7, n_uc=16 | **0 / 1 401 920** | 87 620 → 0 — FAIL |
| **value-sharpening** (Option 4) | F1TENTH 56³, `tanh(3·)`, γ=0.7, n_uc=16 | **0 / 989 664** | 61 854 → 0 — FAIL |
| **+ skip sub-split** (2³) | sharpened F1TENTH candidates (subset) | **0** | — |

Both sharpened runs give **exactly skip = 0**: the value-sharpening, which makes
the *true* gate selective, fires the *sound* skip not at all — direct evidence
of the §4.1 tension (the steeper net's looser CROWN cancels the gain).

### 4.1 The irreducible tension (value selectivity vs. CROWN tightness)

The skip needs, per `(cell, u-cell)`,
`ub_CROWN[min_d Q] < γ · lb_CROWN[V]`. To make the *true* antecedent negative
for set-leaving actions you need a **steep** value (so bad-`u` `Q` is clearly
low). But a steeper network has **larger weights**, which **loosens** the CROWN
bounds — both inflating `ub[Q]` and, worse, *deflating* `lb[V]` so the threshold
`γ·lb[V]` collapses toward zero. Sharpening therefore *cannot* help the skip:
the very property that makes the gate selective (steepness) is the property that
makes the verifier loose. Sub-splitting tightens `ub[Q]` but not the `lb[V]`
threshold, so it does not fire skips either. **This is a structural limitation
of an IBP/CROWN antecedent-skip on a learned value, independent of resolution,
γ, sharpening, or sub-split — for plants whose forward drift forces C2 to reason
about every admissible action.**

### 4.2 The CBF fix — implemented, and the deeper obstruction it exposes

We implemented the recommended fix and a clean skip-free certificate, then
validated cheaply (Dubins 81³ teacher, bicycle 51³ teacher; 48–64³ lattices):

- **`qcbf.nets.mlp.finetune_cbf`** co-trains `(V,Q,π)`: a decrease hinge
  `relu(γV − min_d Q(x,π(x),d) + margin)`, a `Q ↔ V∘f` consistency term (so C3
  reflects the *true* decrease and Q inherits V's selectivity), a light `V*`
  anchor, and a **Lipschitz/weight penalty** on V, Q (and π). It works as a CBF
  trainer: decrease hinge → ~1e-3, **C3 holes fall from ~12% to ~4%**, and the
  successor-box undershoot shrinks (see below).
- **Two skip-free certificates** (`qcbf.certify.refine`): `c2_fallback_fixed_point`
  (cover only the fallback successor) and, cleaner, **`certify_sublevel_invariant`**
  — a *one-pass* check that `{V≥c}` is invariant: for every cell with `ubV≥c`,
  `gmin≥0` and the CROWN lower bound of V over the fallback successor box `≥ c`.

**Result: still empty, for a deeper reason.** Even with a low-Lipschitz CBF and
γ→1 (a near-non-decreasing barrier, decrease-hinge ≈ 3e-3), **no level `c` is
certifiable** (bicycle 64³, all c in [0, 0.8] fail). The blocker is now precise
and *structural*:

| lattice | successor-box undershoot `med(lbV − lbVs)` |
|---|---|
| 48³ | 0.029 |
| 64³ | **0.005** |

The undershoot shrinks with resolution but is never 0, and **every cell-
discretised level set `{V≥c}` has a boundary of *straddling* cells** (`ubV≥c`
but `lbV<c`) whose interval successor box dips below `c`. To absorb that dip the
plant would need a *contraction margin* `V(x⁺) ≥ V(x) + undershoot` near the
boundary — i.e. the ability to **raise** V. A **fixed-speed** car cannot
(it can't slow or stop; its maximal invariant sets are **constant-V orbits**),
so the margin is structurally zero and the box undershoot wins at any feasible
resolution. This is the **cell+box-vs-fixed-speed obstruction**, and it sits
*below* the value function: no CBF, γ, resolution, skip, or sub-split removes it.

### 4.3 What this means, and the genuine path to Gate D — **resolved (Gate D PASS)**

- The **certificate is sound** throughout (T1–T8); the limitation is
  *completeness*, and it is now characterised down to its geometric root.
- **The fix is the plant, not the verifier or the value** — and it is now
  **implemented and certified** (`experiments/f1tenth_e2`, `qcbf/dynamics/
  bicycle_accel.py`). A **variable-speed / brakeable** 4-state bicycle (px,py,ψ,v,
  v_min=0) with an **analytic braking-distance CBF** `V = clearance − D(v) −
  margin` (D = exact discrete braking distance) is robustly forward-invariant by
  a one-line structural argument: under braking the clearance lost (≤dt·v) is
  exactly cancelled by the stopping distance recovered (D(v)−D(v−dt·b)=dt·v), so
  `V(x⁺)−V(x) ≥ 0` — the contraction margin the fixed-speed orbit lacked.
  **Result: Gate D PASS** — certified set 86 580 cells (22.5% of the (p,v)
  domain, up to 2.3 m/s), braking-invariant (grid min decrease −0.0000), and
  **zero certified-but-violated** states under extremal + greedy adversaries
  (min g ≥ +0.12), while the car actually races. The certificate needs no skip,
  no cascade, no learned value — just the analytic CBF and the (tested) interval
  primitives.
- A **non-interval one-step image** (contraction-metric / non-box tube) would
  also remove the fixed-speed undershoot, but the brakeable plant is the simpler,
  more realistic fix and the one demonstrated here.

### 4.4 Certifying the **learned** object (E2-L) — sound FAIL, sharply attributed

E2's Gate-D pass uses the **analytic** CBF. The actual project claim is to certify
the **deployed learned** trio `V_θ, Q_θ, π♭_φ` *directly* — Theorem A's (C1)–(C3)
by CROWN on the networks, **no `V_θ≈V_target`** in the proof (driver
`experiments/f1tenth_e2/run_cert_learned.py`; distillation `distill.py`,
`MSE(V_θ,V_target)≈0.005`). Both sound routes fail, and the attribution is exact:

- **Route A — heading-free braking sub-level `{V_θ≥c}`** (the learned analogue of
  E2): **never closes.** The brake-successor undershoot `ubV_θ − lbV_θ(x⁺)` equals
  the **CROWN gap on `V_θ` ≈ the true per-cell variation of `V`** (≈0.15 at a
  0.14 m cell; *not* fit error). It is strictly `>0`, exceeding the **exactly-zero**
  worst-case braking contraction (clearance is a distance function, so the
  worst-heading loss is *exactly* `dt·v`). The analytic CBF closes A via that
  **exact 1-Lipschitz cancellation** (`verify_braking_decrease`), which a
  CROWN-box-bounded black-box `V_θ` cannot reproduce. The margin `m` moves only
  **C3 / liveness**, never the C2 boundary (m cannot add C2 slack without assuming
  `Q_θ≈V_θ∘f`, forbidden). Undershoot shrinks with finer cells but never reaches 0.
- **Route B — heading-inclusive 4-D cell-reachability** (Tarski GFP under the brake
  map, the only route to a heading *tube*): **erodes to empty even for the analytic
  IDEAL `V`** (candidate → 0 in ~7 iterations). The outward-rounded successor box
  always overlaps neighbour cells, so closure forces a boundary-free set — the same
  **cell+box obstruction** as §4.2–4.3, independent of the learned net.
- **Finding (scoped — corrected 2026-06-11):** Route B erodes the **analytic ideal
  `V` to empty too**, so its failure characterises the **certification primitive**
  (outward-rounded cell+box one-step reachability), **not** the learned object — a
  method that cannot close on a known-invariant ideal set cannot prove the learned
  object uncertifiable. The correct claim: *outward-rounded cell+box one-step
  reachability cannot non-vacuously certify any bounded robust-invariant set here.*
  Route A's knife-edge failure is a genuine learned-vs-analytic structural fact but
  is specific to the **brake-only** sub-level. **Whether a tighter sound primitive
  (direct-composition CROWN (P1) / zonotope / contraction) certifies the learned
  trio is the open experimental question** — each tried ideal-first (the ideal must
  pass, ~86,580 cells), so any residual learned failure is cleanly attributable.
  See `experiments/f1tenth_e2/README_LEARNED.md`, `run_cert_p1.py`, and the
  `e2_learned_report.json` + figures.

---

## 5. Takeaways for the method (paper-relevant)

A clean chain of *successively deeper* obstructions, each ruled out empirically:

1. **Not resolution / disturbance / sub-split.** The interval-C2 successor box is
   ~constant in cell units (≈2×2×2.5), so the all-actions C2 collapses identically
   across these knobs (even `d_max=0`).
2. **Not the antecedent skip.** It fires ~never; γ and value-sharpening can't make
   it (the selectivity↔CROWN-tightness tension, §4.1).
3. **Not the value function.** A properly CBF-trained low-Lipschitz `V` (decrease
   verified, holes ~4%) still yields an empty certificate under *every* C2 form
   (all-actions, fallback-only, and the direct sub-level check).
4. **The root is the plant geometry.** Any cell-discretised invariant set has a
   *straddling* boundary whose interval successor box undershoots; absorbing it
   needs a contraction margin (`V(x⁺) ≥ V(x) + undershoot`), which a **fixed-speed**
   car structurally lacks (constant-V orbits). → certify a **variable-speed /
   brakeable** plant, where `certify_sublevel_invariant` closes on the interior.

The headline is a *characterisation*: the certifier is sound; its completeness on
fixed-speed vehicles is bounded by the cell+box one-step over-approximation, and
the fix is a brakeable plant (or a non-box one-step image), not more tuning.

## 6. Next steps

### 6.1 F1TENTH port — **built and verified** (`qcbf/dynamics/bicycle.py`)

The fixed-speed kinematic bicycle shares the Dubins position update, so the
oracle stencil and the *entire* verifier reuse directly; only the heading rate
`(v/L)·tan(δ+d)` and a monotone `tan` interval are new. Delivered this session:

- `qcbf/dynamics/bicycle.py` — `BicycleModel`, `BicycleConfig` (F1TENTH-scale:
  `L=0.33 m`, `v=2 m/s`, `δ_max=0.4`, steering disturbance `d_max=0.1`),
  `tan_interval`, `successor_boxes`. **Soundness suite `tests/test_bicycle.py`
  passes** (tan enclosure, g-box, successor containment with wrap-split).
- **Plant-agnostic certifier**: introduced `control_max` and `model.heading_rate`
  hooks and an injectable `successor_boxes_fn` (default Dubins). The grid oracle
  is now `GridOracle(dyn, cfg, model=…)` (alias `DubinsOracle` kept). Dubins
  T1–T8 + a Dubins-oracle check confirm **zero regression**.
- **End-to-end driver** `experiments/f1tenth_e1/run_cert.py` runs
  oracle→train(+sharpen)→certify→c-sweep→Gate-D→audit on the bicycle. Smoke
  (20³) completes in 35 s; the `--scale pilot` (56³, sharpened) is the genuine
  Gate-D attempt. Realistic authority `(v/L)·tan(δ_max) = 2.56 rad/s` vs a
  `~0.6 rad/s` disturbance gives a *fatter* robust set than Dubins, so the
  certificate should close more readily once sharpened.
- Hardening: fixed an empty-candidate edge case in `crown_bounds_chunked`
  (returns well-shaped empties) surfaced by the tiny F1TENTH smoke.

### 6.2 ROS real-car package (planned)

Wrap the runtime `CertifiedFilter` as a ROS 2 node: subscribe to the state
estimate, run `batch_select` at the control rate, publish the filtered command,
and expose `is_certified(x)` as a safety-status topic. The certificate
(`accepted` mask + frozen `v/q/pi.npz` + lattice) is computed offline and
shipped read-only; the node performs only the sound CROWN feasibility check
(microseconds for the small `q_net`). No solver, no autodiff at runtime.
