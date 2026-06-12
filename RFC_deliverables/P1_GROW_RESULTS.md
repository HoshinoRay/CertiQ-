# P1 — Grow-from-seed certified invariant-set expansion (go/no-go)

**Experiment P1 of `grow_from_seed_certified_expansion.md` §9.** Tests the central
LFP claim of the next-step framework: the verified grow operator

```
G_V(R) = R ∪ ( 𝒦 ∩ 𝒱₀ ∩ Pre_ver(R) ),   Pre_ver(R) = {c : ∃u∈U_menu, Enc(c,u) ⊆ ⋃R}
```

seeded at `R₀ = S_brake` (the current RFC brake-funnel certificate, ρ≈0.80)
grows the certified robust-invariant set **outward** by adding *heading-specific*
cells that can **drive** into the already-certified set in one sound step — with
**every iterate already a sound robust invariant set** (anytime soundness,
Theorem A; no gfp erosion).

> This is **not** the dead sub-level route in `run_cert_p1.py`
> (`{V_θ≥c}` one-step invariance, killed by Gate-0b level-independent holes).
> Grow-from-seed certifies a strictly **larger invariant set**, not a level set
> of `V_θ`. The learned object `(V_θ,Q_θ,π♭)` is unchanged and only enters via
> the B1 anchor `𝒱₀={lb V_θ≥0}` (pure set subtraction, never an invariance
> assumption). Soundness rests only on {A3 envelope, A4 membership, A2 seed}.

Driver: [`experiments/f1tenth_e2/run_cert_p1_grow.py`](../experiments/f1tenth_e2/run_cert_p1_grow.py)
· engine [`qcbf/certify/grow.py`](../qcbf/certify/grow.py)
· new sound primitive `heading_successor_interval` in
[`qcbf/dynamics/bicycle_accel.py`](../qcbf/dynamics/bicycle_accel.py).

---

## Headline (preliminary)

**The lfp is sound and non-vacuous — it grows the certified set beyond the brake
seed, and the gain increases with resolution.** Currently **WEAK** (Δρ ≈
+0.02…+0.04, below the +0.05 "GO" bar) but **GO-leaning**: the analytic ideal is
cleanly resolution-monotone and already crosses +0.05; the learned object holds a
+0.021 plateau at res 36–44 then jumps to **+0.039 at res 56** (this jump should
be confirmed at res 80 before calling the learned trend monotone). Verdict per
the §9 decision rule: this is the **WEAK band → diagnose + scale the levers, do
not abandon** — and every diagnostic (resolution trend, heading-resolution
ablation, ideal crossing +0.05) points toward GO.

| res (npx×npy×**npsi**×nv) | ρ_brake (seed) | ρ_∞ (grown) | **Δρ learned** | Δρ ideal* | grown cells / waves | cbv |
|---|---|---|---|---|---|---|
| 36×36×**16**×27 | 0.7970 | 0.8176 | **+0.0206** | +0.0377 | 2 296 / 7 | **0** |
| 44×44×**16**×33 | 0.8103 | 0.8312 | **+0.0209** | +0.0427 | 4 430 / 8 | **0** |
| 56×56×**16**×42 | 0.8166 | 0.8559 | **+0.0393** | +0.0515 | 17 880 / 12 | **0** |
| 44×44×**32**×33 | 0.8224 | 0.8498 | **+0.0274** | +0.0558 | 11 626 / 8 | **0** |

Best certified coverage so far: **ρ_∞ = 0.856** (res 56). Heading refinement also
raises the *seed* itself (ρ_brake 0.810→0.822 at res 44 as npsi 16→32: finer
heading cells let more cells pass the brake funnel), so npsi helps twice.

\* *Ideal iron-rule*: grow a deliberately **shrunk** analytic seed `{V_analytic≥0.30}`
back toward `{V_analytic≥0}` under the *same* engine. The primitive **must**
grow on a known-invariant ideal before any learned result is trusted — it does,
and more so at finer resolution (empirical Theorem C / Prop 3).

ρ is measured in the full 4-D state volume: `ρ = |R_4d| / (|Ω*_3d|·npsi)`, so the
seed (heading-free `S_brake` lifted to all headings) reproduces the RFC
`ρ_brake≈0.80` exactly, and any Δρ>0 is genuine heading-dependent expansion.

---

## Soundness (the non-negotiable part)

At **every** resolution, three independent checks pass with **zero** violations:

1. **Enc envelope (A3)** — `20 actions × 200 000` Monte-Carlo samples per action;
   the true successor (position, speed, un-wrapped heading) lies in the Enc box
   in **0** cases.
2. **Grown-cell one-step containment** — `20 000` states drawn from grown cells,
   witness action + worst-of-corners adversary `d`: successor leaves `R` in **0**
   cases, `g<0` in **0** cases. (This is the direct grow claim; any escape would
   be an Enc/index bug.)
3. **End-to-end layered-policy audit** — extremal + greedy-`d` rollouts of the
   layered policy σ (brake in the seed, witness `u*_c` in grown cells),
   horizon 300: **certified-but-violated = 0**, min `g` = +0.12…+0.18.

So the grown set is a **machine-checked sound robust invariant set** — the
anytime-soundness guarantee is met empirically, not just claimed.

---

## What limits the gain (failure decomposition)

For candidates never added at convergence (res 44):

```
no_valid_action  : 18 690   (57%)  — every menu action's one-step successor
                                      leaves 𝒦∩𝒱₀ under worst d  → one-step
                                      conservatism / capture-basin boundary
in_region_blocked: 13 920   (43%)  — a valid action exists but its successor
                                      touches a non-R cell that itself never
                                      enters R  → capture-basin frontier
```

The binding obstruction is **one-step conservatism**, and it is **resolution-
sensitive** in exactly the way the framework predicts:

- **(px,py,v) refinement** (res 36→44→56, npsi fixed): Δρ +0.021→+0.021→**+0.039**.
- **Heading refinement** (res 44, npsi 16→32): ideal Δρ +0.043→**+0.056**,
  learned Δρ +0.021→**+0.027** (and ρ_brake 0.810→0.822). The position successor
  box is inflated by `cos/sin` over a heading **cell** (22.5° at npsi 16), so
  halving the heading cell directly tightens Enc and unlocks more growth on both
  the ideal and the learned object. → **heading-cell width is a primary
  conservatism knob.**

Both findings are consistent with Prop 3 (R_∞ monotone non-decreasing under grid
refinement) and the Theorem-C relative-completeness picture
`Ω*∖R_∞ ≈ (Viab∖Capt) ∪ discretisation(h) ∪ envelope-overshoot(e(h))`.

## Q_θ proposal value (offline accelerator, never in the soundness chain)

Ranking the 20 menu actions by `min_d Q_θ(center,u,d)` and checking where the
verified winning action lands (res 44, over 4 430 grown cells):

```
top-1 = 0.19    top-3 = 0.43    top-5 = 1.00    mean rank = 2.47 / 20
```

`Q_θ` puts the winning action in its **top-5 for 100% of grown cells** — so a
Q-ranked frontier sweep (try top-m, fall back to exhaustive) would discharge the
grow at a fraction of the menu cost while changing the certified set by nothing
(over-permissive proposals are simply rejected by the verifier; A5/§5).

---

## Preliminary conclusion & recommendation

- **Go/no-go answer:** ρ **does** grow beyond the 0.80 brake seed — soundly,
  resolution-monotonically — but the current gain (**Δρ ≈ +0.02…+0.04**) sits in
  the **WEAK** band of the §9 rule (`0 < Δρ < +0.05`). The §9 prescription for
  this band is *run diagnostics + richer seed before judging, don't abandon* —
  and the diagnostics (resolution trend, heading-resolution ablation, ideal
  crossing +0.05) all point **GO**.
- **The novelty that already holds regardless of the final ρ:** an *anytime-
  sound, lfp, seed-anchored-in-a-learned-object, Q-guided* certified-expansion
  pipeline whose every iterate is a verified robust invariant set (`cbv≡0` by
  construction, confirmed by audit) — distinct from the dead sub-level route and
  from the rejected naive racing closure (Theorem S).
- **P2 levers (highest-expected-value first):**
  1. **Resolution / heading refinement** — already +0.04 at res 56; push res 80
     and npsi 24–32 (the ideal hits +0.056 at npsi 32). Cheap, monotone.
  2. **2–3-step verified macro-actions** (motion primitives) — directly attacks
     the dominant `no_valid_action` mass in the knife-edge band (§9 risk 2).
  3. **Multi-funnel seed** `R₀ = S_brake ∪ {left,right full-steer funnels}` —
     enlarges the capture target the lfp grows from (§9 risk 1 / Prop 3 monotone).
  4. **Center-form / heading-split Enc** — tightens `e(h)` without finer grids.

---

## Reproduce

```bash
python experiments/f1tenth_e2/run_cert_p1_grow.py --quick            # smoke (~15 s)
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44           # headline
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 56           # resolution trend
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44 --npsi 32 # heading-conservatism ablation
```

Report: `results/f1tenth_e2/p1_grow_report.json` (+ per-resolution `p1_grow_res*.log`).
Every run prints the Enc MC self-check, the ideal iron-rule, the ρ(k) wave
history, the layer histogram, the failure decomposition, the Q-proposal value,
and the adversarial audit (cbv).
