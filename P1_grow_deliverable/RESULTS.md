# P1 — Grow-from-seed certified invariant-set expansion (go/no-go + calibration)

**Experiment P1 of `grow_from_seed_certified_expansion.md` §9**, refined with the
diagnostics, capture-basin calibration, and deployed three-way rollout requested
in review. Tests the central LFP claim: the verified grow operator

```
G_V(R) = R ∪ ( 𝒦 ∩ 𝒱₀ ∩ Pre_ver(R) ),   Pre_ver(R) = {c : ∃u∈U_menu, Enc(c,u) ⊆ ⋃R}
```

seeded at `R₀ = S_brake` (RFC brake-funnel certificate, ρ≈0.80) grows the
certified robust-invariant set outward by adding *heading-specific* cells that can
**drive** into the certified set in one sound step — every iterate already a sound
robust invariant set (anytime soundness, Theorem A; no gfp erosion). The learned
trio `(V_θ,Q_θ,π♭)` is unchanged and enters only through `𝒱₀={lb V_θ≥0}`.

Driver [`run_cert_p1_grow.py`](../experiments/f1tenth_e2/run_cert_p1_grow.py)
(`--full` adds calibration + deployment) · engine
[`qcbf/certify/grow.py`](../qcbf/certify/grow.py) · capture reference
[`qcbf/certify/viab_reference.py`](../qcbf/certify/viab_reference.py) · analysis
[`analyze_p1_grow.py`](../experiments/f1tenth_e2/analyze_p1_grow.py).

---

## Headline (preliminary) — the load-bearing result is soundness, not ρ

**The lfp-from-below construction is anytime-sound, confirmed empirically at every
configuration: `cbv ≡ 0`.** This discharges Theorem A from a paper proof to an
empirical fact — each iterate `R_k` is a *real* robust invariant set, certified
under three independent checks (Enc MC, grown-cell one-step containment, layered
end-to-end audit), all **zero** violations at res 36/44/56/64/80. That is the
承重 (load-bearing) contribution and it is in hand.

The ρ gain is the secondary characteristic. `ρ_∞` is **monotone increasing** with
resolution and the grow contribution **Δρ reaches GO (+0.0563 ≥ +0.05) at res 80**
(`ρ_∞=0.880`); the analytic ideal crosses +0.05 by res 56. The **capture-basin
calibration** then reframes the *remaining* residual: the sound set already
recovers **86 %** of the reference reachable basin within the learned anchor
(conservatism ≈ 14 %, recoverable), the capture gap of Ω\* is **≈ 0** (Ω\* is
almost entirely reachable — the residual is *not* intrinsic), and the true capture
basin is **≈ 1.9× Ω\*** (the brake-anchored denominator understates the safe set;
the grow already certifies 1 288 cells *beyond* Ω\*). So the certified set is
near-complete inside a conservative anchor, and refinement keeps growing it.

---

## 1. Resolution sweep — `ρ_∞` monotone↑; Δρ crosses +0.05 (GO) at res 80

ρ in the full 4-D volume, `ρ = |R_4d| / (|Ω*_3d|·npsi)`; the seed (heading-free
`S_brake` lifted) reproduces `ρ_brake≈0.80`, so Δρ>0 is genuine heading-dependent
expansion. `h = position cell width = 6/npx`.

| res (npx²×**npsi**×nv) | h | ρ_brake | ρ_∞ | **Δρ learned** | Δρ ideal | cbv |
|---|---|---|---|---|---|---|
| 36²×**16**×27 | 0.167 | 0.7970 | 0.8176 | **+0.0206** | +0.0377 | 0 |
| 44²×**16**×33 | 0.136 | 0.8103 | 0.8312 | **+0.0209** | +0.0427 | 0 |
| 56²×**16**×42 | 0.107 | 0.8166 | 0.8559 | **+0.0393** | +0.0516 | 0 |
| 64²×**16**×48 | 0.094 | 0.8190 | 0.8563 | **+0.0373** | +0.0524 | 0 |
| **80²×16×60** | 0.075 | 0.8233 | **0.8796** | **+0.0563** | +0.0598 | 0 |
| 44²×**32**×33 | 0.136 | 0.8224 | 0.8498 | **+0.0274** | +0.0558 | 0 |

**`ρ_∞` is monotone increasing in resolution** — 0.818 → 0.831 → 0.856 → 0.856 →
**0.880** (res 36→80) — and the grow contribution **Δρ crosses the +0.05 GO bar at
res 80 (+0.0563)**; the analytic ideal crosses it by res 56. (Δρ itself is
grid-noisy — the res 64 dip to +0.037 is `ρ_brake` growing faster that step, not
`ρ_∞` falling; `ρ_∞` never decreases, as Prop 3 requires.) So the framework's
**resolution-monotone claim is confirmed for the learned object and GO is reached
by refinement** (res 80, npsi 16); the calibration (§3) shows how much further is
recoverable beyond that.

## 2. Two serial bottlenecks, each with its own knob (proven decomposition)

The 4-way failure attribution of the candidates never added (mutually exclusive,
by most-recoverable reason) localises the obstruction:

| res | frontier_blocked | **v0_blocked** | k_blocked | domain |
|---|---|---|---|---|
| 44 (learned) | 13 920 | **17 976** | 714 | 0 |
| 64 (learned) | 55 615 | **37 083** | 442 | 0 |

- **`v0_blocked`** (no action keeps Enc ⊆ K∩𝒱₀, but one keeps Enc ⊆ K) = the
  **learned anchor `{lb V_θ≥0}`** is binding: the successor stays collision-free
  but leaves the learned safe sub-level. Its learned-minus-ideal excess is the
  `lb V_θ` CROWN looseness, which **tightens with spatial res**.
- **`frontier_blocked`** (a valid action exists but the successor touches a
  not-yet-`R` cell) + `k_blocked` = **one-step Enc conservatism**, dominated by
  the heading-cell width (the position box inflates by `~v·Δψ`), which **tightens
  with heading res (npsi)**.

The res×npsi ablation confirms both are live and *serial* (each needs its own
knob): res 44, npsi 16→32 lifts the ideal Δρ +0.043→**+0.056** and the learned
Δρ +0.021→**+0.027** (and ρ_brake 0.810→0.822). At high res the breakdown is
**`frontier`-dominated** (res 80: 102 565 frontier vs 55 023 `v0`): once `lb V_θ`
tightens spatially the binding obstruction is the one-step Enc envelope (heading),
which is why macro-actions / center-form Enc (§5) are the cheapest *further* lever
even though resolution alone already reaches GO at res 80.

## 3. Capture-basin calibration — the residual is recoverable, and Ω\* understates Viab

Reference robust capture basin of `S_brake` by backward reachability on the
**sampled true dynamics** (cell centres, adversarial over a 3×3 `d`-grid). It is
a NOMINAL reference (optimistic — an upper target on what a sound primitive could
certify at this resolution), giving `R_∞ <~ Capt_ref <~ true capture basin`.
At **res 44**:

| quantity | value | reading |
|---|---|---|
| `ρ(R_∞)` (sound) | 0.831 | certified |
| `ρ(Capt within K∩𝒱₀)` | 0.971 | reachable within the learned anchor |
| **recovery within anchor** `\|R_∞\|/\|Capt\|` | **0.856** | only **14 %** method conservatism |
| **capture gap of Ω\*** | **0.007** | Ω\* is **fully** capture-reachable — residual not intrinsic |
| `ρ(Capt within K)` | **1.876** | **Viab ≈ 1.9× Ω\*** — brake-anchor understates the safe set |
| capture **beyond** Ω\* | +0.883 | safe-maneuverable states excluded by the brake-stop definition |
| `R_∞ ∖ Capt` (sanity) | **0** | sound ⊆ reference ✓ |
| `R_∞ ∖ Ω\*` | 1 288 | grow soundly certifies cells **beyond** brake-safe Ω\* (Viab>Ω\*, sound) |

**This is the decisive reframe.** The residual `Ω*∖R_∞` is ≈ all *recoverable
conservatism* (capture gap ≈ 0), and the anchor Ω\* itself captures only ≈ 53 %
of the reference Viab. The method is *near-complete within a conservative anchor*,
and it already steps past that anchor (1 288 sound cells with `V_analytic<0` that
drive to safety).

## 4. Deployed Q-CBF filter (the main claim) + Theorem-S

`A_ver(c) = {u∈U_menu : Enc(c,u) ⊆ R_∞}` (Lemma 5.1 saturation table): **mean 14.9
certified actions/cell, 84 % of R_∞ has a certified *racing* action** (the rest
brake-only). Three-way rollout from R_∞ under extremal+greedy `d`, res 44:

| policy | rule | **cbv** | min g | mean v |
|---|---|---|---|---|
| **certified shield** (`Q_θ` selector) | `Q_θ` picks within `A_ver(c)` | **0** | +0.17 | 0.01 |
| **certified shield** (progress selector) | max-accel within `A_ver(c)` | **0** | +0.17 | 0.10 |
| naive `Φ_θ` | race iff `Φ_θ` else brake (RFC) | **303** | −0.34 | 0.45 |
| pure brake | always brake | **0** | +0.10 | 0.01 |

**Safety is decoupled from the selector (the key structural fact):** *every*
in-`A_ver` selector is provably safe (cbv 0) — the verifier, not the chooser,
guarantees invariance. Naive `Φ_θ` (unrestricted) is **provably unsafe** (cbv 303,
min g −0.34). A revealing learned-object finding: `Q_θ` *within* `A_ver` prefers
**conservative/braking** actions (it was distilled toward the brake fallback), so
the `Q_θ`-selector shield is safe-but-slow (≈ pure brake); a **progress** selector
within the *same* certified `A_ver` reaches mean v 0.10 (10× brake) at the *same*
cbv 0. So the safety/performance split is clean: the certified envelope is the
deployment object; the selector trades progress for nothing in safety. **Theorem-S
quantitative:** over R_∞, **24.4 % of the `Φ_θ`-feasible (c,u) pairs escape R_∞**
(846 380 of 3 473 840) — the unrestricted learned predicate is massively
over-permissive, which is exactly why post-hoc verification is needed.

**Q-duality (write into the paper).** The *same* `Q_θ` is **over-permissive as a
safety predicate** (rejected; 24 % false-feasible) yet **near-perfect as an action
proposal**: among the grown cells the verified winning action is `Q_θ`'s **top-5
in 100 %** of cells (top-3 43 %, top-1 19 %, mean rank 2.5/20). "Q guides,
verifier proves" — demonstrated on the same network.

## 5. Empirical scaling extrapolation (with caveats)

`ρ_∞(h) = ρ* − a·h^p` fit (pure-numpy p-grid LSQ; **not** Richardson — `h_p,h_v,Δψ`
do not co-refine, so this is empirical scaling, reported with that caveat). The
5 npsi-16 points (h = 6/npx ∈ [0.075, 0.167]) extrapolate to
**`ρ_∞^learned(h→0) ≈ 0.9–1.0`** [finest grid 0.880] — the fit is sensitive (`p`
rails at the grid edge once the res-80 jump is included: 0.91 on the 4 coarse
points at p≈1.05, 1.00 on all 5 at p≈0.5), so the honest claim is "spatial
refinement recovers ≈ 90 %+ of Ω\* and is *consistent with* approaching the
reference reachable ceiling `Capt(K∩𝒱₀) = 0.97`". The ideal extrapolates to
`ρ*_ideal ≈ 0.61` for the shrunk-seed grow (clean p ≈ 1.2). The residual to 0.97
is one-step / heading conservatism that macro-actions / finer npsi / center-form
Enc remove; the residual *above* 0.97 (toward Viab ≈ 1.9× Ω\*) needs anchor
relaxation (§3).

---

## Verdict & reprioritised P2 levers

- **Go/no-go: GO.** `ρ_∞` is resolution-monotone (0.818→0.880) and Δρ **crosses
  +0.05 at res 80 (+0.0563)** with cbv 0 — the framework's resolution-monotone
  claim holds for the learned object. The contribution remains the **sound anytime
  lfp + deployed certified shield + quantitative Theorem-S**; ρ is the coverage
  metric, and the calibration shows the residual is recoverable conservatism
  (capture gap ≈ 0) inside a conservative anchor (Viab ≈ 1.9× Ω\*).
- **P2 levers** (resolution still helps — res 80 is GO — but the cheapest further
  headroom is elsewhere):
  1. **2–3-step verified macro-actions** (+ center-form Enc) — close the
     0.88→0.97 reachable-reference conservatism faster than grids; oracle-probe
     first (how many `frontier/k`-blocked cells are 2-step-recoverable).
  2. **Anchor relaxation** — Viab ≈ 1.9× Ω\* and the grow already steps past Ω\*;
     a multi-funnel seed (brake × left/right full-steer × low-v loiter) and/or a
     less conservative membership target accesses that headroom.
  3. **Joint spatial + heading** for the `v0`/`frontier` split (npsi 32 already
     lifts both ideal and learned at res 44).
- **Keep the main claim front and centre:** the deployed shield (Q within A_ver) +
  Theorem-S mass + Q-duality are the headline; ρ is the coverage metric, not the
  result.

## Reproduce

```bash
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44 --full   # grow + calib + deploy
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 56          # resolution trend
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44 --npsi 32# heading-conservatism
python experiments/f1tenth_e2/analyze_p1_grow.py                    # sweep table + fits + calib
python -m tests.test_bicycle_accel                                  # primitive soundness (+ heading)
```
