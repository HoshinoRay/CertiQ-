# P1 grow-from-seed — technical note (method, primitives, soundness)

Engine for the construction in `grow_from_seed_certified_expansion.md`: given a
frozen learned robust Q-CBF trio `(V_θ, Q_θ, π♭)` and an already-certified brake
seed `S_brake`, build a **sound, anytime** robust forward-invariant set that grows
outward from the seed, and deploy a certified Q-CBF shield on it.

---

## 1. Objects and trust anchors

Discrete-time disturbed bicycle `x_{t+1}=f(x_t,u_t,d_t)`, `x=(px,py,ψ,v)`,
`u=(a,δ)`, `d=(d_a,d_δ)∈D` (compact, adversarial full-state feedback).
Constraint set `𝒦={g≥0}` (obstacle + wall). Learned trio is a plain ReLU MLP
tuple — **no analytic `D(v)`, no hard-Lipschitz**.

Safety depends only on three trust anchors (the learned nets are *not* among them):

- **A2 — seed.** `R₀ = S_brake`: each cell has a verified brake-to-stop
  trajectory keeping `lb V_θ≥0 ∧ g≥0` for every heading and `∀d`; `v=0` is a
  robust fixed point. (Delivered by the RFC funnel `brake_funnel_cert`.)
- **A3 — sound envelope.** A computable `Enc(c,u) ⊇ Reach(c,u) = {f(x,u,d):x∈c,d∈D}`,
  monotone under cell refinement.
- **A4 — sound membership.** `box ⊆ ⋃R` checked conservatively (closed cells,
  ψ wrap, outward float rounding).

`V_θ` enters **only** through the B1 anchor `𝒱₀={c: lb V_θ(c)≥0}` (a CROWN lower
bound — pure set subtraction, never an invariance assumption). `Q_θ` is used only
to *propose* / *select* actions and as a *reported metric* — never in the safety
chain.

## 2. The grow operator (lfp from below)

On the 4-D lattice `Grid = (px,py,ψ,v)` (the heading axis is needed because a
racing/steer action evolves ψ, unlike the heading-free brake successor):

```
Pre_ver(R) = { c : ∃u∈U_menu,  Enc(c,u) ⊆ ⋃R }
G_V(R)     = R ∪ ( 𝒦 ∩ 𝒱₀ ∩ Pre_ver(R) )
R_{k+1}    = G_V(R_k),   R₀ = S_brake (lifted to all headings),   R_∞ = ⋃_k R_k
```

`U_menu` = 4 accel × 5 steer = 20 fixed commands (A5). The seed is heading-free,
so `R₀` lifts to all ψ-cells of each `S_brake` cell (sound: every heading is
brake-certified). `ℓ(c)=min{k:c∈R_k}` is the layer; `u*_c` the witness action.

**Why lfp, not gfp.** Both directions share the fixed-point equation
`S = R₀ ∪ (𝒦∩Pre(S))`, but the gfp (carve from `𝒦` inward) is over-approximate
until convergence — at finite resolution it mistakes an un-converged set for an
invariant one (the RFC naive racing closure: cbv=205). The lfp (grow from `R₀`
out) is **under-approximate at every step**: each iterate is *already* a sound
robust invariant set (Theorem A), so containment is always a single one-step box
test (no wrapping/erosion).

**Theorem A (anytime soundness), layered policy σ.** At `x`, take `c=cell(x)`;
if `ℓ(c)=0` brake (π♭), else apply `u*_c`. For every adversarial `d`-sequence the
closed loop stays in `𝒦` ∀t and reaches `R₀` within `ℓ(c)` steps, then `v=0`.
Induction on `k`: the successor of `u*_c` lands in `⋃R_{k}` for *all* `d` by
A3+A4, layer strictly decreases, base case is A2. Hence **every** `R_k` has
`cbv≡0` by construction.

## 3. Sound primitives (the trusted computing base)

- **`successor_box`** (existing, tested): exact interval image of `(px⁺,py⁺,v⁺)`
  over a state×accel×`d_a` box (`cos/sin` interval over the heading cell, `v⁺`
  clipped to `[0,v_max]`).
- **`heading_successor_interval`** (new, `bicycle_accel.py`, tested
  `tests/test_bicycle_accel.py`): sound un-wrapped interval of
  `ψ⁺ = ψ + dt·(v/L)·tan(δ+d_δ)` over the `(ψ,v)` cell and full `d_δ`; `tan`
  monotone on the (here strictly-interior) argument range, `v/L≥0` ⇒ 4-corner
  yaw-rate product.
- **A4 membership / containment.** Outward-rounded inclusive cell-index ranges
  per axis (`floor((·−lo)/w ± ε)`); the periodic ψ-range is split into ≤2
  contiguous index ranges; out-of-`[p_lo,p_hi]` ⇒ pair invalid. "`box ⊆ R`" is a
  4-D **summed-area (prefix-sum) box query** (16-corner inclusion–exclusion,
  O(1) per box, wrap handled by summing the two ψ-ranges) compared to the box
  cell-count. A pair is *permanently invalid* if its touched cells leave `𝒦∩𝒱₀`
  (pre-filtered once via prefix sums of the lifted `𝒦` and `𝒱₀` masks).
- **MC self-check (A3).** 20 actions × 2·10⁵ random (state,d) samples per action;
  the true successor must lie in `Enc` — 0 violations at every resolution.

## 4. Failure attribution, A_ver saturation, deployment

- **4-way decomposition** of candidates never added, by most-recoverable reason
  (mutually exclusive): `frontier` (a valid action exists, blocked by a non-`R`
  cell), `v0` (stays in `𝒦` but leaves `𝒱₀` — the learned anchor binds), `k`
  (leaves `𝒦`), `domain`. The `v0` learned-minus-ideal excess is the `lb V_θ`
  CROWN looseness; it tightens with spatial resolution.
- **`A_ver(c) = {u : Enc(c,u) ⊆ R_∞}`** (Lemma 5.1 saturation table, a per-cell
  20-bit mask over `R_∞`). Every selector within `A_ver` keeps `R_∞` invariant.
- **Deployed shield.** Runtime: pick within `A_ver(c)` (e.g. `Q_θ`-argmax, or a
  progress objective); brake if `A_ver` empty (pure seed). Safety is the
  verifier's, performance the selector's.
- **Theorem-S mass.** `#{(c,u): u∈Φ_θ(center(c)), Enc(c,u)⊄R_∞}` quantifies the
  over-permissiveness of the unrestricted learned predicate `Φ_θ`.

## 5. Capture-basin reference (calibration only — NOT a certificate)

To split the residual `Ω*∖R_∞` into recoverable conservatism vs intrinsic gap, a
reference robust capture basin is computed by backward reachability on the
**sampled true dynamics** (cell centres, adversarial over a 3×3 `d`-grid):
`Capt = lfp_C[ seed ∪ {c∈dom: ∃u ∀sampled d, cell(f(centre c,u,d))∈C} ]`. It is a
NOMINAL (optimistic) reference, so `R_∞ ⪅ Capt_ref ⪅ true capture basin`; the
recovery fraction `|R_∞|/|Capt(𝒦∩𝒱₀)|` upper-bounds the method conservatism, and
`|Capt(𝒦)|` exposes whether the true capture basin exceeds the brake-anchored
`Ω*` (Viab > Ω*).

## 6. Complexity

Candidate cells = headings of `(𝒦∩𝒱₀)∖seed` (~10⁴–10⁵ at res 44–80). Each `Enc`
index box is precomputed once; the lfp re-tests only prefix-sum containment per
wave (≈ seconds total). The expensive stage is the *seed* funnel (CROWN per
brake step). Pure NumPy; the certificate is the table `{(c,u*_c,ℓ(c))} ∪ S_brake`,
independently re-checkable.

## 7. Files

| file | role |
|---|---|
| `qcbf/dynamics/bicycle_accel.py` | `successor_box`, **`heading_successor_interval`**, `brake_cbf_bounds`, `g_bounds_sq` |
| `qcbf/certify/grow.py` | `Grid4D`, `GrowEngine` (lfp, prefix-sum containment, breakdown, `A_ver`) |
| `qcbf/certify/viab_reference.py` | reference robust capture basin (calibration) |
| `experiments/f1tenth_e2/run_cert_p1_grow.py` | driver (`--full` calib+deploy, `--dump` grid, `--res/--npsi`) |
| `experiments/f1tenth_e2/analyze_p1_grow.py` | sweep table + scaling fit + decomposition |
| `experiments/f1tenth_e2/make_p1_figures.py` | all figures |
| `tests/test_bicycle_accel.py` | primitive soundness (incl. heading interval) |

```bash
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44 --full
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 80
python experiments/f1tenth_e2/analyze_p1_grow.py
python experiments/f1tenth_e2/make_p1_figures.py
python -m tests.test_bicycle_accel
```
