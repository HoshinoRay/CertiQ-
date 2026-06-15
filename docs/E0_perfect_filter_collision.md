# Dubins-E0 — Does the ground-truth HJ filter collide? (CORRECTED)

*Fixed-speed Dubins car. Ground-truth `V_HJ`/`Q_HJ` are **grid tables from value
iteration — NOT neural networks**, so there is no CROWN and no fitting error here.
Script: `experiments/dubins_e0/run_perfect_filter_collision.py`.*

> **CORRECTION (supersedes the first draft of this file).** The first version
> reported "the perfect filter collides 14.8% at d=0 / 65.7% under the adversary"
> and framed it as the discounted value being un-certifiable. That framing was
> **wrong** — not a bug in the value/controller/sim, but a sampling error on my
> part: I deployed from the *entire* `{V_HJ≥0}` at `γ_teach=0.92`, which is an
> **optimistic over-approximation** of the true viability kernel. Starting cars in
> the over-claimed shell and then counting collisions was circular. **The HJ/Q
> foundation is sound; the ground-truth filter is safe on the true kernel.**

## 0. The actual fact about the discounted (Fisac) value

`V = (1−γ)g + γ·min(g, max_u min_d V(f))`.

- **γ=1 (undiscounted).** `{V≥0}` is the true viability kernel and IS robustly
  forward-invariant: the greedy controller keeps `g≥0` forever, for all `d`. **0%
  collision, guaranteed** (this is the HJ reachability theorem).
- **γ<1.** The `(1−γ)g` term lifts the value, so `{V_γ≥0}` is an **outer (optimistic)
  over-approximation** of the kernel by a shell of thickness `~(1−γ)g`. Proof:
  `T_γ[V_∞] = (1−γ)g + γV_∞ ≥ V_∞` (since `V_∞≤g`), and `T_γ` monotone ⇒ `V_γ≥V_∞` ⇒
  `{V_γ≥0} ⊇ {V_∞≥0}`. The shell is **not** truly safe.

So "collisions when started from `{V_γ≥0}`" is **expected and correct** — those
starts are in the optimistic shell, never in the true safe set. The number says
how thick the γ=0.92 shell is, nothing more.

## 1. The decisive experiment: γ-sweep (collision → 0 as γ → 1)

Recompute `V_HJ` by VI at several `γ_teach`, deploy the HJ-greedy filter
(`u*=argmax_u min_d V_HJ(f)`, re-evaluated continuously) from `Ω*={V≥0}`, H=80,
5000-cell sample:

| γ_teach | VI resid | Ω* frac | d=0 (robust greedy) | d=0 (nominal greedy) | worst-case adv |
|--------:|---------:|--------:|--------------------:|---------------------:|---------------:|
| 0.92    | 9e−6     | 0.423   | 14.3%               | 14.3%                | 65.2%          |
| 0.97    | 1e−5     | 0.280   | **0.0%**            | **0.0%**             | 58.5%          |
| 0.99    | 1e−5     | 0.076   | **0.0%**            | **0.0%**             | 12.2%          |
| 0.995   | 2e−5     | 0.000   | (grid VI collapses) | —                    | —              |

Reading:
1. **d=0 collision 14.3% → 0.0% by γ=0.97**, and stays 0. The d=0 "collisions" at
   γ=0.92 were ENTIRELY the optimistic `(1−γ)g` shell; remove it (raise γ) and the
   ground-truth filter is exactly collision-free at d=0, as HJ theory requires.
2. **Ω\* shrinks 0.42 → 0.28 → 0.076** toward the true robust kernel as γ→1.
3. **The adversary number falls too** (65→58→12%) — same shell, now under
   adversarial value-decay; it →0 as γ→1 but the **grid VI collapses near γ=0.995**
   (Ω*→0, the multilinear-interp breakdown the oracle docstring warns about). This
   collapse is exactly why `γ_teach<1` is used: a numerical convenience, not a
   safety claim.

## 2. Honest takeaways

1. **No bug; the foundation is sound.** Value, controller, and simulator are
   correct (controller verified 0/1500 mismatch vs a per-state reference; the
   discounted fixed-point identity `max_u min_d V(f)=(V−(1−γ)g)/γ` holds; d=0
   collision →0 as γ→1). The ground-truth HJ filter is absolutely safe on the true
   kernel, in this deterministic sim.

2. **`{V_θ≥0}` is NOT the safe set at `γ_teach=0.92`** — it is ~15% larger than the
   viability kernel (the d=0 shell), and the adversary can drain the discount
   margin across it. Therefore:
   - A sound certificate must target the **certified subset**, not all of `{V≥0}`.
     The repo already does this — `qcbf/audit/falsify.py` samples rollouts from
     `filt.accepted`, never from `{V≥0}`. This experiment **confirms** that design.
   - A tighter `γ_teach` (~0.97–0.99) yields a genuinely-safe, tighter reference
     set at the cost of a smaller `Ω*` — a real, quantified design knob (the
     numerically-usable window is `γ ≲ 0.99` before VI collapse).

3. **What this does NOT say.** It does NOT say the perfect filter fails, that the
   discounted value is "un-certifiable," or that the Q foundation is wrong. The
   earlier draft's dramatic framing is retracted.

## 3. (Superseded) the γ=0.92 numbers, for the record

From all 28 080 `Ω*={V_0.92≥0}` cells, H=150: d=0 14.8%, d=±0.3 34.1% (symmetric —
the GT filter has no L/R bias, unlike the chiral learned witness), random 15.2%,
matched adversary 65.7%; first-violation mostly the world boundary. These are
reproducible and correct **as a measurement of the γ=0.92 optimistic shell** — but
they are NOT a property of "the perfect filter," because the starts include the
non-safe shell. See §1 for the corrected interpretation.
