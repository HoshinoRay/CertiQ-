# Development Log and Engineering Rules

This file is the running technical development document for this repository.
Every meaningful code, config, training, certification, or artifact-interface
change must update this file in the same edit cycle.

The goal is simple: keep the project clean, auditable, and aligned with the
main certificate objective. Do not let local fixes, temporary scripts, stale
artifacts, or unclear interfaces corrupt the research direction again.

## Active Project Line

Current active line:

\[
\texttt{experiments/dubins\_e0}
\]

Current active artifact:

\[
\left(V_\theta,\; Q_\theta,\; \pi_\theta\right)
\]

Current teacher/oracle object:

\[
V_{\mathrm{HJ}}, \qquad
Q_{\mathrm{HJ}}(x,u,d)=V_{\mathrm{HJ}}(f(x,u,d)).
\]

The deployed certificate is checked at

\[
\gamma_{\mathrm{deploy}}=0.9,
\]

while the teacher value iteration uses the separate discount

\[
\gamma_{\mathrm{teach}}=0.92.
\]

These two quantities must not be collapsed into one variable. The teacher is a
label/reference generator; the certificate is on the frozen learned networks.

## Non-Negotiable Development Rules

1. Keep the main objective above local convenience.
   No small detail, workaround, metric, plotting need, or temporary experiment is
   allowed to violate the core certificate design:

   \[
   \text{prove the deployed robust Q-CBF filter on the frozen artifact.}
   \]

2. Keep the codebase clean.
   Do not leave duplicate pipelines, half-deleted experiment families, stale
   scripts, ambiguous names, or hidden old interfaces. If code is no longer part
   of the active line, either remove it cleanly or document why it remains.

3. Keep interfaces explicit.
   Every artifact-consuming stage must know exactly which config and weights
   produced the artifact it reads. Do not let audit, figures, reports, or
   downstream analysis silently read stale files.

4. Keep artifacts separate from source.
   Generated files belong under `results/<experiment_name>/`. Source code,
   configs, and docs must not depend on undeclared files outside that directory.

5. Never report a certificate from an unchecked or mismatched artifact.
   A certification mask is valid only for the exact tuple

   \[
   \left(\text{config}, V_\theta, Q_\theta, \pi_\theta\right)
   \]

   used to generate it.

6. Do not weaken the theorem to make a number look better.
   If a detail makes \(\mathrm{C1}\), \(\mathrm{C3}\), or \(\mathrm{C4}\) fail,
   record the failure and fix the model/spec/code honestly. Do not redefine the
   claim after seeing a bad result.

7. Every substantial modification must update this document.
   Required update fields: what changed, why, affected files, generated
   artifacts, command/config used, pass/fail status, and next risk.

## Certificate Interface Contract

`certificate.npz` must bind the mask to the exact current artifact. Required
fields:

- `config_hash`
- `v_hash`
- `q_hash`
- `pi_hash`
- `strict_spec_pass`
- `n_cells`
- `gamma_deploy`
- `accepted`

`run_audit.py` must reject a certificate if any of the following differ from
the current run:

\[
\texttt{config\_hash},\quad
\texttt{v\_hash},\quad
\texttt{q\_hash},\quad
\texttt{pi\_hash}.
\]

If

\[
\texttt{strict\_spec\_pass}=\mathrm{false},
\]

audit must stop and report that no certified closed-loop audit is meaningful.

## Certification Logic

The strict deployed specification is:

\[
\mathrm{C1}:\quad \{x:V_\theta(x)\ge 0\}\subseteq \mathcal K,
\]

\[
\mathrm{C3}:\quad
\min_{d} Q_\theta(x,\pi_\theta(x),d)
\ge
\gamma_{\mathrm{deploy}}V_\theta(x)+\varepsilon,
\]

\[
\mathrm{C4}:\quad
\min_{d} Q_\theta(x,u,d)
\le
\min_{d} V_\theta(f(x,u,d)),
\]

for all deployed menu actions and the deployed witness action.

The robust one-step condition

\[
V_\theta(f(x,u,d))\ge \gamma_{\mathrm{deploy}}V_\theta(x)
\]

is implied by the runtime gate plus \(\mathrm{C4}\). It is not a separate old
style \(\mathrm{C2}\) sweep.

If \(\mathrm{C1}\) fails, strict certification has already failed. The verifier
may fail fast and skip expensive \(\mathrm{C3}\)/\(\mathrm{C4}\) work.

## Update: 2026-06-13 Clean Retrain

Reason:

Old generated artifacts existed under `results/smoke/`, including
`oracle.npz`, `v.npz`, `q.npz`, `pi.npz`, and `certificate.npz`. They were
residual outputs from before the latest code cleanup and could confuse later
audit or reporting.

Action:

1. Deleted old generated results.
2. Re-ran the current clean pipeline using:

   ```bash
   python experiments/dubins_e0/run_all.py --config experiments/dubins_e0/config_pilot.json
   ```

3. The full `run_all.py` attempt timed out during the expensive
   \(\mathrm{C3}\)-subsplit certification stage, after oracle and training had
   already completed.
4. Added \(\mathrm{C1}\) fail-fast behavior in `qcbf/certify/spec.py`.
5. Re-ran:

   ```bash
   python experiments/dubins_e0/run_certify.py --config experiments/dubins_e0/config_pilot.json
   python experiments/dubins_e0/run_audit.py --config experiments/dubins_e0/config_pilot.json
   ```

Current clean artifact directory:

```text
results/dubins_e0_pilot/
```

Generated files:

- `oracle.npz`
- `v.npz`
- `q.npz`
- `pi.npz`
- `certificate.npz`
- `train_report.json`
- `cert_report.json`
- `audit_report.json`
- `manifest.json`

Manifest:

- experiment: `dubins_e0_pilot`
- config hash: `24cff0d8007d`
- oracle residual: \(8.702278137207031\times 10^{-6}\)
- oracle iterations: \(76\)
- \(V_\theta\) hash: `732e5fc70c20`
- \(Q_\theta\) hash: `ca6c027a0876`
- \(\pi_\theta\) hash: `deb3a595f3b7`

Training diagnostics:

- \(V_\theta\) MSE: \(0.00691047684745599\)
- \(Q_\theta\) MSE: \(0.03509685587915567\)
- \(\pi_\theta\) MSE: \(0.24975225897038372\)
- witness margin mean: \(0.010054307020833979\)
- witness margin \(5\%\) percentile: \(-0.17306918660572262\)
- witness margin fraction above \(\varepsilon\): \(0.4874507127691841\)

Certification result:

\[
\texttt{strict\_spec\_pass}=\mathrm{false}.
\]

Failure mode:

\[
\mathrm{C1}\text{ failed}.
\]

Key counts:

- total cells: \(64000\)
- `superlevel_possible`: \(34079\)
- `superlevel_inner`: \(22813\)
- `c1_bad`: \(5709\)
- accepted cells: \(0\)
- best \(\rho\): \(0.0\)

Interpretation:

The newly trained \(V_\theta\) has cells that may intersect
\(\{x:V_\theta(x)\ge0\}\) while the exact lower bound on the safety margin is
unsafe. Therefore the learned artifact cannot currently certify

\[
\{x:V_\theta(x)\ge0\}\subseteq \mathcal K.
\]

Audit result:

Audit correctly stopped because the strict spec failed:

```json
{"pass": false, "reason": "strict spec failed"}
```

This is the correct behavior. A rollout audit from an uncertified artifact must
not be presented as a certified closed-loop result.

## Known Current Problem

The main blocker is not artifact mismatch anymore. The pipeline is now clean
enough to reveal the real issue:

\[
V_\theta \text{ is too optimistic or too loose near unsafe regions, causing }
\mathrm{C1}\text{ failure}.
\]

Next work should focus on making \(V_\theta\)'s nonnegative superlevel set stay
inside \(\mathcal K\), for example by improving labels, adding safety-floor
training pressure, changing the certified level set, or revisiting the theorem
target. Any such change must preserve the main deployed certificate logic.

## Update: 2026-06-13 Oracle Strict-Spec Test

Purpose:

Test the same strict-spec logic used for neural certification, but on the
oracle ground-truth objects instead of the learned networks:

\[
V_{\mathrm{HJ}},\qquad
Q_{\mathrm{HJ}}(x,u,d)=V_{\mathrm{HJ}}(f(x,u,d)),\qquad
\pi_{\mathrm{HJ}}(x)=\arg\max_u\min_d Q_{\mathrm{HJ}}(x,u,d).
\]

Important rule:

This test intentionally does **not** include old \(\mathrm{C2}\). It uses the
same semantic structure as the neural strict spec:

\[
\mathrm{C1},\qquad \mathrm{C3},\qquad \mathrm{C4}.
\]

Files added:

- `experiments/dubins_e0/run_oracle_spec.py`

Command:

```bash
python experiments/dubins_e0/run_oracle_spec.py --config experiments/dubins_e0/config_pilot.json --n-mc 100000
```

Output:

- `results/dubins_e0_pilot/oracle_spec_report.json`
- `manifest.json` stage: `oracle_spec`

Result summary:

Grid-node oracle test:

- active \(V_{\mathrm{HJ}}\ge0\) states: \(52752/124848\)
- \(\mathrm{C1}\) bad states: \(0\)
- \(\mathrm{C3}\) failures: \(14664\)
- \(\mathrm{C3}\) pass fraction: \(0.7220200181983621\)
- \(\mathrm{C3}\) minimum margin: \(-0.13500341773033142\)
- \(\mathrm{C4}\) menu: pass
- \(\mathrm{C4}\) witness: pass
- overall grid \(\mathrm{C1}/\mathrm{C3}/\mathrm{C4}\): fail

Monte-Carlo oracle interpolation test:

- samples: \(100000\)
- active \(V_{\mathrm{HJ}}\ge0\) samples: \(43928\)
- \(\mathrm{C1}\) bad samples: \(20\)
- \(\mathrm{C3}\) failures: \(12193\)
- \(\mathrm{C3}\) pass fraction: \(0.7224321617191768\)
- \(\mathrm{C3}\) minimum margin: \(-0.14653341472554238\)
- \(\mathrm{C4}\) menu: pass
- \(\mathrm{C4}\) witness: pass
- overall MC \(\mathrm{C1}/\mathrm{C3}/\mathrm{C4}\): fail

Interpretation:

The oracle construction satisfies \(\mathrm{C4}\) exactly because

\[
Q_{\mathrm{HJ}}(x,u,d)=V_{\mathrm{HJ}}(f(x,u,d)).
\]

On grid nodes, \(\mathrm{C1}\) also passes for \(\{V_{\mathrm{HJ}}\ge0\}\). However,
\(\mathrm{C3}\) fails for a substantial boundary/interior subset at
\(\gamma_{\mathrm{deploy}}=0.9\) and \(\varepsilon=0.005\). This confirms that the
discounted teacher is not itself a strict deployed Q-CBF certificate at the
current deployed decay; it is a label/reference object.

The small Monte-Carlo \(\mathrm{C1}\) failures indicate interpolation can produce
\(V_{\mathrm{HJ}}\ge0\) slightly outside \(\mathcal K\), which is consistent with
the warning that the oracle is not the proof object.

## Update: 2026-06-14 Learned Strict-Spec Diagnostic

Purpose:

Test the learned artifact

\[
\left(V_\theta,Q_\theta,\pi_\theta\right)
\]

with the same neural strict-spec semantics:

\[
\mathrm{C1},\qquad \mathrm{C3},\qquad \mathrm{C4}.
\]

This diagnostic intentionally does **not** include old \(\mathrm{C2}\). The
official certificate can fail fast after \(\mathrm{C1}\), but this diagnostic
continues on a deterministic random subset of active-after-\(\mathrm{C1}\) cells
to measure how \(\mathrm{C3}\) and \(\mathrm{C4}\) behave.

Files added:

- `experiments/dubins_e0/run_learned_spec_diagnostic.py`

Command:

```bash
python experiments/dubins_e0/run_learned_spec_diagnostic.py --config experiments/dubins_e0/config_pilot.json --max-c3-cells 2048 --max-c4-cells 2048
```

Output:

- `results/dubins_e0_pilot/learned_spec_diagnostic_report.json`
- `manifest.json` stage: `learned_spec_diagnostic`

Artifact hashes:

- config hash: `24cff0d8007d`
- \(V_\theta\) hash: `732e5fc70c20`
- \(Q_\theta\) hash: `ca6c027a0876`
- \(\pi_\theta\) hash: `deb3a595f3b7`

Full-lattice \(\mathrm{C1}\):

- total cells: \(64000\)
- `superlevel_possible`: \(34079\)
- `superlevel_inner`: \(22813\)
- \(\mathrm{C1}\) bad cells: \(5709\)
- bad fraction among possible superlevel cells: \(0.16752252120073946\)
- \(\mathrm{C1}\): fail

Active cells after removing \(\mathrm{C1}\)-bad cells:

- \(28370\)

Sampled \(\mathrm{C3}\) witness-gate diagnostic:

- sampled cells: \(2048\)
- pass: \(252\)
- fail: \(1796\)
- pass fraction: \(0.123046875\)
- minimum margin: \(-0.9403028536453979\)
- \(1\%\) margin percentile: \(-0.6998643383021488\)
- \(5\%\) margin percentile: \(-0.4694947338160576\)
- mean margin: \(-0.15108757384496535\)

Sampled \(\mathrm{C4}\) menu diagnostic:

- sampled cells: \(2048\)
- pass: \(1\)
- fail: \(2047\)
- pass fraction: \(0.00048828125\)
- minimum margin: \(-1.3173977263234868\)
- mean margin: \(-0.42729179468977213\)

Sampled \(\mathrm{C4}\) witness diagnostic:

- sampled cells: \(2048\)
- pass: \(0\)
- fail: \(2048\)
- pass fraction: \(0.0\)
- minimum margin: \(-2.9575244305240744\)
- mean margin: \(-0.7322201496980834\)

Interpretation:

The learned artifact is much weaker than the oracle strict-spec diagnostic:

- \(\mathrm{C1}\) already fails on the learned \(V_\theta\), so the official
  certificate correctly fails.
- \(\mathrm{C3}\) is not zero, but it is poor: only about \(12.3\%\) of the
  sampled active-after-\(\mathrm{C1}\) cells pass.
- \(\mathrm{C4}\) is the major learned-model failure: the sampled menu check
  passes only \(1/2048\), and the sampled witness check passes \(0/2048\).

This indicates that the learned \(Q_\theta\) does not reliably understate or
match the learned successor value \(V_\theta(f(x,u,d))\). The problem is not
only policy feasibility; the learned \(V_\theta,Q_\theta\) pair is structurally
inconsistent with the strict deployed certificate condition

\[
\min_d Q_\theta(x,u,d)
\le
\min_d V_\theta(f(x,u,d)).
\]

## Core Lesson: Do Not Destroy the Q-CBF Object to Get a Nonzero Set

This is a hard development rule.

Earlier attempts drifted toward adding technical machinery just to make the
certified set nonzero or to increase pass counts. That is not acceptable if the
added machinery changes the object so much that the result is no longer a clean
robust Q-CBF certificate.

The central value of this project is not merely producing a nonempty mask. The
central value is preserving a meaningful, auditable certificate for a deployed
state-action robust Q-CBF filter:

\[
\left(V_\theta,Q_\theta,\pi_\theta\right),
\qquad
\min_d Q_\theta(x,u,d)\ge \gamma_{\mathrm{deploy}}V_\theta(x),
\qquad
\min_d Q_\theta(x,u,d)\le \min_d V_\theta(f(x,u,d)).
\]

Therefore:

1. Do not add ad hoc filters, hidden fallback logic, post-hoc masks, or special
   case patches whose only purpose is to inflate the certified set.
2. Do not change the theorem after seeing a failure.
3. Do not introduce structures that make the artifact stop being a genuine
   learned Q-CBF object.
4. Do not bury failure behind plotting, c-sweeps, extra gates, or hand-tuned
   carve-outs unless they are explicitly part of the theorem and runtime loop.
5. Every new training loss, architecture constraint, verifier tightening, or
   set restriction must answer this question:

   \[
   \text{Does it preserve the deployed Q-CBF semantics and make the proof more honest?}
   \]

   If the answer is no, it is not allowed.
6. Do not use a model-based, traditional, non-scalable rescue as the core
   method. If a proposed addition depends on hand-crafted model-specific
   controllers, grid dynamic programming at deployment scale, exhaustive set
   carving, non-scalable reachability sweeps, or special-case geometry that only
   works for the toy Dubins setting, then even a successful result would be
   self-defeating. It would abandon the scalable learned Q-CBF direction.
7. A new architecture or training mechanism must be compatible with the later
   learning path: reinforcement learning, adversarial training, and larger
   systems. A method is stronger if it can be injected as a loss, constraint,
   differentiable module, verifier-aware regularizer, or adversarial objective
   in those later stages. A method is suspicious if it only works as an offline,
   hand-tuned, model-specific patch.

Allowed directions are those that improve alignment with the existing theorem:

- training \(Q_\theta\) and \(V_\theta\) to satisfy the conservative consistency
  direction required by \(\mathrm{C4}\);
- improving \(V_\theta\)'s safety boundary so \(\mathrm{C1}\) becomes true;
- improving \(\pi_\theta\)'s robust gate margin so \(\mathrm{C3}\) becomes true;
- tightening CROWN/interval bounds without changing the mathematical claim;
- making a theorem-level restriction explicit only if the deployed runtime uses
  the exact same restriction.
- adding losses or architectures that can later be reused in RL/adversarial
  training, for example certificate-aligned one-sided consistency losses,
  adversarially sampled disturbances, or differentiable robust-margin training.

Forbidden directions are those that merely make metrics look better while
weakening the scientific object:

- certifying a set that the runtime does not actually respect;
- using a different \(\gamma\), gate, policy, or disturbance set in certification
  than in deployment;
- replacing \(Q_\theta\) with an arbitrary score that no longer represents a
  successor-value/action-value object;
- reporting rollout success as a certificate;
- hiding failed \(\mathrm{C1}\), \(\mathrm{C3}\), or \(\mathrm{C4}\) conditions behind
  secondary summaries.
- using the oracle grid, handcrafted geometry, model-specific dynamic
  programming, or any other non-scalable traditional controller as the deployed
  safety mechanism while still presenting the result as a scalable learned
  Q-CBF method.
- adding a trick that cannot plausibly transfer to later RL or adversarial
  training pipelines except as a one-off offline patch.

Decision test for every proposed addition:

\[
\begin{aligned}
&\text{Q1: Does it preserve }(V_\theta,Q_\theta,\pi_\theta)\text{ as the proof object?}\\
&\text{Q2: Does it keep the runtime gate and certificate gate identical?}\\
&\text{Q3: Does it scale beyond the current grid/Dubins setup?}\\
&\text{Q4: Can it be injected into future RL/adversarial training?}\\
&\text{Q5: Does it make the proof more honest rather than merely the metric larger?}
\end{aligned}
\]

If any answer is no, the change is rejected until the theorem and runtime story
are made explicit.

## Pre-Commit / Pre-Result Checklist

Before claiming a result:

1. Confirm the active experiment directory.
2. Confirm `manifest.json` matches the config.
3. Confirm `certificate.npz` provenance matches current weights.
4. Confirm \(\gamma_{\mathrm{teach}}\) is only used for teacher labels.
5. Confirm \(\gamma_{\mathrm{deploy}}\) is used by runtime gate, \(\mathrm{C3}\),
   and \(\mathrm{C4}\).
6. Confirm `strict_spec_pass` before any certified audit claim.
7. If spec fails, report the failing condition honestly.
8. Update this document with exact hashes and result status.

## Modification Log Template

Use this template for every future meaningful change:

```text
Date:
Purpose:
Files changed:
Artifact/config affected:
Commands run:
Result:
Certificate status:
Known risk:
Next action:
```

## Update: 2026-06-14 Model-vs-Verifier Decomposition (analysis, no code change)

Purpose:

The learned strict-spec diagnostic reports C3 ~12% and C4 ~0% pass, but those
numbers come from the INTERVAL verifier, which bundles two very different causes:
model error (the networks truly violate the inequality) and verification
looseness (the networks are fine pointwise but CROWN/successor-box intervals lose
it). The fix is opposite in each case, so they must be separated before any
change. Read-only probe over 1500 sampled active cells of the pilot artifact
(`732e5fc70c20`/`ca6c027a0876`/`deb3a595f3b7`), comparing, per condition, the
pointwise margin at the cell centre (exact net evals, true f, 9-point d-grid)
against the interval margin the real verifier computes.

Identity check (confirms the probe is faithful): `m_int = m_point - slack_V -
slack_Q` holds to the digit on every condition.

| condition  | m_point (model, centre) | m_int (verifier) | slack split |
|---|---|---|---|
| C4-menu    | mean +0.025, 62% >=0  | mean -0.426, 0% >=0 | slack_V +0.228, slack_Q +0.223 |
| C4-witness | mean +0.054, 75% >=0  | mean -0.722, 0% >=0 | slack_V +0.378 (full [-w,w]), slack_Q +0.399 |
| C3         | mean -0.016, 44% >=0  | mean -0.738, 0% >=0 | slack +0.722 (whole-cell h3 CROWN) |
| C1         | exact gmin (no interval slack) | 45.8% of c1_bad cells have a genuine sampled point V>=0 & g<0; max V on g<0 = +0.265 |

Diagnosis:

- **C4 is NOT a model failure. It is verification looseness.** The learned
  Q_theta already under-estimates V_theta(f) at ~2/3 (menu) and ~3/4 (witness) of
  cell centres; the verifier discharges 0% only because the sound bounds lose
  ~0.45 (menu) / ~0.78 (witness). This reframes the whole problem: "C4 0/2048"
  was reading as a broken Q object; it is mostly a loose-bound artifact.
- The witness V-successor bound encloses the FULL control interval `[-w,w]`; that
  alone is +0.378 of slack. Confirmed lever (read-only A/B, same frozen weights):
  bounding `u=clip(pi(x))` per cell via the T4-tested `compile_policy` gives a
  per-cell range of mean width 0.55 (vs 2.0), lifts lb_Vsucc by **+0.13**
  (slack_V 0.378 -> 0.246), and `clip(pi(x)) in [u_lo,u_hi]` checks sound.
- **C1 is ~half genuine model overshoot** (V_theta>=0 where g<0, by up to +0.26)
  and ~half V-upper-bound looseness. The teacher V_HJ satisfies V<=g by
  construction, so the overshoot is pure fit error -> a safety-floor training
  pressure is the honest fix.
- **C3 is both**: pointwise witness margin is ~0 (mean -0.016, matches the
  training witness-margin mean +0.010) AND the whole-cell h3 CROWN slack is large
  (+0.722); the staged C3 subsplit only partly recovers this.

Caveat (kept honest): per-condition `slack` mixes irreducible cell-variation
(the verifier must bound the cell-worst point, not the centre) with recoverable
bound-looseness. The W1 A/B avoids this — it compares two SOUND bounds of the
same worst-case, so its +0.13 gain is purely recoverable.

Prioritized plan (each passes the Q1-Q5 decision test: preserves
(V,Q,pi) and the identical runtime gate; scales; RL/adversarial-injectable;
makes the proof more honest, not just the metric larger):

1. TIGHTEN THE VERIFIER FIRST (claim-preserving; object, runtime, gamma all
   unchanged; only replaces sound bounds with tighter sound bounds):
   - W1 witness control range: replace `[-w,w]` enclosure in
     `_witness_successor_v_lower` with per-cell `[u_lo,u_hi]` from
     `compile_policy` CROWN. Needs a soundness test (clip(pi(x)) in range).
   - T1 relational successor-value bound: V_theta(f(x,u,d)) cannot be compiled to
     a pure ReLU net (Dubins f has cos/sin), so use an affine CROWN lower
     functional of V_theta in its input (the previously-removed
     `crown_lower_affine` is the right tool) and substitute the analytic
     successor, minimising the cos/sin term exactly over the heading cell. This
     removes the box double-loss behind slack_V.
   - T2 cheap knobs: state-cell (esp. heading) subsplit for the C4 successor
     bound, and `ante_d_probes` 1->3 for the Q upper bound (cuts slack_Q's
     d-coverage component).
2. CERTIFICATE-ALIGNED TRAINING (object improves honestly; differentiable, hence
   reusable as RL/adversarial losses):
   - C1 safety-floor hinge `relu(V_theta(x) - g(x) + m1)`.
   - C4 one-sided consistency hinge `relu(Q_theta(x,u,d) - V_theta(f(x,u,d)) + m4)`
     so Q under-states the successor value with margin.
   - C3 stronger witness-margin (raise m_target / improve pi) so the pointwise
     C3 margin is comfortably positive, not ~0.

Sequencing rationale: step 1 changes nothing about the certified object or the
claim and is independently falsifiable on the FROZEN artifact (slack must drop),
so it is the safe first move; step 2 is then evaluated on a re-frozen artifact.
None of this adds c-sweeps, GFP, hand-tuned carve-outs, or model-specific rescue
(all forbidden above).

Files changed: none (analysis only; probes ran from a scratch dir outside the
repo). Certificate status: unchanged (still strict_spec_pass=false). Known risk:
T1 relational bound touches the TCB and needs its own soundness test before use.
Next action: implement W1 + its soundness test (smallest, safest, highest-value
tightening), re-measure witness C4 slack, then proceed to T1/T2.

## Design Constraint: Target Deployment Artifact Structure (Q-primary)

The toy currently distills V_theta and Q_theta INDEPENDENTLY from the oracle.
The intended deployment structure is different and all future training-side work
must stay compatible with it:

- Q_theta is the PRIMARY learned object, trained by RL (robust/adversarial).
- V_theta is a DISTILLED head of Q_theta:  V_theta(x) ~= max_u min_d Q_theta(x,u,d).
- pi_theta is the gate witness:  pi_theta(x) ~= argmax_u min_d Q_theta(x,u,d).

The certificate is unaffected at verification time: it always reads the frozen
V_theta / Q_theta / pi_theta NETWORKS directly, regardless of how V_theta was
produced. So the entire step-1 verifier tightening (W1/T1/T2) is
architecture-agnostic and remains valid as written.

Implications for step-2 training (must respect this):

- The C4 one-sided loss is really a Q-side BELLMAN-CONSISTENCY pressure: with
  V_theta = max_u min_d Q_theta, C4 (min_d Q_theta(x,u,d) <= min_d V_theta(f))
  becomes a self-consistency condition on Q_theta alone. For the true reach-avoid
  object Q(x,u,d)=V(f(x,u,d)) and V=max min Q give C4 with EQUALITY, so the honest
  alignment target is to train Q_theta toward Q_theta(x,u,d) <~ V_theta(f(x,u,d))
  -- a one-sided Bellman residual, which is exactly an RL-injectable loss (Q5
  yes). Do NOT design a C4 fix that assumes V_theta and Q_theta are unrelated.
- High-value candidate that also matches deployment: distill V_theta from
  max_u min_d Q_theta (instead of from V_HJ). This makes V_theta(f) and Q_theta
  Bellman-consistent by construction, directly attacking the C4 MODEL margin,
  and is the structure RL will use anyway. Evaluate it as part of step 2.
- C1 floor and C3 margin pressures then act on the Q-derived V_theta / pi_theta
  heads, not on an independent V network.

This note does not change current code; it constrains what step-2 additions are
allowed to assume.

## Update: 2026-06-14 Step-1 Verifier Tightening (W1 + heading subsplit + probes)

Purpose: claim-preserving tightening of the C4 bounds (the verifier must read
the SAME frozen networks; only sound bounds are replaced with tighter sound
bounds). Passes Q1-Q5.

Files changed:
- `qcbf/certify/spec.py`: replaced the two successor helpers with `_control_range`
  (per-cell CROWN range of `clip(pi(x))` from the compiled policy -- W1) and
  `_succ_v_lower` (heading-subsplit x d-subsplit successor lower bound, accepts a
  scalar menu action or the witness per-cell control range). `run_strict_spec_
  certificate` and `_check_witness_q_consistency` now take `pol_net`.
- `qcbf/config.py`: `CertConfig.c4_psi_subsplit` (default 1 = off; measured a
  minor lever, see below).
- `qcbf/verify/.../run_certify.py`, `run_learned_spec_diagnostic.py`: pass the
  compiled `pol_net`.
- `tests/test_soundness.py`: T9 -- `_control_range` encloses `clip(pi(x))` and
  `_succ_v_lower` lower-bounds `V(f)` over random in-cell (x,u-in-range,d). PASS.

Result (read-only, on the FROZEN pilot artifact `732e5fc70c20`...): C4-witness
mean margin -0.722 -> -0.568 (the W1 +0.15 lands), C4-menu -0.426 -> -0.400
(heading subsplit + probes barely move it). Pass rate stays ~0%.

Conclusion (decisive): the verifier tightening is sound and helps modestly, but
the dominant C4 gap is NOT recoverable looseness. The pointwise model margin is
near zero and the irreducible CELL-WORST gap (the verifier must bound the worst
point in a 0.1-wide cell, not the centre) dominates. So step-1 alone cannot give
a pass; the MODEL must be trained to satisfy the conditions with a margin that
clears the cell-worst interval slack. `c4_psi_subsplit` left default-off to keep
recert fast; raise it only if trig becomes the binding slack post-training.

Certificate status: unchanged (strict_spec_pass=false). Known risk: none (sound,
T9-tested). Next: certificate-aligned training.

## Update: 2026-06-14 Step-2 Certificate-Aligned Training (C1 floor, C4 one-sided, CBF decrease, weight decay)

Purpose: shape the LEARNED (V_theta,Q_theta,pi_theta) so the conditions hold with
a margin that survives the verifier's interval slack. All pressures are
differentiable hinges (training nudges, never proof assumptions; the verifier
still checks the frozen nets) and are exactly the form reusable as RL/adversarial
losses -- passes Q1-Q5.

Files changed:
- `qcbf/nets/mlp.py`: `train_v_cbf` (MSE-to-teacher + C1 safety-floor hinge
  `relu(V - g + floor_margin)` + CBF decrease hinge
  `relu(gamma V(x) + dec_margin - max_u min_d V(f(x,u,d)))` + weight decay) and
  `train_q_oneside` (Q TRACKS the frozen `V_theta(f)` from below:
  `MSE(Q, V(f)) + relu(Q - V(f) + c4_margin)`; no oracle Q labels -- Bellman-
  consistent, matches the Q-primary deployment structure). `train_v_floor`
  removed (superseded).
- `qcbf/config.py`: `TrainConfig` knobs `c1_floor_w/margin`, `c4_oneside_w/margin`,
  `v_dec_w/margin/n_u/n_d`, `weight_decay`.
- `experiments/.../run_train.py`: V via `train_v_cbf`, Q via `train_q_oneside`
  (drops the `oracle.q_star` labels).

Key design correction (logged so we don't repeat it): the FIRST C4 attempt pushed
Q arbitrarily low, which DESTROYED the C3 gate margin (witness margin +0.09 ->
-0.08). Fix: Q must TRACK V_theta(f) from just below, not be pushed down -- C4 and
C3 then both reduce to a property of V_theta/pi (the decrease margin), which is
why the explicit CBF decrease loss on V_theta is required.

Result so far (pilot, before the decrease loop): C1-bad 5709 -> 3077 (-46%, floor
working); Q tracks V(f) (c4_viol 0.005). But C3/C4 interval pass still ~0% because
the pointwise margins (~0.03-0.1) are far below the cell-worst slack (~0.3-0.4).

THE BINDING BARRIER (the real finding to attack): certifying C4 over a cell needs
`Q <= V(f) - slack`; C3 needs `min_d Q(x,pi,d) >= gamma V + eps`; together they
require the one-step DECREASE margin `min_d V(f(x,pi,d)) - gamma V` to exceed
`eps + slack`. The Fisac teacher's decrease margin is only ~(1-gamma_deploy)V
(~0.1V, ->0 at the boundary), far below the slack. Hence the two new levers:
(a) the CBF decrease loss makes the decrease margin a trained quantity, and
(b) weight decay lowers the network Lipschitz, which shrinks `slack`
(slack ~ cell_width x Lipschitz). Finer cells are the third (orthogonal) lever.

Certificate status: pilot retraining with the decrease loss now; numbers to be
appended. Known risk: large dec_margin/floor_margin shrink {V_theta>=0} (volume
cost) -- acceptable (a more conservative learned set is sound, not a theorem
change), but watch that it stays non-empty. Next: measure pilot C1/C3/C4 pass
fractions; tune (dec_margin, weight_decay, resolution); then full certify+audit.

Result (pilot, dev run with reduced epochs 30/25/30/30 for iteration speed;
config restored to 60/40/60/40 for the record; `train_v_cbf` decrease backup
vectorized to one batched forward -> 4.5 s/epoch):

| metric | baseline (no cert-losses) | + floor+C4 | + decrease+wd |
|---|---|---|---|
| C1 bad cells | 5709 | 3077 | 3639 |
| C3 interval pass (sampled) | ~12% | ~7% | ~13% (min margin -0.96 -> -0.75) |
| C4-menu interval pass | ~0.05% | ~0.1% | 0% (min -1.29) |
| C4-witness interval pass | 0% | 0% | 0% (min -1.81) |
| witness margin on Omega* (pointwise) | +0.010 | -0.029 | -0.002 (>=eps frac 0.42) |

Reading: the decrease loss + weight decay raised the POINTWISE witness/decrease
margin (mean ~0) and improved C3, but the INTERVAL C4 stays 0%.  Decisive: at
0.1-cell resolution the CROWN cell-worst slack is ~0.4 (menu) / ~0.8 (witness),
and mild weight decay (3e-4) does not shrink it near the ~0.1 achievable decrease
margin.  This is the slack-vs-margin barrier, now quantified on a trained model.

## Conclusion / Next Lever: the barrier is interval slack at fixed resolution

Established beyond doubt this session: a binary certificate over {V_theta>=0}
needs, on EVERY active cell, decrease_margin >= eps + cell_slack.  We can raise
decrease_margin to ~0.1 (decrease loss) and shave cell_slack a little (W1, weight
decay), but cell_slack ~ cell_width x network_Lipschitz ~ 0.3-0.8 still dominates.
The pointwise model is now essentially correct (Q ~ V(f) from below; decrease
margin ~0); the certificate fails almost entirely on bound looseness over cells.

The honest, scalable, log-compliant levers (in priority order), all of which keep
the object/runtime/claim fixed and are RL-injectable:

1. **Verifier-in-the-loop (IBP/CROWN) training.**  Minimize the actual CERTIFIED
   margin (the interval lb/ub the verifier computes), not the pointwise margin.
   This is the gold-standard certified-training method and is the principled fix:
   it trains the networks so the cell-worst bound -- not just the centre -- clears
   eps.  Differentiable; injectable into RL as a verifier-aware loss.  This is the
   recommended next build.
2. **Stronger Lipschitz control** (heavy weight decay / spectral norm / smaller
   nets): directly shrinks cell_slack; cheap to try first as a knob sweep.
3. **Finer cells + the T1 relational successor bound** (deferred earlier):
   cell_slack ~ width, and T1 removes the successor-box double-loss.  Combine so
   the lattice stays affordable.

Decision test: all three preserve (V,Q,pi) and the runtime gate, scale, and are
RL-injectable (Q1-Q5 pass).  None is a hand-tuned carve-out.

Files changed this entry: `qcbf/nets/mlp.py` (vectorized `train_v_cbf` decrease
backup).  Certificate status: strict_spec_pass=false (C4 interval not yet met).
Next action: implement lever 1 (IBP-in-the-loop margin training) OR run a quick
Lipschitz/resolution knob sweep (lever 2) to bound what is reachable without it.

## Update: 2026-06-14 Correction (NO Lipschitz shrink) + IBP-in-the-loop finding

REMOVED weight_decay entirely (config + both train fns + run_train).  Rationale
(hard rule going forward): a generic Lipschitz/weight-norm/spectral penalty just
FLATTENS V_theta,Q_theta everywhere and destroys the real safety/action-value
geometry -- it changes the learned OBJECT's semantics to make a bound pass.  That
is exactly the "硬凑/forced" path and is DISALLOWED.  The only allowed ways to
reduce interval slack are (a) verifier tightening (W1/T1/heading split -- doesn't
touch the object) and (b) verifier-in-the-loop training of the ACTUAL proof
condition (shapes the net to satisfy C4 at cell-worst, not a blanket flatten).

Built the principled lever's infrastructure:
- `qcbf/nets/mlp.py`: differentiable `ibp_forward`/`ibp_backward` (hand-coded
  backward, gradient-checked vs finite differences -- T10, max err 3.6e-9).  The
  verifier's own bounds in `verify/bounds.py` remain the separate trusted copy.
- `qcbf/nets/certified_train.py`: `train_q_certified` pushes
  `ub_IBP Q(C,u,D) <= lb_CROWN V(f(C,u,D)) - margin` on a pool of active cells
  (frozen-V successor lower bound precomputed once; IBP looser than CROWN, so
  IBP-pass => deployed-CROWN-pass).  Wired into run_train as an OPTIONAL stage.

FINDING (why it is OFF by default, `cert_c4_w=0`): pure IBP is far too loose for
the 64/96-wide 2-hidden-layer nets over a 5-D cell box -- the initial cell-worst
IBP violation is ~11 (vs the ~0.4 CROWN slack).  Driving the IBP upper bound down
that far CRUSHES Q (witness margin collapses to -1.05, C3 lost).  So naive
IBP-in-the-loop degenerates into the same forbidden flattening.  The honest fixes:
(1) CROWN-in-the-loop (differentiable backward-CROWN -- tight, the real build), or
(2) a CROWN-IBP eps-schedule (grow the cell box from 0, mix clean+certified loss)
so the net stays expressive.  The differentiable IBP + the cell-worst loss
scaffold are kept (T10-tested) as the foundation for either; the stage is gated
off so the default pipeline is not degraded.

Also (loss design, per review): the MSE-to-teacher is only an ANCHOR, not the
objective -- added `teacher_fit_w` (default 0.5) to down-weight it so it does not
fight the floor/decrease shaping.  Plain MSE is itself not obviously optimal; a
margin/sign-aware fit (care about the {V=0} level-set and the one-sided margins,
not pointwise magnitude) is a candidate refinement to try next.

Files changed: `qcbf/config.py` (drop weight_decay; add teacher_fit_w, cert_c4_*
OFF), `qcbf/nets/mlp.py` (ibp_*, fit_w, no wd), `qcbf/nets/certified_train.py`
(new), `experiments/.../run_train.py` (optional cert stage, teacher_fit_w),
`tests/test_soundness.py` (T10).  Certificate status: strict_spec_pass=false.
Soundness T1-T10 pass.  Next action (scope with user): differentiable
CROWN-in-the-loop OR CROWN-IBP schedule; and/or the margin/sign-aware fit loss.

## Update: 2026-06-14 CROWN-IBP eps-schedule (C4 lever) -- C4 SOLVED, C3 collapsed

Implemented fix (2) from the finding above: `train_q_certified` now ramps the
certified Q-input box from a point (eps=0, exact, no slack) to the full cell
(eps=1, the deployed condition) over the first `cert_eps_warmup_frac` of the cert
epochs, then holds at 1.  `_eps_q_box` builds center +- eps*half-width; at eps=1
it reproduces the cell x d-subbox EXACTLY (verified 1.1e-16).  This is the
textbook CROWN-IBP remedy for the pure-IBP collapse.  Added a C3 gate-health
proxy (max_u min_d Q(center,u,d) - gamma V(center)) reported before/after to watch
for collapse.  Scope: ONLY the C4 lever (V-fit/MSE untouched, per user).  Knobs:
`cert_eps_start=0.0`, `cert_eps_warmup_frac=0.5`, stage gated by `cert_c4_w`
(default 0; pilot config sets 1.0).  Self-review: eps-box exactness, functional
smoke, soundness suite (7 passed) -- ran clean before the experiment.

Matched A/B on the pilot (same config, only `cert_c4_w` 0 vs 1; V identical so C1
is unchanged), CROWN cell-worst on 2048 sampled active cells:

  metric (CROWN, sampled)     OFF (lever off)        ON (eps-schedule)
  ----------------------------------------------------------------------
  C4 menu     pass / mean     0.0%  / -0.390         99.0% / +0.881
  C4 witness  pass / mean     0.0%  / -0.602         86.6% / +0.567
  C3 gate     pass / mean     7.6%  / -0.198          0.0% / -1.424
  C1 bad / inner (V frozen)   3762 / 20837           3762 / 20837
  qcert IBP cell-worst eps=1  --                     5.04 -> 0.048
  qcert gate(center) feasible --                     36% -> 0%

TWO findings:

(1) The lever WORKS and is a genuine first: C4 went from 0% certifiable to 99%
(menu) / 87% (witness) on the REAL CROWN verifier -- the cell-worst margins
literally FLIPPED from negative (mean -0.39/-0.60) to positive (+0.88/+0.57).
Done soundly (IBP looser than CROWN, so the eps=1 IBP condition implies the
deployed CROWN C4) and WITHOUT forbidden flattening: it is verifier-in-the-loop
on the actual C4 proof condition, masked to violating cells only.  The eps-schedule
cured the pure-IBP numerical collapse (cell-worst 5.04 -> 0.048, smooth, no
divergence).  First time C4 has certified at scale in this project.

(2) It DESTROYED C3 by crushing Q's VALUE: gate proxy 36% -> 0% feasible, C3
0% pass, witness margin on Omega* frac>=eps 0.38 -> 0.00.  The C4 push (w=1.0)
overran the weak anchor (anchor_w=0.1) and dragged Q far below the C3 gate.  This
is the C3/C4 tension made exact: C4 wants ub Q <= lb V(f) (Q LOW); C3 wants
min_d Q(x,pi,d) >= gamma V + eps (Q HIGH).  Q must live in the band
[gamma V + eps,  lb V(f) - m]; the lever shoved it to the floor and past it.

DECISION (per the user's protocol "if schedule plateaus, escalate to
differentiable CROWN"): this is NOT a plateau -- the lever SUCCEEDED on its
objective.  So do NOT escalate to differentiable CROWN yet: it would face the
IDENTICAL tension (tightening C4's CROWN upper bound by lowering Q would equally
kill C3) and burn the heavy build on the wrong problem.  The right next step is to
make the lever TWO-SIDED/BALANCED: train C4 (ub_IBP Q <= lb V(f) - m) AND C3
(lb_IBP Q(x,pi,d) >= gamma V + eps) together, the same sound verifier-in-loop way,
so Q lands inside the band instead of at the floor.  Crucially, now that the lever
shrank Q's slack (5.04 -> 0.048), the binding term for a non-empty band is V's
OWN cell slack + m + eps vs V's decrease margin (v_dec_margin=0.12) -- i.e. the
remaining barrier has moved from Q to V.  Did NOT run certify/audit on the ON run:
C1 is unchanged (V frozen) so the strict cert fail-fasts on C1 regardless; the
diagnostic already covers C1/C3/C4 at full fidelity.

Files changed: `qcbf/nets/certified_train.py` (eps-schedule + `_eps_q_box`,
`_cellworst_c4_viol`, `_gate_health`), `qcbf/config.py` (cert_eps_start,
cert_eps_warmup_frac), `experiments/dubins_e0/run_train.py` (pass schedule knobs
+ gamma_deploy, persist qcert report), `config_pilot.json` (cert_c4_w=1.0).
Artifacts: OFF `results/dubins_e0_pilot_nocert/`, ON `results/dubins_e0_pilot/`.
Certificate status: strict_spec_pass=false (C1 unchanged; C3 collapsed; C4 now
~99%/87%).  Next (scope with user): two-sided balanced C4+C3 verifier-in-loop.

## Update: 2026-06-14 Two-sided C4+C3 verifier-in-loop -- tension RESOLVED, barrier -> V

Built the balanced lever (user-chosen): `train_q_certified` now trains BOTH gates
on the same sound IBP bounds at cell-worst, eps-scheduled:
  C4 (push DOWN ub):  ub_IBP Q(C,u,D)        <= lb V(f) - c4_margin   (all menu u)
  C3 (push UP   lb):  min_d lb_IBP Q(C,u*,D) >= gamma*ubV + eps        (best valid u*)
Soundness is symmetric: IBP ub >= CROWN ub (trained C4 => verifier C4) and IBP lb
<= CROWN lb (trained C3 => verifier C3); the C3 RHS gamma*ubV+eps is exactly the
verifier's witness-C3 threshold (spec.py).  Implementation: pass-1 forwards all
(u,d) and stores bounds+caches; pass-2 backprops C4 on all actions and C3 on the
per-cell best DOMAIN-VALID action (argmax_u of min_d lb), squeezing Q into the
band [gamma*ubV+eps, lb V(f)-m].  `c3_w=0` recovers the one-sided lever.  New knob
`cert_c3_w` (default 0; pilot 1.0).  Self-review: C3-only test raises menu-gate
feasibility 0.41->0.97 (sign/selection correct); eps=1 box exact; suite 7 passed.

3-way pilot A/B (same cfg; OFF cert_c4_w=0 / C4-only c3_w=0 / two-sided c3_w=1;
V identical so C1=3762 throughout), CROWN cell-worst on ~2048 sampled cells:

  cond (pass% | mean)   OFF (no lever)   C4-only (1-sided)   two-sided C4+C3
  ----------------------------------------------------------------------------
  C3 gate               7.6% / -0.198    0.0%  / -1.424       27.2% / -0.181
  C4 menu               0.0% / -0.390    99.0% / +0.881       21.0% / -0.341
  C4 witness            0.0% / -0.602    86.6% / +0.567        7.4% / -0.704
  witness margin frac>=eps (Omega*, pointwise)
                        0.380            0.000               0.518
  qcert: C4 IBP cellworst 5.04->0.048 (1-sided)            5.04->0.314 (2-sided)
         menu-gate(C3) feas  0->0 (1-sided)                0->0.30 (2-sided)
         center gate >=0     36->0% (1-sided)              36->58% (2-sided)

FINDING -- the C3/C4 tension is RESOLVED as a mechanism: two-sided is the ONLY
config where BOTH gates are simultaneously alive.  C3 27.2% is the best of all
three (beats even no-lever 7.6%); C4 menu/witness stay nonzero (21%/7.4% vs OFF
0%/0%); the pointwise witness margin frac 0.518 beats no-lever 0.380.  C4-only is
a corner solution (C4 maxed, C3 dead); two-sided moves Q off the corner into the
band.  This is sound and not flattening (verifier-in-loop on the real gates,
masked to violating cells).

BUT the JOINT certificate is still ~0, for two V-SIDE reasons the Q lever cannot
touch (V is frozen during Q-cert):
  (1) C1 still fails (3762 bad): {V>=0} leaks outside K -- independent, pre-existing,
      a V-floor problem; the strict cert fail-fasts on C1 regardless.
  (2) Band too narrow for the witness (C4-witness only 7.4%): the band
      [gamma*ubV+eps, lb V(f)-m] is non-empty iff lb V(f) - gamma*ubV >= m + eps +
      (cell slack).  The Q lever shrank Q's slack, so the binding term is now V's
      decrease margin (v_dec_margin=0.12) vs the combined V cell slack -- the
      barrier has MOVED from Q to V, exactly as predicted.

DECISION: NOT a plateau, do NOT escalate to differentiable CROWN (the two-sided Q
lever succeeded; the residual is V-side, which CROWN-in-loop on Q would not fix).
Next lever is V-SIDE: widen the band by training V with a LARGER decrease margin
(v_dec_margin 0.12 -> ~0.20-0.25 so lb V(f) - gamma*ubV clears the slack), and
strengthen the C1 floor (push {V>=0} strictly inside K to kill the 3762 bad
cells); then re-run the two-sided Q lever to fill the widened band.  c3_w is a
frontier knob (raises C3% at the cost of C4%); tune AFTER the band is widened.

Files changed: `qcbf/nets/certified_train.py` (two-sided: `_menu_gate_feas`,
pass1/pass2 split, C3 up-grad on best valid action, decoupled feas reporting),
`qcbf/config.py` (cert_c3_w), `experiments/dubins_e0/run_train.py` (gate_thresh =
gamma*ubV+eps, pass c3_w, print menu-gate feas), `config_pilot.json` (cert_c3_w=1.0).
Artifacts: OFF `results/dubins_e0_pilot_nocert/`, C4-only `dubins_e0_pilot_c4only/`,
two-sided `dubins_e0_pilot/`.  Certificate status: strict_spec_pass=false (C1
unchanged; C3 7.6->27%; C4 alive both gates).  Soundness suite 7 passed.

## Update: 2026-06-14 V-side band-widen + C1 floor -- C1 partly fixed, band-widen BACKFIRED

User-chosen V-side lever (config-only; train_v_cbf already supports the knobs, no
code change).  Raised the C1 floor (c1_floor_w 1->2, c1_floor_margin 0.08->0.15)
and the decrease margin (v_dec_margin 0.12->0.25) to widen the band
[gamma*ubV+eps, lb V(f)-m]; kept the two-sided Q lever (c4_w=c3_w=1).

4-way CROWN cell-worst (OFF / C4-only / two-sided V0.12 / two-sided V0.25):

  cond (pass% | mean)   OFF        C4-only     2sided V0.12  2sided V0.25
  ----------------------------------------------------------------------
  C3 gate               7.6/-.198  0.0/-1.42   27.2/-.181    33.6/-.132
  C4 menu               0.0/-.390  99.0/+.881  21.0/-.341    16.6/-.373
  C4 witness            0.0/-.602  86.6/+.567   7.4/-.704     3.2/-.738
  C1 bad                3762       3762        3762          2293
  witness-margin frac>=eps (Omega*, pointwise)
                        0.380      0.000       0.518         0.616

TWO findings:
(1) C1 floor WORKS (partial): bad 3762 -> 2293 (-39%) from the stronger floor.
    Directionally correct; needs to go further (higher w/margin) to reach 0.
(2) Band-widen via v_dec_margin BACKFIRED on C4: the binding C4-witness gate fell
    7.4% -> 3.2% (and C4-menu 21 -> 17%) even though the POINTWISE witness margin
    ROSE (0.52 -> 0.62) and C3 rose (27 -> 34%).  Mechanism: forcing a bigger
    pointwise decrease margin roughened V (dec_viol 0.27 high), which inflated
    V's CELL-WORST slack, so lb V(f) at cell-worst did NOT rise -- it fell.  Same
    slack-vs-pointwise lesson as the feature-map and the IBP findings, now on V:
    a bigger pointwise margin does not buy a tighter cell-worst bound.

REFRAME (the important part): the binding barrier is V's CELL-WORST SLACK -- ub V
(C1: {V>=0} leaks out) and lb V(f) (C4-witness).  Structural gap: Q got
verifier-in-the-loop CELL-WORST training (the two-sided IBP lever) but V is still
trained on POINTWISE hinges (train_v_cbf floor/decrease at sampled points).  That
is exactly why V's slack is now the wall.  Pointwise V-margin knobs cannot fix it
(can worsen it).  The principled fix = the V ANALOG of the Q lever: cell-worst
CROWN-IBP training of V's C1 floor (ub_IBP V(C) < 0 where g<0) and decrease
(lb_IBP V(f(C)) >= gamma*ubV + m), directly shrinking V's slack without flattening.
Alternatives: keep v_dec_margin LOW (0.12 was better for C4; 0.25 was a mistake),
push C1 floor harder to zero the 2293, and/or verifier-side finer-cell tightening.

Files: `config_pilot.json` only (c1_floor_w=2, c1_floor_margin=0.15,
v_dec_margin=0.25).  Artifacts: 2sided V0.12 `dubins_e0_pilot_2sided_v012/`,
2sided V0.25 `dubins_e0_pilot/`.  Certificate status: strict_spec_pass=false
(C1 bad 2293; C4-witness 3.2% is the binding gate).  Next: scope with user --
cell-worst V training (V analog of the Q lever) vs cheap knob revert + harder C1.

## Update: 2026-06-14 Cell-worst V training (train_v_certified) -- FAILED (C1 inflation)

Built train_v_certified (the V analog of the Q lever): cell-worst CROWN-IBP on
V's C1 floor (ub_IBP V(C) < -m where g<0) and decrease (min_d lb_IBP V(f(C,u*,D))
>= gamma*ub_IBP V(C) + m), eps-scheduled, with a V_HJ anchor; runs after the
pointwise train_v_cbf, before Q, so Q tracks the tighter V.  Self-review found and
fixed a real bug (copied train_q_certified's sc=1/(nu*nd) scaling, but here every
term fires once per cell -> it made the C1/dec push 8x too weak vs the anchor;
removed sc).  Isolated C1 mechanism verified (ub V on unsafe cells 0.18 -> -0.78,
leak 0.70 -> 0).  cert_v_w=2, c1/dec margin 0.10, anchor 1.0.

RESULT vs prev best (two-sided V0.12), CROWN cell-worst:
  cond (pass% | mean | min)   2sided V0.12          + cell-worst V
  ----------------------------------------------------------------------
  C3 gate                     27.2% / -.181 / -.881  90.8% / +.028 / -.077
  C4 menu                     21.0% / -.341 / -2.05   0.3% / -.079 / -.193
  C4 witness                   7.4% / -.704 / -3.31   1.5% / -.069 / -.176
  C1 bad                      3762                   30079
  C1 possible (ub V>=0)       32686                  63999

FAILED -- C1 DESTROYED: V inflated to >=0 almost everywhere (possible 32686 ->
63999), incl. deep in the obstacle, so C1 bad exploded 3762 -> 30079.  Three
compounding causes:
 (1) The DECREASE push inflates V globally.  Raising lb V(f) (successor values)
     propagates to raise V everywhere -- a self-referential inflation with NO
     safety clamp (unlike the teacher's Fisac min(g,.) backup).  The C1 floor +
     anchor could not contain it.
 (2) Pool too small for C1: vpool = cert_n_cells (4096) sampled cells, but there
     are 30079 C1-bad cells -- the C1 floor never even sees most of them.
 (3) The V_HJ anchor FIGHTS C1 on straddling boundary cells (V_HJ(center)>=0
     there), pinning ub V >= 0 (vcert C1-leak frac 1.0 -> 1.0, unchanged).

SURPRISING POSITIVE (the real signal): every C3/C4 MARGIN got dramatically
TIGHTER -- C4-witness mean -0.704 -> -0.069, min -3.31 -> -0.176; C3 to 90.8%.
A well-shaped, internally-consistent V makes C3 AND C4 NEARLY certifiable (all
within ~0.07-0.19 of passing).  So the binding problem is keeping {V>=0} \subseteq
K (C1) at the SAME TIME as a smooth/consistent V -- the core CBF tension, now
sharp: C1 wants {V>=0} SMALL (inside K); C3/C4 tightness wants V smooth/high.

LESSONS / options (scope with user):
 * The decrease/band cannot be widened by naive cell-worst training -- it inflates
   V (this run) or roughens it (the v_dec_margin=0.25 backfire).  Drop the V-cert
   decrease push.
 * The C1 cell-worst floor is sound but needs (a) FULL coverage (train on ALL g<0
   boundary cells, not a 4096 sample), (b) NO anchor on unsafe cells (let C1 push
   them down), (c) to be the DOMINANT pressure.  Worth retrying C1-floor-ONLY.
 * Bigger picture: the inflated-V run shows C3/C4 are within reach if V is
   consistent; the open problem is a V whose {V>=0} hugs K (C1) while staying
   smooth.  Candidates: a tighter C1 floor coupled to the existing (un-inflated)
   teacher V; or verifier-side finer cells to absorb the residual ~0.07-0.19 C4
   slack on the un-inflated two-sided V0.12 (which had real C1=3762 and C4 within
   ~0.1 pointwise).

Files: `qcbf/nets/certified_train.py` (train_v_certified + helpers; sc bug fixed),
`qcbf/config.py` (cert_v_w/_c1_margin/_dec_margin/_anchor_w), `run_train.py`
(V-cert stage before Q), `config_pilot.json` (cert_v_w=2, v_dec_margin reverted to
0.12).  Artifact: `results/dubins_e0_pilot/` (this failed run).  Certificate
status: strict_spec_pass=false (C1 bad 30079).  cert_v_w left =2 in config pending
the user's redirect (revert to 0 to disable the stage).

## Update: 2026-06-14 C1-floor-ONLY V-cert -- C1 3762 -> 23 (milestone), C4 now the wall

Per user redirect: backed up the good two-sided V0.12 base to github (commit
10deb9b), then made V-cert C1-floor-ONLY: dropped the inflating decrease push
(new knob cert_v_dec_w, default 0; the successor forwards are skipped, so it is
fast), FULL coverage of the boundary band (vpool = ubV0 >= -0.2, ~38k cells, no
sampling -- every C1-bad cell is driven down), and the teacher anchor masked to
SAFE cells only (V_HJ(center) >= 0 on straddling cells would otherwise fight the
floor).  cert_v_w=5, c1_margin=0.10.

RESULT (vs two-sided V0.12), CROWN cell-worst:
  cond (pass% | mean | min)   two-sided V0.12       + C1-floor-only V-cert
  ----------------------------------------------------------------------
  C3 gate                     27.2% / -.181 / -.881  45.7% / -.031 / -.833
  C4 menu                     21.0% / -.341 / -2.05  10.9% / -.500 / -3.45
  C4 witness                   7.4% / -.704 / -3.31   1.4% / -.775 / -3.65
  C1 bad                      3762                   23
  C1 possible / active        32686 / 28924          26620 / 26597
  vcert C1-leak frac          --                     1.000 -> 0.067

MILESTONE: C1 bad 3762 -> 23 (the cleanest C1 yet; the inflation run was 30079).
The C1-floor lever WORKS as intended -- full coverage + safe-only anchor + no
decrease push sharpens {V>=0} strictly inside K without inflating V.  C3 also rose
(27 -> 46%).  C1 is essentially solved (23 stubborn straddling cells left; a
slightly stronger floor / margin should finish 23 -> 0).

NEW TENSION (the honest cost): the C1 floor lowered V at the boundary, which
dropped lb V(f) there and HURT C4 -- menu 21 -> 11%, witness 7.4 -> 1.4% (mean
-.704 -> -.775, min to -3.65).  So C4 is now unambiguously the binding wall, and
there is a real C1-vs-C4 tension: pushing {V>=0} inside K (C1) lowers the boundary
successor values C4 leans on.  The C4 cell-worst gap (mean ~-0.5/-0.78) is also
LARGER than the "~0.1 finer-cell slack" hoped for -- part slack, part the
C1-floor-induced V drop.

NOTE: even at C1=0 the strict cert would still FAIL on C4 (needs ALL active cells
to pass C3 AND C4), so finishing C1 23->0 alone yields no passing cert until C4 is
addressed.  Next (scope with user): finish C1->0; then C4 -- finer verifier cells
to shrink slack AND manage the C1-vs-C4 boundary tension (e.g. a softer C1 margin,
or a band-aware C1 floor that does not over-lower boundary successors).

Files: `qcbf/nets/certified_train.py` (C1-floor-only: dec gated on dec_w>0,
anchor masked to safe cells), `qcbf/config.py` (cert_v_dec_w), `run_train.py`
(full-coverage vpool when decrease off), `config_pilot.json` (cert_v_w=5,
cert_v_dec_w=0).  Artifact: `results/dubins_e0_pilot/`.  Certificate status:
strict_spec_pass=false (C1 bad 23; C4 witness 1.4% is the wall).

## Update: 2026-06-14 GROUND-TRUTH reference (V_HJ, Q_HJ=V_HJ o f) -- the spec hardness

Ran the SAME C1/C3/C4 + joint-certified metrics on the oracle ground truth to
separate "spec/verifier hardness" from "learned-model quality".

POINTWISE (run_oracle_spec.py, grid 124848 nodes / MC): C1 bad 0, C3 pass 72.2%
(fails 28% near the {V>=0} boundary where the deployed gamma=0.90 gate is tight,
min margin -0.135), C4 exact pass (Q_HJ=V_HJ o f).  JOINT certified = 38088/52752
active -> rho_pointwise = 0.722 of Omega*.

CELL-WORST (same verifier semantics as the learned diagnostic; V_HJ cell spread
over the 40^3 lattice via a 3x3x3 grid stencil): mean spread 0.481, p50 0.430,
p95 0.944, max 1.319.  Because Q_HJ ≡ V_HJ o f EXACTLY, the cell-worst C4 check
ub Q(cell) <= lb V(f)(cell) becomes ub(V o f) <= lb(V o f), failing by the spread:
C4 cell-worst margin = -(spread) ~ -0.48 < 0 STRUCTURALLY -> cell-worst C4 ~0%,
JOINT rho_cellworst ~ 0.

TWO STRUCTURAL FINDINGS (reframe the whole effort):
 (1) rho ceiling ~0.72 POINTWISE -- even a perfect value function only certifies
     72% of Omega*, capped by C3 at the {V>=0} boundary under deploy gamma=0.90.
     Not a learned artifact.
 (2) The IDEAL Q_HJ=V o f is UN-CERTIFIABLE cell-worst (rho~0): it has ZERO C4
     margin, so the cell slack (~0.48) sinks it.  The learned ONE-SIDED Q
     (Q <= V(f) - m) is therefore the RIGHT object, not a degraded one -- the
     below-V(f) gap is exactly what cell-worst C4 needs, which is why the C4-only
     lever hit 99% where the ideal Q gets 0%.  For this certificate learned-Q can
     beat ground-truth-Q.

So the binding walls -- the ~0.48 cell-worst slack and the deploy-gamma C3 boundary
(rho<=0.72) -- are INHERENT to the spec/verifier at this resolution; they bind the
teacher too.  The earlier "works on ground truth" intuition holds POINTWISE only.
Realistic target: rho ceiling 0.72; cell-worst rho>0 needs finer cells (shrink the
~0.48 slack) AND the learned Q's one-sided margin.  Numbers added to
docs/RESULTS_E0_round1.md (ground-truth section).

## Update: 2026-06-14 JOINT certified set -- which C4 (menu vs witness)?  -> WITNESS

User question: the joint used C4_menu AND C4_witness (intersection); should the two
C4s be UNION instead?  Resolution: the SOUND safety set is C1 ∧ C3 ∧ C4_WITNESS,
NOT the intersection-with-menu and NOT the pure union.  C3 guarantees the witness
u=pi(x) is always gate-feasible, so deploying the witness gives
min_d V(f) >= min_d Q(x,pi,d) >= gamma V >= 0 -- the witness is the universal safe
fallback.  C4_menu is only for minimum-intervention (menu actions are not feasible
everywhere), so the menu/witness UNION over-counts (menu-only cells are unsafe at
states with no feasible menu action) and the earlier intersection was over-strict.
Empirically MOOT: recomputed rho under intersection / union / witness-only /
menu-only -- ALL 0 on the learned runs, because the C3-passing and C4-passing cells
are essentially DISJOINT (C3-only rho 0.284 V0.12 / 0.433 C1-floor).  So the
binding problem is the cell-level C3 ⊥ C4 anti-correlation, not the C4 combination.
Fixed `run_learned_spec_diagnostic.py`: joint_certified_set is now C1∧C3∧C4_witness
(sound), with `joint_with_menu_frac` reported separately (the stricter as-is
gate-only filter).  Docs/RESULTS updated.

## Update: 2026-06-14 RECURRENCE reformulation (drop V/C4) -- BREAKS THE WALL, rho 0 -> 0.64

User proposal (with a precise correction): do NOT certify the discounted decrease
Q >= gamma V + C4; instead certify the EXACT sufficient condition for control-
invariance -- RECURRENCE.  Drop V entirely; define the closed-loop safety value
W(x) = min(g(x), net(x)) (Fisac/CBVF clamp), safe set {W>=0}, and verify ONE
condition: min_d W(f(x, pi(x), d)) >= m over {W>=0}.  gamma was only Fisac's
learning-stability conservatism, never required for safety; on the feasible set
Q>=gamma V already implies Q>=0, so this is the WEAKEST still-sufficient condition.
Three pinned points (all correct/necessary): (1) C1 moves to Q and is FREE via the
clamp W=min(g,net) => {W>=0}⊆{g>=0} by construction (also structurally blocks the
V-cert inflation failure -- there IS a clamp now); (2) keep margin W>=m, not W>0,
to clear cell slack; (3) runtime must become Q-only or cert != deploy.

FAST TEST (no retraining): reinterpret the EXISTING V_theta, pi_theta as
W = min(g, V_theta) and verify the single recurrence min_d W(f(x,pi,d)) >= m over
{W>=0}.  Sound: lb W(succ) = min(lb g(succ) [exact box], lb_CROWN V(succ)); witness
range = CROWN'd compiled pi; successor_boxes sound.

  artifact   OLD two-sided C3/C4 rho   NEW recurrence rho (m=.005)   recur pass%
  ----------------------------------------------------------------------------
  V0.12      0.000                     0.638                         58.7
  C1-floor   0.000                     0.537                         61.6

rho at m=0: 0.642 / 0.540; still 0.60 / 0.51 at m=0.05.  Pointwise ground-truth
ceiling was 0.722, so the CELL-WORST recurrence on V0.12 (0.64) is ~89% of the
pointwise ceiling -- the one-sided slack only costs ~0.08 of rho.

THE WALL IS BROKEN: rho 0 -> ~0.64 on existing nets, just by reformulating to the
single recurrence.  Why: (a) the C3 ⊥ C4 disjointness (our rho=0 cause) is GONE --
one one-sided lower-bound condition, no opposing pull; (b) CS free via the clamp
(no C1 floor needed -- and indeed V0.12 (no floor) BEATS C1-floor 0.64 vs 0.54,
because the C1 floor we built for the OLD cert lowers boundary V and hurts the
recurrence); (c) one-sided => ~half the slack budget.

CAVEAT (honest): rho=0.64 is the per-cell RECURRENCE PASS RATE.  A proven forward-
invariant set is the largest subset where the recurrence holds AND successors land
back inside it (the standard largest-controlled-invariant-set / GFP fixed point);
it will trim 0.64 somewhat (failing 41% are near the boundary) but stays >> 0.  Now
sound here because CS is structural (no c-sweep/hand-carve -- the clamp does it).

NEXT (the real build): (1) a proper recurrence-certificate stage = the GFP fixed
point on {W>=0} under the witness recurrence, exporting the invariant subset + rho,
and the Q-only runtime (gate on Q>=m); (2) retrain a CLAMPED Q=min(g,net) with the
one-sided recurrence objective (build margin, like the C4-only lever's 99%) to push
0.64 toward the 0.72 ceiling.  This dissolves the entire C3/C4 + band-vs-slack +
C1-vs-C4 wall stack documented above.

GROUND-TRUTH under the recurrence standard (grid-table bounds for V_HJ / pi_HJ):
  metric                        OLD cert (gamma-C3)   recurrence
  ----------------------------------------------------------------
  pointwise ground-truth rho    0.722                 0.964
  cell-worst ground-truth rho   0 (Q=Vof, C4 fails)   0.577
  cell-worst learned V0.12 rho  0                     0.642
POINTWISE 0.72 -> 0.96: by the HJ fixed point the recurrence margin IS q_robust =
max_u min_d V_HJ(f) >= V_HJ >= 0 on Omega*, so it holds on 96.4% of Omega* (mean
margin +0.52).  The old gamma=0.90 C3 gate q_robust >= gamma V + eps is what capped
it at 0.72; recurrence (no gamma) removes the cap (the CORRECTED ceiling is ~0.96).
CELL-WORST SURPRISE: ground truth (0.58) is BELOW learned (0.64) because pi_HJ is
BANG-BANG (adjacent-cell control jump ~0.385 of [-1,1]) -> wide per-cell action
range -> wide successor boxes -> recurrence loses cell-worst near switching
surfaces (plus grid bounds looser than CROWN).  The SMOOTH learned pi_theta + CROWN
beats the jagged grid teacher -- same theme as "learned Q beats ground-truth Q for
C4".  Takeaway: recurrence ceiling ~0.96 (huge headroom vs 0.72); learned already
0.64 cell-worst; the gap is boundary cell slack -> finer cells + retrain clamped Q
for recurrence margin (a smooth high-margin policy, unlike bang-bang pi_HJ).

## CORRECTION (recurrence certificate): GFP was the WRONG tool; honest numbers

Retraction of the "NEXT (the real build) (1) GFP" plan above and the rho=0.64
framing.  Theorem A's recurrence condition is a VALUE condition on a barrier
function W = min(g, V_theta):  for all x in S_m={W>=m}, lb_W(f(x,pi(x),d)) >= m.
Because S_m is the m-superlevel set of the SAME W, "W(successor) >= m" IS
"successor in S_m" -- so proving it on ALL active cells gives forward invariance
DIRECTLY.  No fixed point.  A viability-kernel GFP is for the opposite case (no
barrier function, maximal invariant set computed geometrically); it discards W at
the successor, rounds the successor box up to whole lattice cells, and iterates
erosion -- strictly looser.  Built and ran it anyway (deleted): the geometric GFP
collapses to EMPTY (0 cells) at both 40^3 and 80^3 in ~10 sweeps, because the
fixed-speed car moves exactly one cell-width per step (k=dt*v=0.1=hx) so every
cell's cell-worst successor straddles 2-3 cells.  That is an artifact of the wrong
method, not a real result.  Tool: experiments/dubins_e0/run_recurrence_cert.py
(the correct check; the gfp script is removed).

HONEST NUMBERS (correct Theorem-A recurrence on the frozen V0.12, 40^3, full
active set, witness u=clip(pi), W=min(g,V), d-subsplit 2):
  m       active   pass%   inner rho({lb_W>=m})   complete?
  0.00    32515    59.2    0.7415                 NO (13270 fail)
  0.10    29773    54.4    0.6294                 NO
  0.20    26618    50.4    0.5205                 NO
  0.30    23300    46.9    0.4322                 NO
=> The prior "rho=0.64" was the per-cell PASS RATE (here 0.592 exact), NOT a
certified volume.  rho_inner({W>=0})=0.74 is the SIZE of {W>=0} but only 59% of it
closes under the witness, so NEITHER is a complete certificate.  Raising m makes
pass% WORSE (the recurrence threshold rises faster than the active set shrinks), so
NO level m closes {W>=m}.  rho_certified = 0 for the frozen V0.12.

FAILURE DIAGNOSIS at m=0 (13270 fails): inner 2793 (cells fully in S_0 whose
successor leaks below 0 -- not a granularity artifact) + straddle 10477; succ_lbW
on fails min -1.66, median -0.23; only 600 verifier near-misses (>= -0.02), 7400
DEEP leaks (< -0.2).  => the failures are GENUINE policy leaks, not verifier slack:
finer cells recovers <= ~600, level-raising is worse, GFP is wrong.  Root cause:
V_theta was trained for the gamma=0.90-DISCOUNTED Q-CBF decrease (V(f) >= gamma V),
which explicitly PERMITS V dropping toward 0 at the boundary; strict recurrence
W(f) >= m forbids exactly that.  Discounted-decrease training and undiscounted
set-recurrence are incompatible at the boundary -- this is a MODEL problem.

NEXT (route 1, per Theorem A -- prove recurrence on the WHOLE {W>=m}): retrain a
clamped W = min(g, V_theta') with the verifier-in-the-loop recurrence objective
lb_W(f(C,[u_lo,u_hi],D)) >= m on all active cells (the allowed lever; same CROWN-
IBP machinery as the C4-only run that hit 99%), pushing pass% -> 100% so the whole
{W>=m} closes, then report rho = Vol({W>=m})/Vol(Omega*).  NOT GFP, NOT verifier-
only, NOT level-raising.  Ceiling reference: pointwise ground-truth recurrence 0.96.

## T4 shrink-refinement (Sec. 5.9) implemented faithfully — rho=0 confirmed THREE ways

Correction to the "GFP is the wrong tool" framing: per theory_core.md Sec. 5.9 / Prop. T4,
a monotone shrink-refinement T_Phi(S)=S cap K cap Pre^all_Phi(S), iterated from
S_0={W>=m} cap K to its greatest fixed point, IS the framework's own way to certify a
forward-invariant SUBSET when the full superlevel set does not close.  So a fixed point
is NOT a wrong move; what was wrong earlier was (a) framing it as REPLACING Theorem A,
and (b) the pure-geometric viability kernel that ROUNDS the successor to whole cells and
DISCARDS the W-value.  The realized T4 (Sec. 5.9 "sound inner Pre via 5.7") uses the
VALUE bound for superlevel membership: a cell C survives iff
  (value, TIGHT)     lb_W(succ_C) >= m          [succ subset {W>=m}, no cell-rounding]
  AND (geometric)    reach(C) subset keep        [succ avoids removed cells]
one-control witness predecessor (u=clip(pi), Sec. 5.9 allows the one-control form).
Implemented in run_recurrence_cert.py --t4 (reach via 3-D summed-area table).

RESULT on frozen V0.12 @40^3: the T4 greatest fixed point is EMPTY at EVERY level
  m = 0.0 / 0.1 / 0.2 / 0.3 / 0.5  ->  0 cells (erodes in 5-8 sweeps).
No deep-interior viability core survives.  So rho_certified=0 is now confirmed THREE
independent sound ways: (1) full superlevel recurrence pass 59.2%<100% at m=0; (2) no
level m closes {W>=m} (pass falls with m); (3) T4 greatest fixed point empty at all m.
The realized T4 reach is a sound SUPERSET (full-box, no subsplit) so it is slightly
conservative, but that is not the cause: the VALUE gate alone caps at 59% (m=0) / 36%
(m=0.5), and the deep genuine leaks (median -0.23) percolate inward.  T4 done correctly
CONFIRMS the empty result rather than rescuing it -- there is genuinely no robustly
forward-invariant subset of the frozen V0.12's witness on this lattice.

CONCLUSION (unified): the wall is the MODEL (gamma=0.90-discounted training leaks under
undiscounted recurrence), not GFP/T4, not verifier slack, not the level.  T4 is the
correct framework operator and it will certify a non-empty subset ONLY AFTER route 1:
retrain a clamped W=min(g,V') with the verifier-in-loop recurrence objective so the
deployed witness stops leaking (push value-pass toward 100% / leave a non-empty core),
THEN T4/superlevel reports rho=Vol(S)/Vol(Omega*).  Pointwise GT recurrence ceiling 0.96
(verified: 0.9636 on Omega* grid nodes; that is a POINTWISE pass-fraction of Omega*, not
a sound cell-worst volume -- the apples-to-apples sound GT number is the cell-worst ~0.58).

## Route 1: clamped recurrence-barrier training of W_tilde -- model wall BROKEN

Built train_w_recurrence (certified_train.py) + run_w_recurrence.py on branch
`clamped-w-recurrence`.  Certificate object = CLAMPED RECURRENCE BARRIER
W_theta=min(g,W_tilde), S_m={W_theta>=m}; witness u=clip(pi) FROZEN; runtime==cert
(no Q>=gamma*V gate).  C1 structural via the min(g,.) clamp.  Training = BOUNDED
directional hinge on the verifier's own IBP quantity (UP: raise lb_IBP W_tilde(f)
to m where the successor is g-feasible; DOWN: exclude physical g-leaks), eps-
scheduled point->cell.  No discount, no V-floor.

IMPLEMENTATION NOTE (real, logged): the value-regression Fisac backup
W_tilde(C)<-min(g, lb_IBP W_tilde(f)) DIVERGES -- IBP lb is far looser than the
CROWN verifier (~11 vs ~0.4 at eps=1) so the cell-worst target races to -inf and
W_tilde collapses (active->0, mse 1.8->1.4e4).  The BOUNDED hinge (push only TO m)
is the fix.  Two-sided UP+DOWN also destabilizes (DOWN on the g-leak shell + net
smoothness cascades the active set down past the kernel).  UP-ONLY is clean and
stable.

RESULT (UP-only, V0.12 warm start, 36 ep, eps_warmup 0.45, @40^3 cert):
  CROWN cell-worst recurrence pass  59.0% -> 82.9%  at m=0  (huge, the model wall);
  wleak (W_tilde-fixable leaks) 34088 -> ~0;  gleak fixed at 7032 (=17%).
So 82.9% = (active-gleak)/active EXACTLY: W_tilde now satisfies the value
recurrence on EVERY g-feasible cell; the ONLY residual fails are the g-leaks.
This is rule 4 working: the failure split cleanly separates the W_tilde-fixable
part (DONE) from the physical g-leak (the frozen witness drives the {g>=0}-
boundary shell out of {g>=0}) -- which W_tilde provably cannot fix (g caps it),
so it routes to a pi_theta retrain, NOT more W_tilde push.

REMAINING WALL (cleanly isolated to two ALLOWED levers):
  (a) g-leak 17% -> pi_theta retrain (the witness leaks the boundary shell);
  (b) T4 still empties at 40^3 even on this good W_tilde -- pure GEOMETRIC cell-
      rounding erosion (forced motion 0.1=1 cell/step + 0.3 disturbance fan),
      a verifier-RESOLUTION wall independent of the model.  [80^3 test pending.]
NB inner-rho({W_tilde>=0})=1.21>1 because UP-only pushed W_tilde up to ~g, so
{W_tilde>=0} ~ {g>=0} which is LARGER than Omega* (the boundary shell is included
and is exactly what leaks).  A proper carve (W_tilde -> Fisac value, low on the
non-viable shell) would set {W>=0}~Omega*; the stable carve is the open item
(the value backup that does it diverges under IBP -> needs differentiable CROWN
or an alternating UP-train / T4-shrink scheme).

## Route-1 follow-up: 80^3 resolution test -- value-pass up, T4 still empty

Ran run_recurrence_cert --cells 80 --t4 on the trained W_tilde (318400 candidate
cells, ~10 min).  CROWN cell-worst recurrence pass 82.9% (@40^3) -> 90.0% (@80^3)
at m=0 -- finer cells tighten the verifier slack and lift the VALUE pass (a real,
sound gain).  But the T4 greatest fixed point is STILL EMPTY (0 cells, 16 sweeps).

So the geometric T4 erosion is RESOLUTION-INDEPENDENT and MODEL-INDEPENDENT: empty
at 40^3 and 80^3, on frozen V0.12 AND on the trained W_tilde.  Finer cells help
the value-recurrence pass-rate but NOT the geometric set-invariance, because the
binding problem is that the frozen witness physically leaks the {g>=0}-boundary
shell out, and that leak percolates geometrically inward to empty regardless of
cell size or W_tilde quality.

DEFINITIVE LOCALIZATION.  Two distinct objects, two distinct walls:
  * VALUE recurrence (the model object): W_tilde training + finer cells took it
    59% -> 82.9% -> 90%.  Essentially SOLVED as a lever; ceiling = (1 - g-leak
    fraction).  This is the route-1 win.
  * The CERTIFIED SET still needs {W>=0} ~ Omega* (currently {W_tilde>=0} ~
    {g>=0}, inner-rho 1.21-1.29 > 1 -- W_tilde OVER-claims the boundary shell that
    leaks).  Neither T4 (geometric, empties) nor finer cells fix this; it needs
    the CARVE -- W_tilde lowered to the Fisac value on the non-viable shell, after
    which {W_tilde>=0} ~ Omega* and the VALUE recurrence passes ~100% on it with
    NO T4 (fixed-point identity).  Carve is the open item (IBP backup diverges).
  * Plus the residual physical g-leak -> pi_theta retrain (rule 4).

CONCLUSION: route 1 broke the MODEL wall; the remaining gap is the value-route
CARVE (make {W>=0}=Omega*) and/or a pi_theta retrain -- NOT T4, NOT finer cells.
