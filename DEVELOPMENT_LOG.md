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
