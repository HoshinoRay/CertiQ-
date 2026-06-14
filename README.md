# Certified Robust Q-CBF: Dubins E0

This repository is the Dubins-only certification codebase for a learned robust
state-action Q-CBF filter.  The active deployed artifact is

```text
(V_theta, Q_theta, pi_theta)
```

and every safety claim must be proved post hoc against the true Dubins dynamics
and the full disturbance set.

## Clean Flow

1. Define the fixed-speed Dubins car and obstacle/workspace safety margin.
2. Compute the grid teacher \(V_{\mathrm{HJ}}\) with the **discounted safety
   backup** \(V\leftarrow(1-\lambda)g+\lambda\min(g,\max_u\min_d V(f))\), teacher
   discount \(\lambda=\gamma_{\mathrm{teach}}\approx0.92\).  The discount makes the
   backup a contraction and keeps \(\Omega^\star=\{V_{\mathrm{HJ}}\ge0\}\)
   non-empty and resolution-robust; it is a teacher knob only and never enters
   the certificate.
3. Define \(Q_{\mathrm{HJ}}(x,u,d)=V_{\mathrm{HJ}}(f(x,u,d))\).  \(V\) and \(Q\)
   are learned by **separate** networks, so no \(V=\max_u\min_d Q\) identity is
   imposed; the runtime gate \(\min_d Q_\theta\ge\gamma_{\mathrm{deploy}}V_\theta\)
   is exactly the robust one-step decrease at the deployed decay
   \(\gamma_{\mathrm{deploy}}\approx0.90\).
4. Train \(V_\theta,Q_\theta,\pi_\theta\) by supervised distillation from
   \(V_{\mathrm{HJ}},Q_{\mathrm{HJ}}\), and the robust-greedy witness labels.
5. Certify the frozen learned artifact with \(C1,C3,C4\) at
   \(\gamma_{\mathrm{deploy}}\).  Training loss and oracle accuracy are
   diagnostics, not proof assumptions.

## Certification Target (Theorem A)

- \(C1:\ \{x:V_\theta(x)\ge0\}\subseteq\mathcal K\).
- \(C3:\ \min_d Q_\theta(x,\pi_\theta(x),d)\ge\gamma_{\mathrm{deploy}}V_\theta(x)+\varepsilon\),
  using the deployed witness policy (recursive feasibility).
- \(C4:\ \min_d Q_\theta(x,u,d)\le\min_d V_\theta(f(x,u,d))\) for the runtime
  menu actions and the witness action, so the \(Q_\theta\) gate does not
  overclaim (gate \(\Rightarrow\) decrease).

The robust one-step decrease \(V_\theta(f)\ge\gamma_{\mathrm{deploy}}V_\theta\)
is **not** a separate obligation: any applied action passes its own gate
\(\min_d Q_\theta\ge\gamma_{\mathrm{deploy}}V_\theta\), and \(C4\) turns that into
\(\min_d V_\theta(f)\ge\gamma_{\mathrm{deploy}}V_\theta\ge0\).  Together with
\(C1\) this gives robust forward invariance of \(\{V_\theta\ge0\}\) and safety.

The verifier is intentionally conservative: unknown cells are not certified.

## Active Components

```text
qcbf/
  config.py              Shared experiment configuration.
  dynamics/dubins.py     Dubins dynamics, safety margin, interval successors.
  oracle/value_iteration.py
                         Gamma-consistent grid CBVF/HJ truth.
  nets/mlp.py            NumPy ReLU MLPs and supervised training utilities.
  verify/                IBP/CROWN bounds and compiled predicates.
  certify/               Cell lattice, strict C1-C4 verifier, volume utility.
  runtime/filter.py      Runtime Q-CBF filter over the certified action menu.
  audit/falsify.py       Rollout audit of the certified closed loop.
```

## Active Experiment

```text
experiments/dubins_e0/
  run_oracle.py
  run_train.py
  run_certify.py
  run_audit.py
  run_all.py
```

## Test Surface

```bash
python -m tests.test_soundness
```

Only the Dubins \(E0\) pipeline is active in this tree.
