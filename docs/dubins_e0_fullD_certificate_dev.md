# Dubins E0 Full-$\mathcal D$ Certificate Experiment Development Guide

## 0. Purpose

This document defines the minimum paper-facing Dubins experiment for the v9 project frame:

$$
\text{sound post-hoc certification of the deployed learned robust state-action }Q\text{-CBF filter}.
$$

The experiment is deliberately not a Theorem S / learned-adversary experiment. It bypasses the deployed learned-adversary shortcut and directly verifies the full disturbance set:

$$
\Phi_{\mathrm{rob}}(x,u)
\Longleftrightarrow
\min_{d\in\mathcal D}Q_\theta(x,u,d)
\ge
\beta(V_\theta(x)).
$$

The target outcome is a non-vacuous certified set for the frozen learned artifact

$$
(V_\theta,Q_\theta,\pi^\flat_\varphi),
$$

with no approximation assumption such as

$$
Q_\theta\approx Q^\star,
\qquad
V_\theta\approx V^\star.
$$

Training quality is used only to make the certificate non-vacuous. Soundness comes only from verified inequalities.

---

## 1. What This Experiment Answers

### 1.1 Go / No-Go Question

The first question is:

$$
\exists\ \Omega_c^{\mathrm{cert}}\neq\varnothing
\quad\text{such that}\quad
\Omega_c^{\mathrm{cert}}
\text{ is robustly invariant under full }
\mathcal D?
$$

If the answer is no on Dubins, the project has a central non-vacuity problem.

If the answer is yes, this experiment can become the main paper figure:

$$
\frac{
\operatorname{Vol}(\Omega_c^{\mathrm{cert}})
}{
\operatorname{Vol}(\Omega^\star)
}
>0,
\qquad
\#\{\text{certified-but-violated under full }\mathcal D\}=0.
$$

### 1.2 What This Experiment Does Not Need

Do not build these first:

$$
d_\psi,
\qquad
\text{adversarial RL},
\qquad
\text{conformal calibration},
\qquad
\text{hardware},
\qquad
\text{Theorem S sweep}.
$$

Those are story / extension experiments. The core E0 result is the deterministic full-$\mathcal D$ certificate.

---

## 2. Dubins Model

### 2.1 State, Control, Disturbance

Use the fixed-speed Dubins car:

$$
x=(p_x,p_y,\psi)\in\mathcal X\subset\mathbb R^2\times[-\pi,\pi),
$$

where \(p=(p_x,p_y)\) is position and \(\psi\) is heading.

The control is angular velocity:

$$
u\in\mathcal U=[-\omega_{\max},\omega_{\max}].
$$

The disturbance is an additive angular-velocity disturbance:

$$
d\in\mathcal D=[-d_{\max},d_{\max}].
$$

The discrete-time dynamics are

$$
p_x^+
=
p_x+\Delta t\,v\cos\psi,
$$

$$
p_y^+
=
p_y+\Delta t\,v\sin\psi,
$$

$$
\psi^+
=
\operatorname{wrap}_{[-\pi,\pi)}
\left(
\psi+\Delta t\,(u+d)
\right).
$$

Default pilot parameters:

| Symbol | Value | Note |
|---|---:|---|
| \(\Delta t\) | \(0.05\) or \(0.10\) | start with \(0.10\) for speed |
| \(v\) | \(1.0\) | fixed forward speed |
| \(\omega_{\max}\) | \(1.0\) | control authority |
| \(d_{\max}\) | \(0.25\) to \(0.40\) | robust disturbance |
| \(p_x,p_y\) domain | \([-2,2]^2\) | bounded workspace |
| \(\psi\) domain | \([-\pi,\pi)\) | periodic |

The disturbance should be large enough to materially change safety:

$$
0<d_{\max}<\omega_{\max}.
$$

If \(d_{\max}\) is too small, everything passes but the robust axis is weak. If \(d_{\max}\) is too large, the robust safe set may become tiny.

---

## 3. Safety Specification

### 3.1 Primary Scenario: Obstacle Inside a Bounded Workspace

Use one circular obstacle and one circular workspace boundary.

Obstacle:

$$
\mathcal O
=
\{p:\|p-o\|_2^2\le r_{\mathrm{obs}}^2\}.
$$

Workspace:

$$
\mathcal W
=
\{p:\|p\|_2^2\le R_{\mathrm{world}}^2\}.
$$

The safe set is

$$
\mathcal K
=
\left\{
x:
\|p-o\|_2^2-r_{\mathrm{obs}}^2\ge0,
\quad
R_{\mathrm{world}}^2-\|p\|_2^2\ge0
\right\}.
$$

Use the scalar margin

$$
g(x)
=
\min\left\{
\|p-o\|_2^2-r_{\mathrm{obs}}^2,
\;
R_{\mathrm{world}}^2-\|p\|_2^2
\right\}.
$$

Default values:

$$
o=(0,0),
\qquad
r_{\mathrm{obs}}=0.45,
\qquad
R_{\mathrm{world}}=1.8.
$$

This scenario has real geometry: the vehicle cannot move sideways, cannot stop, and must use heading to avoid both the obstacle and the boundary.

### 3.2 Why Dubins Is Meaningful

Unlike a \(1\mathrm D\) double integrator, Dubins has a nonholonomic safety structure. Whether a state is safe depends jointly on

$$
p_x,
\quad
p_y,
\quad
\psi.
$$

The same position can be safe or unsafe depending on the heading:

$$
V^\star(p_x,p_y,\psi_1)\neq V^\star(p_x,p_y,\psi_2).
$$

The worst-case steering disturbance can also switch across regions:

$$
d^\star(x,u)\in\{-d_{\max},d_{\max}\}.
$$

That switching structure is what later makes Theorem S natural. But E0 does not need to exploit it; E0 directly verifies the full set \(\mathcal D\).

---

## 4. Oracle Construction

### 4.1 Bellman-Isaacs Safety Value

Compute a grid oracle \(V^\star\) by value iteration:

$$
V^{(0)}(x)=g(x),
$$

$$
V^{(k+1)}(x)
=
\min\left\{
g(x),
\max_{u\in\mathcal U_{\mathrm{grid}}}
\min_{d\in\mathcal D_{\mathrm{grid}}}
V^{(k)}(f(x,u,d))
\right\}.
$$

Terminate when

$$
\|V^{(k+1)}-V^{(k)}\|_\infty\le \tau_{\mathrm{VI}},
$$

or when the horizon budget is reached.

Recommended pilot grid:

$$
N_{p_x}\times N_{p_y}\times N_\psi
=
51\times51\times41.
$$

Recommended paper grid:

$$
N_{p_x}\times N_{p_y}\times N_\psi
=
81\times81\times61.
$$

Recommended action and disturbance grids:

$$
|\mathcal U_{\mathrm{grid}}|=21,
\qquad
|\mathcal D_{\mathrm{grid}}|=9
\quad\text{or}\quad
11.
$$

If \(f(x,u,d)\) leaves the workspace grid, assign a failure value:

$$
V^{(k)}(f(x,u,d)) := g_{\mathrm{fail}}<0.
$$

Do not silently clamp out-of-domain successors to the boundary.

### 4.2 State-Action Oracle

After convergence, define the oracle state-action value:

$$
Q^\star(x,u,d)
=
V^\star(f(x,u,d)).
$$

Use interpolation of \(V^\star\) at off-grid successors.

### 4.3 Oracle Fallback Witness

Define an oracle robust fallback selector:

$$
u^\flat(x)
\in
\arg\max_{u\in\mathcal U_{\mathrm{grid}}}
\min_{d\in\mathcal D_{\mathrm{grid}}}
Q^\star(x,u,d).
$$

This selector is used only as a supervised label for \(\pi^\flat_\varphi\). It is not a proof object after training.

---

## 5. Network Training

### 5.1 Networks

Use small verifier-friendly networks first.

Recommended first architecture:

$$
V_\theta:\mathbb R^3\to\mathbb R,
\qquad
\text{MLP}(3,128,128,1),
$$

$$
Q_\theta:\mathbb R^5\to\mathbb R,
\qquad
\text{MLP}(5,256,256,1),
$$

$$
\pi^\flat_\varphi:\mathbb R^3\to[-\omega_{\max},\omega_{\max}],
\qquad
\omega_{\max}\tanh(\text{MLP}(3,128,128,1)).
$$

Use ReLU or another activation supported by the chosen verifier. If using \(\tanh\) or SiLU, confirm that auto-LiRPA / CROWN can produce sound bounds for that activation before committing.

### 5.2 Supervised Losses

Train \(V_\theta\) against \(V^\star\):

$$
\mathcal L_V
=
\mathbb E_x
\left[
\left(
V_\theta(x)-V^\star(x)
\right)^2
\right].
$$

Train \(Q_\theta\) against \(Q^\star\):

$$
\mathcal L_Q
=
\mathbb E_{x,u,d}
\left[
\left(
Q_\theta(x,u,d)-Q^\star(x,u,d)
\right)^2
\right].
$$

Train \(\pi^\flat_\varphi\) against the oracle selector:

$$
\mathcal L_\pi
=
\mathbb E_x
\left[
\left(
\pi^\flat_\varphi(x)-u^\flat(x)
\right)^2
\right].
$$

Add a witness-margin loss:

$$
m_\theta(x)
=
\min_{d\in\mathcal D_{\mathrm{train}}}
Q_\theta(x,\pi^\flat_\varphi(x),d)
-
\beta(V_\theta(x)),
$$

$$
\mathcal L_{\mathrm{margin}}
=
\mathbb E_x
\left[
\left[
m_{\mathrm{target}}-m_\theta(x)
\right]_+
\right].
$$

Use

$$
\beta(r)=\gamma r,
\qquad
\gamma\in\{0.3,0.5,0.7,0.9\}.
$$

The total training loss can be

$$
\mathcal L
=
\lambda_V\mathcal L_V
+
\lambda_Q\mathcal L_Q
+
\lambda_\pi\mathcal L_\pi
+
\lambda_m\mathcal L_{\mathrm{margin}}.
$$

Training is not part of the proof. It is only a way to obtain a frozen artifact likely to pass verification.

### 5.3 Training Diagnostics

Before CROWN, report:

$$
\operatorname{MSE}(V_\theta,V^\star),
\qquad
\operatorname{MSE}(Q_\theta,Q^\star),
$$

and the witness margin distribution:

$$
m_\theta(x)
=
\min_{d\in\mathcal D_{\mathrm{dense}}}
Q_\theta(x,\pi^\flat_\varphi(x),d)
-
\beta(V_\theta(x)).
$$

Also report the oracle selector agreement:

$$
\Pr_x
\left[
\left|
\pi^\flat_\varphi(x)-u^\flat(x)
\right|
\le
\delta_u
\right].
$$

These diagnostics are go / no-go aids. They are not certificate assumptions.

---

## 6. Runtime Filter

The deployed robust predicate is

$$
\Phi_{\mathrm{rob}}(x,u)
\Longleftrightarrow
\min_{d\in\mathcal D}
Q_\theta(x,u,d)
\ge
\beta(V_\theta(x)).
$$

The runtime safety filter selects

$$
u_{\mathrm{safe}}(x)
\in
\arg\min_{u\in\mathcal U}
\|u-u_{\mathrm{task}}(x)\|_2^2
\quad
\text{s.t.}
\quad
\Phi_{\mathrm{rob}}(x,u).
$$

The witness \(\pi^\flat_\varphi\) is used to prove recursive feasibility:

$$
\forall x\in \Omega_c^{\mathrm{cert}},
\qquad
\pi^\flat_\varphi(x)
\text{ is feasible under full }
\mathcal D.
$$

---

## 7. Certified Set Definition

### 7.1 Superlevel Candidate

For a threshold \(c\), define

$$
\Omega_c
=
\{x:V_\theta(x)\ge c\}.
$$

Increasing \(c\) shrinks the candidate set:

$$
c_1<c_2
\Longrightarrow
\Omega_{c_2}\subseteq\Omega_{c_1}.
$$

The \(c\)-search is a primary experimental knob.

### 7.2 Practical Cell-Set Certificate

In implementation, use a partition of the state domain into boxes:

$$
\mathcal P_X=\{X_i\}.
$$

The reported certified set is a union of accepted boxes:

$$
\Omega_c^{\mathrm{cert}}
=
\bigcup_{i\in I_{\mathrm{cert}}}X_i.
$$

For sound induction, use one of two modes.

**Mode A: full-superlevel certificate.** Prove that every point in \(\Omega_c\) is covered by verified boxes. Boundary cells that may intersect \(\Omega_c\) must be split until they are either verified or proven outside:

$$
\sup_{x\in X_i} V_\theta(x)<c.
$$

**Mode B: cell-fixed-point certificate.** Treat \(\Omega_c^{\mathrm{cert}}\) itself as the invariant set. Verify:

$$
\forall x\in\Omega_c^{\mathrm{cert}}:\ g(x)\ge0,
$$

$$
\forall x\in\Omega_c^{\mathrm{cert}},
\ \forall d\in\mathcal D:
Q_\theta(x,\pi^\flat_\varphi(x),d)
\ge
\beta(V_\theta(x))+\varepsilon,
$$

and

$$
\forall x\in\Omega_c^{\mathrm{cert}},
\ \forall u\text{ with }\Phi_{\mathrm{rob}}(x,u),
\ \forall d\in\mathcal D:
f(x,u,d)\in\Omega_c^{\mathrm{cert}}.
$$

Mode B is usually easier to implement first. It is the cell-lattice version of monotone shrink-refinement:

$$
S_{k+1}
=
S_k
\cap
\mathcal K
\cap
\mathrm{Feas}_{\pi^\flat}
\cap
\mathrm{Pre}^{\mathrm{all}}_{\Phi_{\mathrm{rob}}}(S_k).
$$

Unknown cells are removed, never counted as safe.

---

## 8. Verification Conditions

### 8.1 Condition \((\mathrm{C1})\): Safety of the Certified Set

For every accepted state box \(X_i\), prove

$$
\forall x\in X_i:
\quad
g(x)\ge0.
$$

For the obstacle-workspace safety function, this is equivalent to proving both:

$$
\forall x\in X_i:
\quad
\|p-o\|_2^2-r_{\mathrm{obs}}^2\ge0,
$$

$$
\forall x\in X_i:
\quad
R_{\mathrm{world}}^2-\|p\|_2^2\ge0.
$$

This can usually be discharged by interval arithmetic. CROWN is not necessary unless \(g\) is represented by a network.

### 8.2 Condition \((\mathrm{C3})\): Witness Feasibility

For every accepted state box \(X_i\), verify:

$$
\forall x\in X_i,
\ \forall d\in\mathcal D:
\quad
h_3(x,d)\ge0,
$$

where

$$
h_3(x,d)
=
Q_\theta(x,\pi^\flat_\varphi(x),d)
-
\beta(V_\theta(x))
-
\varepsilon.
$$

This is the first CROWN target. The input box is

$$
X_i\times\mathcal D.
$$

The verifier must lower-bound \(h_3\):

$$
\underline h_3(X_i,\mathcal D)
\le
\inf_{(x,d)\in X_i\times\mathcal D}h_3(x,d).
$$

The certificate passes if

$$
\underline h_3(X_i,\mathcal D)\ge0.
$$

If this fails, try:

$$
c\uparrow,
\qquad
\gamma\downarrow,
\qquad
\varepsilon\downarrow,
\qquad
\text{or improve }\pi^\flat_\varphi.
$$

### 8.3 Condition \((\mathrm{C2})\): Robust Transition

For every accepted state box \(X_i\), every control box \(U_j\), and all disturbances:

$$
\forall x\in X_i,
\ \forall u\in U_j
\text{ with }\Phi_{\mathrm{rob}}(x,u),
\ \forall d\in\mathcal D:
\quad
f(x,u,d)\in\Omega_c^{\mathrm{cert}}.
$$

Equivalently, in the superlevel mode:

$$
V_\theta(f(x,u,d))\ge c+\varepsilon.
$$

The antecedent \(\Phi_{\mathrm{rob}}\) makes this the hard part.

### 8.4 Conservative Antecedent Handling

Define the antecedent margin:

$$
a(x,u)
=
\min_{\delta\in\mathcal D}
Q_\theta(x,u,\delta)
-
\beta(V_\theta(x)).
$$

The action is robust-feasible when

$$
a(x,u)\ge0.
$$

For a control box \(U_j\), first try to prove it is infeasible:

$$
\forall x\in X_i,\ \forall u\in U_j:
\quad
a(x,u)<0.
$$

A sound sufficient test is to find one disturbance candidate \(\delta_0\in\mathcal D\) such that

$$
\forall x\in X_i,\ \forall u\in U_j:
\quad
Q_\theta(x,u,\delta_0)
-
\beta(V_\theta(x))
<0.
$$

Because

$$
\min_{\delta\in\mathcal D}Q_\theta(x,u,\delta)
\le
Q_\theta(x,u,\delta_0),
$$

this proves \(U_j\) contains no feasible actions for the current box.

If infeasibility cannot be proved, treat the entire \(U_j\) as potentially feasible and verify the transition condition for all

$$
(x,u,d)\in X_i\times U_j\times\mathcal D.
$$

This is conservative but sound.

### 8.5 Two Ways to Verify the Consequent

**Route 1: direct composition.** Verify

$$
h_2(x,u,d)
=
V_\theta(f(x,u,d))-c-\varepsilon
\ge0
$$

over

$$
X_i\times U_j\times\mathcal D.
$$

This is tighter but requires the verifier to handle the dynamics composition, including \(\sin\psi\), \(\cos\psi\), and angle wrapping.

**Route 2: successor-box over-approximation.** Compute a conservative successor box:

$$
X^+_{i,j}
\supseteq
\{f(x,u,d):x\in X_i,\ u\in U_j,\ d\in\mathcal D\}.
$$

Then verify either

$$
\forall x^+\in X^+_{i,j}:
\quad
V_\theta(x^+)\ge c+\varepsilon,
$$

or prove

$$
X^+_{i,j}\subseteq\Omega_c^{\mathrm{cert}}.
$$

Route 2 is recommended first because it separates known dynamics interval bounding from neural-network verification.

For a small heading box \(\Psi=[\psi_L,\psi_U]\), interval-bound:

$$
\cos\Psi\in[\underline c,\overline c],
\qquad
\sin\Psi\in[\underline s,\overline s].
$$

Then

$$
P_x^+
\subseteq
P_x+\Delta t\,v[\underline c,\overline c],
$$

$$
P_y^+
\subseteq
P_y+\Delta t\,v[\underline s,\overline s],
$$

$$
\Psi^+
\subseteq
\Psi+\Delta t\,(U_j+\mathcal D).
$$

If \(\Psi^+\) crosses the periodic boundary, split it into separate boxes. Do not merge a wrapped interval into a large interval covering almost all headings.

---

## 9. Cell Refinement Algorithm

### 9.1 Inputs

Inputs:

$$
V_\theta,\quad Q_\theta,\quad \pi^\flat_\varphi,\quad f,\quad g,\quad \mathcal D,\quad \mathcal U,\quad c,\quad \gamma,\quad \varepsilon.
$$

Partitions:

$$
\mathcal P_X=\{X_i\},
\qquad
\mathcal P_U=\{U_j\}.
$$

### 9.2 Initialization

Start with candidate cells that pass a loose value threshold:

$$
\inf_{x\in X_i}V_\theta(x)\ge c.
$$

If using Mode B, this is only an initialization. The final invariant set is obtained after refinement.

### 9.3 Iterative Shrink-Refinement

Repeat:

1. Remove \(X_i\) if \((\mathrm{C1})\) fails.
2. Remove \(X_i\) if \((\mathrm{C3})\) fails.
3. For each remaining \(X_i\), check all control boxes \(U_j\):
   - if \(U_j\) is proven infeasible, skip it;
   - otherwise verify the consequent for \(X_i\times U_j\times\mathcal D\).
4. Remove \(X_i\) if any potentially feasible \(U_j\) can transition outside the current accepted set.
5. Stop when no cells are removed.

The final fixed point is

$$
\Omega_c^{\mathrm{cert}}
=
S_\infty.
$$

The invariant-set proof is then a direct induction:

$$
x_t\in S_\infty
\Longrightarrow
x_{t+1}\in S_\infty
\Longrightarrow
g(x_t)\ge0.
$$

### 9.4 Unknown Handling

If a verifier result is unknown, mark the corresponding condition as failed for certification:

$$
\text{unknown}
\Longrightarrow
\text{not certified}.
$$

Unknown cells can be split and retried. They cannot be counted in

$$
\Omega_c^{\mathrm{cert}}.
$$

---

## 10. \(c\)-Search Protocol

Sweep:

$$
c\in\mathcal C_{\mathrm{sweep}}.
$$

Recommended initial sweep:

$$
\mathcal C_{\mathrm{sweep}}
=
\{0.00,0.02,0.05,0.10,0.15,0.20\}
$$

after normalizing \(V_\theta\) and \(g\) to comparable scale.

For each \(c\), run the cell-refinement certificate and record:

$$
\operatorname{Vol}(\Omega_c^{\mathrm{cert}}),
$$

$$
\rho(c)
=
\frac{
\operatorname{Vol}(\Omega_c^{\mathrm{cert}})
}{
\operatorname{Vol}(\Omega^\star)
},
$$

verification time, branch count, and unknown-cell count.

Expected behavior:

$$
c\uparrow
\Longrightarrow
\Omega_c\downarrow,
\qquad
\text{but margins improve}.
$$

The best \(c\) is not necessarily \(c=0\). The best \(c\) is the one producing the largest non-vacuous certified set under the verification budget.

---

## 11. Falsification Audit

After obtaining \(\Omega_c^{\mathrm{cert}}\), run an empirical audit. This is not the proof; it is a sanity check and paper evidence.

### 11.1 Dense Rollout Audit

Sample

$$
x_0\in\Omega_c^{\mathrm{cert}},
$$

and simulate the runtime filter under dense disturbance choices:

$$
d_t\in\mathcal D_{\mathrm{dense}}.
$$

Report:

$$
\#\{\text{rollouts leaving }\Omega_c^{\mathrm{cert}}\},
\qquad
\#\{\text{rollouts with }g(x_t)<0\}.
$$

Expected result:

$$
0.
$$

### 11.2 Random and Gradient-Adversarial Audit

Also run:

$$
d_t\sim\operatorname{Unif}(\mathcal D),
$$

and an adversarial search over disturbance sequences:

$$
\min_{d_0,\ldots,d_{T-1}\in\mathcal D}
\min_{0\le t\le T}g(x_t).
$$

The audit should not find certified violations. If it does, either the verifier implementation is wrong, the dynamics used by the verifier differ from rollout dynamics, or the reported set is not actually invariant.

---

## 12. Expected Figures and Tables

### 12.1 Main Figure

Plot a slice of the certified set over the oracle robust safe set:

$$
\Omega^\star
=
\{x:V^\star(x)\ge0\},
\qquad
\Omega_c^{\mathrm{cert}}.
$$

Use heading slices such as:

$$
\psi\in\left\{0,\frac{\pi}{2},\pi,-\frac{\pi}{2}\right\}.
$$

### 12.2 \(c\)-Search Curve

Plot:

$$
c
\mapsto
\rho(c),
$$

and

$$
c
\mapsto
\text{verification time}.
$$

### 12.3 Verification Table

Report:

| Item | Count / Value |
|---|---:|
| total state cells |  |
| cells outside candidate |  |
| cells removed by \((\mathrm{C1})\) |  |
| cells removed by \((\mathrm{C3})\) |  |
| cells removed by \((\mathrm{C2})\) |  |
| final certified cells |  |
| \(\operatorname{Vol}(\Omega_c^{\mathrm{cert}})\) |  |
| \(\rho(c)\) |  |
| certified-but-violated audit count | \(0\) expected |

### 12.4 Ablation Table

Minimum ablations:

| Variant | Expected Outcome |
|---|---|
| full-\(\mathcal D\) certificate | sound, smaller set |
| endpoint-only \(d\) check | may match if worst case is endpoint; report as structure, not proof unless justified |
| no witness pinning | \((\mathrm{C3})\) unavailable / harder |
| fixed \(c=0\) | often smaller or fails |
| \(c\)-search | recovers non-vacuous certificate |

Do not spend early time on a full learned-adversary ablation. That belongs to Theorem S / E1, after E0 works.

---

## 13. Go / No-Go Gates

### Gate A: Oracle Works

Pass if:

$$
\Omega^\star=\{x:V^\star(x)\ge0\}
$$

has a meaningful shape and is not empty / trivial.

### Gate B: Training Produces Margin

Pass if the grid witness margin is positive on a nontrivial subset:

$$
\Pr_{x\sim\Omega^\star}
\left[
m_\theta(x)>0
\right]
$$

is meaningfully above zero.

### Gate C: \((\mathrm{C3})\) Certifies

Pass if CROWN proves witness feasibility on nontrivial cells:

$$
\underline h_3(X_i,\mathcal D)\ge0.
$$

If this fails everywhere, improve \(\pi^\flat_\varphi\), lower \(\gamma\), or raise \(c\).

### Gate D: \((\mathrm{C2})\) Certifies

Pass if the final fixed point is nonempty:

$$
\Omega_c^{\mathrm{cert}}\neq\varnothing.
$$

This is the main project gate.

### Gate E: Paper-Facing Result

A Dubins result becomes paper-facing if:

$$
\rho(c)
=
\frac{
\operatorname{Vol}(\Omega_c^{\mathrm{cert}})
}{
\operatorname{Vol}(\Omega^\star)
}
$$

is visually and numerically meaningful. A rough interpretation:

| \(\rho(c)\) | Interpretation |
|---:|---|
| \(<0.02\) | likely too small for main paper |
| \(0.02\) to \(0.10\) | preliminary / appendix unless the problem is very hard |
| \(0.10\) to \(0.30\) | plausible main result |
| \(>0.30\) | strong low-dimensional certificate |

These thresholds are not theorem-level requirements; they are practical publication judgment.

---

## 14. Development Timeline

### Week 1: Oracle and Training

Deliverables:

$$
V^\star,\quad Q^\star,\quad u^\flat,
\quad
V_\theta,\quad Q_\theta,\quad \pi^\flat_\varphi.
$$

Also deliver the grid pilot:

$$
\Omega_c^{\mathrm{grid}}>0
$$

for at least one \(c\) and \(\gamma\).

### Week 2: CROWN for \((\mathrm{C3})\)

Deliverables:

$$
\underline h_3(X_i,\mathcal D)
$$

for state boxes \(X_i\), plus a map of which cells pass witness feasibility.

### Week 3: CROWN / Interval for \((\mathrm{C2})\)

Deliverables:

$$
\Omega_c^{\mathrm{cert}}
$$

after shrink-refinement, plus the first \(c\)-search curve.

### Week 4: Audit and Figures

Deliverables:

$$
\#\{\text{certified-but-violated}\}=0,
$$

plots of \(\Omega^\star\), \(\Omega_c^{\mathrm{cert}}\), and ablations.

If Week 3 fails, do not move to Theorem S. Fix the certificate pipeline first.

---

## 15. Implementation Skeleton

Recommended files:

| File | Purpose |
|---|---|
| `dubins_model.py` | dynamics \(f\), wrap, safety margin \(g\) |
| `dubins_oracle_dp.py` | grid Bellman-Isaacs value iteration |
| `dubins_dataset.py` | samples for \(V^\star,Q^\star,u^\flat\) |
| `train_dubins_qcbf.py` | supervised training |
| `certify_dubins_c3.py` | CROWN check for \((\mathrm{C3})\) |
| `certify_dubins_c2.py` | control-box loop, successor boxes, CROWN check |
| `refine_dubins_cert.py` | monotone cell shrink-refinement |
| `audit_dubins_cert.py` | dense / random / adversarial falsification |
| `plot_dubins_cert.py` | figures and tables |

Cache artifacts:

| Artifact | Contents |
|---|---|
| `oracle_vstar.npz` | grid, \(V^\star\), convergence metadata |
| `oracle_q_samples.npz` | training samples for \(Q^\star\) |
| `witness_labels.npz` | \(u^\flat(x)\) labels |
| `models/*.pt` | frozen \(V_\theta,Q_\theta,\pi^\flat_\varphi\) |
| `cert_cells.json` | accepted / rejected / unknown cell metadata |
| `audit_results.json` | falsification audit results |

---

## 16. Common Failure Modes

### 16.1 Oracle Safe Set Is Tiny

Symptoms:

$$
\Omega^\star\approx\varnothing.
$$

Fix:

$$
d_{\max}\downarrow,
\qquad
\omega_{\max}\uparrow,
\qquad
r_{\mathrm{obs}}\downarrow,
\qquad
R_{\mathrm{world}}\uparrow.
$$

### 16.2 Witness Fails Everywhere

Symptoms:

$$
\underline h_3(X_i,\mathcal D)<0
\quad
\text{for all }X_i.
$$

Fix:

$$
\gamma\downarrow,
\qquad
c\uparrow,
\qquad
\lambda_m\uparrow,
\qquad
\text{improve }\pi^\flat_\varphi.
$$

### 16.3 \((\mathrm{C2})\) Fails Due to Bound Looseness

Symptoms:

$$
\text{grid pilot passes},
\qquad
\text{CROWN / interval returns unknown or negative bounds}.
$$

Fix:

$$
X_i\text{ split finer},
\qquad
U_j\text{ split finer},
\qquad
\psi\text{ boxes split finer},
\qquad
c\uparrow.
$$

Also try the successor-box route before direct dynamics composition.

### 16.4 Angle Wrapping Creates Huge Successor Boxes

Symptoms:

$$
\Psi^+
\text{ covers most of }[-\pi,\pi).
$$

Fix:

Split heading cells near the wrap boundary. Treat periodic successors as multiple boxes:

$$
\Psi^+
=
\Psi_1^+\cup\Psi_2^+.
$$

Never merge them into one conservative interval unless the result is still useful.

---

## 17. Final Minimum Success Statement

The minimum successful Dubins E0 statement is:

> We trained a frozen deployed robust state-action \(Q\)-CBF artifact
> \((V_\theta,Q_\theta,\pi^\flat_\varphi)\) from a low-dimensional Dubins oracle and then certified the artifact directly. Over the full disturbance set \(\mathcal D\), the verifier proved witness feasibility and robust transition closure on a non-vacuous cell set \(\Omega_c^{\mathrm{cert}}\). Dense, random, and adversarial falsification found zero certified violations. No approximation guarantee to \(V^\star,Q^\star\) is used in the soundness argument.

In formulas, the result is:

$$
\Omega_c^{\mathrm{cert}}\neq\varnothing,
$$

$$
\forall x_0\in\Omega_c^{\mathrm{cert}},
\ \forall \mathbf d\in\mathcal D^\infty:
\quad
x_t\in\Omega_c^{\mathrm{cert}}
\quad
\text{and}
\quad
g(x_t)\ge0
\quad
\forall t\ge0,
$$

with the proof resting only on verified conditions for the deployed networks.

