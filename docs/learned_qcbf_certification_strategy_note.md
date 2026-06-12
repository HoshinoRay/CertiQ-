# Learned \(Q\)-CBF Certification Strategy Note

Date: 2026-06-11

Purpose: record the updated project understanding after the \(P_1\) experiment,
the Step-0 exact ceiling probe, and the discussion about how to keep the project
on the original certification direction without turning the result into an
empirical-only \(RL\) story.

## 1. Non-negotiable project direction

The project direction does **not** change:

\[
\Omega_{\mathrm{cert}}\neq\varnothing,\qquad \rho>0.
\]

The goal remains certification of a deployed learned robust \(Q\)-CBF filter.
The project should not be reframed as a purely empirical learned-safety story and
should not accept a certificate-free rollout claim as a substitute for Gate D.

The learned object is still the frozen neural artifact

\[
(V_\theta,Q_\theta,\pi^\flat_\varphi).
\]

The important correction is not to distrust \(Q_\theta\) because it is learned.
In \(Q\)-CBF, \(Q_\theta\) is supposed to be learned by \(RL\), adversarial
training, or minimax Bellman training. The correction is that learned \(Q_\theta\)
cannot be *assumed* to have the ideal \(Q^\star\) semantics inside the proof.

The firewall remains:

\[
Q_\theta\approx Q^\star,\qquad V_\theta\approx V^\star,\qquad
Q_\theta\approx V_\theta\circ f
\]

may be training targets, diagnostics, or synthesis aids, but they are not proof
assumptions.

## 2. Learning versus certification

The clean project principle is:

\[
\boxed{\text{Training is for non-vacuity; verification is for soundness.}}
\]

Training may include \(RL\), adversarial disturbances, \(Q\)-CBF hinge losses,
Bellman residuals, CEGIS counterexamples, and verifier-aware regularization.
None of these training mechanisms is itself a safety proof.

The proof path is:

\[
\text{learning/synthesis}\;\longrightarrow\;\text{candidate networks}
\]

\[
\text{frozen candidate}+\text{sound verifier}\;\longrightarrow\;\text{certificate}.
\]

The invalid path is:

\[
\text{learning loss small}\;\longrightarrow\;\text{safety guarantee}.
\]

Thus, adding certification-aware losses to \(RL\) is not contradictory. These
losses only bias training toward candidates that a later independent verifier can
prove. The final safety claim must cite only the frozen networks and the verifier
discharge.

## 3. Current experimental facts

### 3.1 \(P_1\) direct-composition result

The \(P_1\) direct-composition primitive removed the intermediate
outward-rounded successor box and passed the analytic ideal first:

\[
\text{analytic ideal pass: } 86{,}580 \text{ cells},\qquad
\min \Delta V\approx -3.05\times 10^{-16}.
\]

For the learned trio at \(80^3\)-scale resolution, the direct-composition
sublevel certificate still failed:

\[
\rho=0.
\]

Median direct undershoot across three seeds was approximately

\[
0.090\text{--}0.151,
\]

while the old box-route undershoot was only about

\[
0.004\text{--}0.008
\]

larger. Therefore the old successor-box slack is not the main term.

### 3.2 Step-0 exact ceiling probe

The exact pointwise boundary-band probe evaluated the learned networks directly,
with no \(CROWN\), no \(IBP\), and exact forward dynamics on the sampled states.
The probe examined the in-set boundary shell

\[
V_\theta(x)\in[0,\delta],\qquad \delta=0.1,
\]

using \(400k\) samples and three seeds.

The witness \(Q\)-predicate was clean:

\[
\min_d Q_\theta(x,\pi^\flat_\varphi(x),d)-\gamma V_\theta(x)
\ge +0.052,
\]

with zero negative samples in the boundary shell.

But the direct successor-value decrease had real negative pockets:

\[
h_V(x,d)
=
V_\theta(f(x,\pi^\flat_\varphi(x),d))-\gamma V_\theta(x).
\]

For the deployed discount \(\gamma=0.5\),

\[
\min h_V\approx -0.115,
\qquad
\Pr[h_V<0]\approx 1.3\%\text{--}5.9\%.
\]

For hard invariance \(\gamma=1\),

\[
\Pr[h_V<0]\approx 6\%\text{--}16\%.
\]

This is decisive as a falsification result: for the current learned object, the
natural set \(\{V_\theta\ge0\}\) is not robustly forward-invariant under the
witness policy on the sampled boundary shell.

## 4. Correct attribution

The fact that the runtime filter can run is explained by \(C3\):

\[
\min_d Q_\theta(x,\pi^\flat_\varphi(x),d)\ge \gamma V_\theta(x).
\]

The fact that the direct successor-value condition fails is explained by a gap
between the learned \(Q_\theta\) predicate and the actual successor value:

\[
Q_\theta(x,u,d)
\quad\text{versus}\quad
V_\theta(f(x,u,d)).
\]

The current diagnosis is:

\[
\text{\(C3\) is clean, but direct \(C2\) has real object-level holes.}
\]

Therefore, if the certificate route requires

\[
V_\theta(f(x,u,d))\ge \gamma V_\theta(x),
\]

then continuing to optimize only verifier relaxations is wasted work until the
object-level holes are repaired.

This does **not** imply \(Q_\theta\) should stop being learned by \(RL\). It means
the \(RL\)-learned \(Q_\theta\) must be trained with certificate-seeking constraints
and later independently verified.

## 5. \(C2\) is not redundant

Rollout success does not imply certified invariance:

\[
\exists\text{ safe rollouts}
\;\not\Rightarrow\;
\forall x\in\Omega_{\mathrm{cert}},\forall d\in\mathcal D,\;
f(x,u,d)\in\Omega_{\mathrm{cert}}.
\]

Some dynamic closure condition is necessary for recursive safety:

\[
x_t\in\Omega_{\mathrm{cert}}
\;\Rightarrow\;
x_{t+1}\in\Omega_{\mathrm{cert}}
\;\Rightarrow\;
x_{t+2}\in\Omega_{\mathrm{cert}}
\;\Rightarrow\;\cdots.
\]

The exact form of \(C2\) can change. It may be a direct successor-value condition,
a \(Q\)-lower-bound consistency condition, an accepted-action closure condition,
or a finite-horizon terminal condition. But it cannot be removed entirely.

Three legitimate certificate forms are:

1. Direct \(V\)-successor closure:

\[
V_\theta(f(x,u,d))\ge \gamma V_\theta(x).
\]

2. \(Q\)-predicate plus conservative successor semantics:

\[
Q_\theta(x,u,d)\ge \gamma V_\theta(x),
\qquad
Q_\theta(x,u,d)\le V_\theta(f(x,u,d)).
\]

3. Runtime accepted-action closure:

\[
\forall u\in\Phi_\theta(x),\forall d\in\mathcal D,\qquad
V_\theta(f(x,u,d))\ge0,
\]

where

\[
\Phi_\theta(x)
=
\{u:\min_d Q_\theta(x,u,d)\ge\gamma V_\theta(x)\}.
\]

## 6. Training modification: keep \(Q_\theta\) learned, add certifiability bias

The next training change should not replace \(Q_\theta\) by a non-\(Q\)-CBF
surrogate. \(Q_\theta\) should remain a learned robust state-action value. The
change is to add certificate-seeking terms to the \(RL\) or distillation process.

The total objective should be viewed as

\[
\mathcal L
=
\mathcal L_{\mathrm{RL}}
+
\lambda_{\mathrm{QCBF}}\mathcal L_{\mathrm{QCBF}}
+
\lambda_{\mathrm{Bell}}\mathcal L_{\mathrm{Bell}}
+
\lambda_{\mathrm{CEGIS}}\mathcal L_{\mathrm{CEGIS}}
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
\]

### 6.1 \(Q\)-CBF witness hinge

Keep the witness feasibility loss:

\[
\mathcal L_{\mathrm{QCBF}}
=
\left[
\eta
+
\gamma V_\theta(x)
-
\min_d Q_\theta(x,\pi^\flat_\varphi(x),d)
\right]_+.
\]

This is already mostly successful in the current Step-0 probe.

### 6.2 Bellman / successor consistency

If the proof needs \(Q_\theta\) to be conservative relative to the true successor
value, add a one-sided consistency term:

\[
\mathcal L_{\mathrm{Bell}}
=
\left[
Q_\theta(x,u,d)
-
V_\theta(f(x,u,d))
+
\eta_Q
\right]_+.
\]

This encourages

\[
Q_\theta(x,u,d)\le V_\theta(f(x,u,d))-\eta_Q.
\]

This is not a proof assumption. It is a training pressure. The final verifier
must still prove the required inequalities on the frozen networks.

### 6.3 Direct \(C2\) counterexample repair

For witness-policy invariance, add direct successor-value training on the
boundary and counterexample buffer:

\[
\mathcal L_{C2}
=
\left[
\eta_V
+
\gamma V_\theta(x)
-
\min_d V_\theta(f(x,\pi^\flat_\varphi(x),d))
\right]_+.
\]

This term directly repairs the negative pockets found by Step-0.

### 6.4 Boundary-focused weighting

Certification losses should be strongest near the safety boundary:

\[
w(x)
=
\exp\left(-\frac{|V_\theta(x)|}{\tau}\right).
\]

Use

\[
w(x)\mathcal L_{\mathrm{QCBF}},\qquad
w(x)\mathcal L_{\mathrm{Bell}},\qquad
w(x)\mathcal L_{C2}.
\]

This minimizes interference with performance far from the boundary.

### 6.5 Non-vacuity anchors

Prevent the network from satisfying constraints by shrinking the certified set
to empty. On an analytic or sampled safe core,

\[
V_{\mathrm{target}}(x)\ge 0.2,
\]

enforce

\[
\left[\mu-V_\theta(x)\right]_+.
\]

On unsafe samples,

\[
g(x)\le -0.05,
\]

enforce

\[
\left[V_\theta(x)+\mu\right]_+.
\]

These are training aids only. They do not enter the final proof.

## 7. Performance preservation

The desired property is not that certificate-aware training has zero mathematical
effect. The desired property is that it minimally repairs the original learned
\(Q\)-safe filter while preserving runtime behavior and performance.

Let \(\theta_{\mathrm{base}}\) be the baseline filter from the original \(RL\)
training. Train the certified candidate from \(\theta_{\mathrm{base}}\) and add a
behavior-matching term:

\[
\mathcal L_{\mathrm{match}}
=
\mathbb E_{(x,u)\sim\mathcal D_{\mathrm{rollout}}}
\left[
Q_\theta(x,u,d)-Q_{\mathrm{base}}(x,u,d)
\right]^2.
\]

Also preserve the ranking of high-value baseline actions:

\[
Q_\theta(x,u_{\mathrm{base}},d)
\ge
Q_\theta(x,u,d)-\epsilon.
\]

The operational claim should be:

\[
\rho>0,
\]

\[
\Pr[u_{\mathrm{cert}}\neq u_{\mathrm{base}}]\le \epsilon_u,
\]

\[
\frac{|J_{\mathrm{cert}}-J_{\mathrm{base}}|}{|J_{\mathrm{base}}|}
\le \epsilon_J.
\]

If certificate constraints only modify hidden unsafe pockets and leave nominal
rollouts unchanged, performance should be preserved. If performance drops
substantially, that is evidence that the baseline performance depended on
uncertified optimistic \(Q_\theta\) regions.

## 8. CEGIS protocol

The synthesis loop should be:

\[
\text{train}
\;\rightarrow\;
\text{exact probe}
\;\rightarrow\;
\text{collect counterexamples}
\;\rightarrow\;
\text{repair}
\;\rightarrow\;
\text{re-probe}
\;\rightarrow\;
\text{verify}.
\]

Step-0 should save states and disturbances satisfying

\[
V_\theta(x)\ge0,\qquad
V_\theta(f(x,\pi^\flat_\varphi(x),d))-\gamma V_\theta(x)<0.
\]

The counterexample buffer should include

\[
x,\quad d^\star,\quad V_\theta(x),\quad
V_\theta(f(x,\pi^\flat_\varphi(x),d^\star)),\quad
Q_\theta(x,\pi^\flat_\varphi(x),d^\star).
\]

Add jitter around counterexamples to train neighborhoods, not isolated points.

Suggested probe gate before expensive verification:

\[
\min h_V(x,d)\ge 0.02
\]

and

\[
\Pr[h_V(x,d)<0]=0
\]

on dense boundary-band samples across at least three seeds.

## 9. Verification route after object repair

Once exact pointwise probes no longer find \(C2\) holes, return to verification.

### 9.1 Relational \(P_{1.5}\)

The most promising next verifier is a relational direct-difference bound:

\[
G_\theta(x,d)
=
V_\theta(f(x,\pi^\flat_\varphi(x),d))-\gamma V_\theta(x).
\]

Instead of separately bounding

\[
\underline V_\theta(f(C))
\quad\text{and}\quad
\overline V_\theta(C),
\]

the verifier should lower-bound \(G_\theta\) over the same input cell:

\[
\underline G_\theta(C,\mathcal D)\ge0.
\]

A practical first implementation can use affine lower and upper functionals:

\[
V_\theta(y)\ge A_+y+\beta_+,
\]

\[
V_\theta(x)\le A_0x+\beta_0,
\]

so that

\[
G_\theta(x,d)
\ge
A_+f(x,d)+\beta_+
-
\gamma(A_0x+\beta_0).
\]

Then minimize the right-hand side over the shared variables
\((x,d)\). This preserves current-successor correlation and avoids paying the
full cell oscillation tax

\[
\operatorname{osc}_C(V_\theta).
\]

### 9.2 Adaptive branch-and-bound

If relational verification is close but not closed, split only active failing
cells:

\[
C:\quad \underline G_\theta(C,\mathcal D)<0.
\]

Uniform global \(P_2\) refinement should be used only as a scaling diagnostic,
not as the primary rescue mechanism.

### 9.3 \(P_4\) fallback

If direct \(C2\), relational verification, and CEGIS still fail, then consider
finite-horizon or recursive-feasibility certificates. This changes the
certificate form more than relational \(P_{1.5}\), so it should not be the first
response to the current diagnosis.

## 10. What not to claim

Do not claim:

\[
\text{\(RL\) training guarantees safety.}
\]

Do not claim:

\[
\text{small training loss guarantees safety.}
\]

Do not claim:

\[
Q_\theta\approx Q^\star
\]

inside the proof.

Do not claim that a successful rollout proves \(C2\), because

\[
\exists\text{ safe visited trajectories}
\;\not\Rightarrow\;
\forall x\in\Omega_{\mathrm{cert}},\forall d\in\mathcal D,\;x^+\in\Omega_{\mathrm{cert}}.
\]

The allowed claim is:

\[
\text{The learned networks are trained as candidates and then independently certified.}
\]

## 11. Paper framing

The strongest paper framing is:

> We train robust neural \(Q\)-CBF filters with certification-aware synthesis, but
> do not trust the training process. The frozen learned networks are certified
> post hoc by a sound verifier. Training is used only to obtain non-vacuous
> candidates; safety follows only from the verified inequalities.

In compact form:

\[
\boxed{
\text{Learned } Q\text{-CBF candidate}
+
\text{sound post-hoc verifier}
\Rightarrow
\text{certified non-empty safe filter}.
}
\]

The empirical burden is now precise:

1. ordinary learned \(Q\)-CBF training may run but can have hidden \(C2\) holes;
2. certification-aware \(Q\)-CBF training repairs those holes while preserving
   nominal filter behavior;
3. the independent verifier proves \(\Omega_{\mathrm{cert}}\neq\varnothing\);
4. runtime performance remains close to the baseline \(Q\)-safe filter.

## 12. Immediate next actions

1. Extend `run_probe_ceiling.py` to save \(C2\) counterexamples.
2. Add direct \(C2\) successor-value loss to `distill.py`.
3. Add one-sided Bellman / \(Q\)-successor consistency loss.
4. Add safe-core and unsafe anchors to prevent vacuous shrinkage.
5. Add behavior-matching loss to preserve the baseline \(Q\)-safe filter.
6. Run quick seed \(0\) probe and compare:

\[
\min h_V,\qquad
\Pr[h_V<0],\qquad
\min h_Q,\qquad
\rho_{\mathrm{proxy}}.
\]

7. If Step-0 exact ceiling becomes positive, implement relational \(P_{1.5}\).
8. Only after relational verification passes or nearly passes should full
   \(P_1\)/resolution/branch-and-bound sweeps be run.

