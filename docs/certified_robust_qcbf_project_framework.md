# Project Framework - Sound Certification for Deployed Robust Q-CBF Filters

> **Status:** v9 project design. This version adopts the sound-certificate reframe: the main contribution is the deterministic, full-$\mathcal D$ certificate for the deployed learned robust state-action $Q$-CBF filter. Theorem S is retained as the structural reason why the learned-adversary shortcut is not a sound robustness certificate. Split conformal is retained as the high-dimensional deployment extension, not the core theorem.
>
> **Core thesis.** Robust state-action $Q$-CBF synthesis gives the ideal object $V^\star,Q^\star$, but the object that runs is a finite learned filter built from
> $$
> (V_\theta,Q_\theta,\pi^\flat_\varphi).
> $$
> This project certifies **that deployed artifact directly**. If the verifier proves (C1)--(C3) over the full disturbance set $\mathcal D$, then Theorem A gives recursive feasibility, robust forward invariance, and safety for every disturbance sequence, with **no assumption** that $Q_\theta\approx Q^\star$ or $V_\theta\approx V^\star$. A poor network only makes the certified set $\Omega_c$ small or empty; it never makes the certificate unsound.
>
> **Role of the other pieces.** Theorem S / Corollary S3 explain why certifying against a continuous learned adversary $d_\psi$ is structurally unsafe: the learned-adversary feasible set can strictly over-state the full-$\mathcal D$ robust feasible set. This motivates the full-$\mathcal D$ discharge in Theorem A; it is not the headline contribution by itself. The conformal lane, Theorem F, is a high-dimensional extension for frozen deployed loops when deterministic full-$\mathcal D$ verification becomes impractical or vacuous.

---

## 0. Spine

Write the project as

$$
\boxed{
\text{sound post-hoc certification of deployed learned robust state-action }Q\text{-CBF filters}
}
$$

with two lanes:

$$
\boxed{
\text{deterministic full-}\mathcal D\text{ certificate}
}
\quad\text{(core)}
\qquad+\qquad
\boxed{
\text{conformal feasibility-survival certificate}
}
\quad\text{(high-dimensional extension)}.
$$

The single locked statement is:

> We give a sound post-hoc certificate for the **deployed learned robust state-action $Q$-CBF filter**. By verifying the actual networks $(V_\theta,Q_\theta,\pi^\flat_\varphi)$ against the true dynamics and the full disturbance set $\mathcal D$, Theorem A proves recursive feasibility, robust forward invariance, and safety on a certified set $\Omega_c$, without any approximation hypothesis to $V^\star,Q^\star$. Theorem S explains why learned-adversary feasibility cannot replace this full-$\mathcal D$ discharge. For high-dimensional deployment regimes where exact verification exits, Theorem F gives a finite-sample conformal feasibility-survival certificate for the frozen deployed loop.

The roles are now:

1. **Theorem A is the main proof chain.** It turns the verified inequalities (C1)--(C3) on $(V_\theta,Q_\theta,\pi^\flat_\varphi)$ into recursive feasibility, robust invariance, and safety.
2. **The verifier machinery is part of the contribution, not an appendix detail.** Witness compression, sound full-$\mathcal D$ discharge, implication encoding, model-error augmentation, and monotone shrink-refinement are what make Theorem A checkable on learned networks.
3. **Theorem S is the structural motivation for full-$\mathcal D$ verification.** It says the continuous learned adversary can miss switching worst cases, so a certificate against $d_\psi$ is not a robustness certificate.
4. **Conformal is the high-dimensional extension.** It gives distributional feasibility-survival coverage for the frozen deployed loop when deterministic verification becomes too costly. It is useful and honest, but it is not the core novelty.

This means the old framing must be inverted:

$$
\text{Theorem S}
\Longrightarrow
\text{why learned-adversary certification is insufficient}
\Longrightarrow
\text{Theorem A full-}\mathcal D\text{ certificate}.
$$

not

$$
\text{Theorem S}
\Longrightarrow
\text{FFR}
\Longrightarrow
\text{conformal as the main contribution}.
$$

---

## 1. Context and Competition

### 1.1 Home Line

- **HCSF / Q-CBF** (`2504.11717`, RSS 2025) proposes the state-action safety-filter object.
- **Maximal Robust Q-CBF** (`2604.13192`, 2026) supplies the robust state-action $Q$-CBF synthesis and proves properties of the exact object $V^\star,Q^\star$, but the deployed object is a finite learned network and the learned-adversary guarantee is local. Its post-hoc-verification remark is the opening for this work.

The home line answers:

$$
\text{How do we synthesize a robust state-action }Q\text{-CBF object?}
$$

This project answers:

$$
\text{How do we soundly certify the learned robust }Q\text{-CBF filter that actually runs?}
$$

### 1.2 Crowded Rooms We Must Not Claim

Do **not** claim:

- first neural-network verifier;
- first learned CBF verification result;
- first conformal safety filter;
- first trajectory-level conformal safety guarantee;
- first conformal recursive-feasibility result;
- first performative / strategic-distribution-shift safety result;
- first high-dimensional conformal CBF result.

The competitive room includes at least:

- Sibai et al. (`2511.07899`): trajectory-level conformal safety filters for learned HJ filters;
- ACoFi (`2604.18482`): adaptive conformal inference for learned HJ safety filters;
- Stamouli--Lindemann--Pappas (`2405.10875`): conformal recursive feasibility for MPC;
- Li et al. (`2505.24097`) and Perdomo et al. (`2002.06673`): performative / decision-dependent risk and distribution shift;
- Lindemann-line interactive robust CP work (`2511.10586`): policy updates, behavior shift, and robust CP in interaction;
- RBN (`2505.11755`): high-dimensional learned CBF / reachability barrier networks with conformal safety;
- high-dimensional learned HJ / MPC recursive-feasibility work such as `2604.23863`.

The defensible novelty is the narrow conjunction:

$$
\text{deployed learned network}
\times
\text{robust state-action }Q\text{-CBF}
\times
\text{full-}\mathcal D\text{ sound certification}
\times
\text{recursive feasibility + robust invariance + safety}.
$$

This is stronger and cleaner than a broad conformal or false-feasibility claim.

---

## 2. Main Claim

### 2.1 Identity Sentence

Use this style in the abstract / intro:

> Robust state-action $Q$-CBF filters are synthesized from an ideal reachability object, but deployment uses finite learned networks and often a continuous learned adversary. We give a sound post-hoc certificate for the deployed learned filter itself: verifying $(V_\theta,Q_\theta,\pi^\flat_\varphi)$ over the full disturbance set $\mathcal D$ yields deterministic recursive feasibility, robust forward invariance, and safety, with no assumption that $Q_\theta$ approximates $Q^\star$. We prove that the learned-adversary shortcut is structurally unsound when worst-case selectors switch, which motivates full-$\mathcal D$ discharge. Finally, for high-dimensional regimes where exact verification becomes impractical, we provide a split-conformal feasibility-survival certificate on the frozen deployed loop.

### 2.2 What This Is

This is a **deployed-object certification** paper. The object of certification is not the ideal robust value pair

$$
(V^\star,Q^\star),
$$

but the actual learned artifact

$$
(V_\theta,Q_\theta,\pi^\flat_\varphi)
$$

and the deployed robust filter

$$
\Phi(x,u)
:\Longleftrightarrow
\min_{d\in\mathcal D}Q_\theta(x,u,d)
\ge
\beta\!\left(V_\theta(x)\right),
$$

with the QP / projection layer selecting a feasible action. On a certified superlevel set

$$
\Omega_c:=\{x:V_\theta(x)\ge c\},
$$

the verifier checks:

$$
\text{(C1)}\qquad
\forall x\in\Omega_c:\ g(x)\ge0,
$$

$$
\text{(C2)}\qquad
\forall x\in\Omega_c,\ \forall u\text{ with }\Phi(x,u),\ \forall d\in\mathcal D:
\quad
V_\theta(f(x,u,d))\ge c+\varepsilon,
$$

$$
\text{(C3)}\qquad
\forall x\in\Omega_c,\ \forall d\in\mathcal D:
\quad
Q_\theta(x,\pi^\flat_\varphi(x),d)
\ge
\beta(V_\theta(x))+\varepsilon.
$$

Theorem A in `theory_core.md` then proves:

$$
(C1)\wedge(C2)\wedge(C3)
\Longrightarrow
\begin{cases}
\text{recursive feasibility},\\
\text{robust forward invariance of }\Omega_c,\\
\text{safety }g(x_t)\ge0\ \forall t,
\end{cases}
\qquad
\forall \mathbf d\in\mathcal D^\infty.
$$

Crucially, the proof uses only the verified inequalities and the true dynamics $f$. It does **not** use

$$
Q_\theta=Q^\star,
\qquad
\|Q_\theta-Q^\star\|\le\delta,
\qquad
V_\theta=V^\star,
$$

or any approximation theorem. This is the soundness firewall:

$$
\text{bad }Q_\theta
\Longrightarrow
\Omega_c\text{ small or empty},
\qquad
\text{not an unsound certificate}.
$$

### 2.3 What This Is Not

- Not a claim about the ideal $Q^\star$ alone. The ideal backbone is lineage; the certificate is on $Q_\theta$ and $V_\theta$.
- Not a certificate against a learned adversary $d_\psi$ only. Robust means quantification over the full relevant set $\mathcal D$.
- Not a generic neural-network verification paper. The contribution is the robust state-action $Q$-CBF proof obligation and its recursive-feasibility structure.
- Not a new conformal prediction theorem. The conformal result is a deployment extension and uses standard split-conformal coverage.
- Not a fully model-free deterministic certificate. The deterministic lane is model-free at synthesis/runtime but model-bounded at certification, because (C2) composes the certified dynamics model $f$ or an augmented uncertainty model.

---

## 3. Technical Contributions

### C1. Deterministic Certificate for the Deployed Learned Filter (Theorem A)

This is the headline theorem.

The certificate is stated for the actual deployed networks:

$$
(V_\theta,Q_\theta,\pi^\flat_\varphi),
$$

not for the exact reachability solution. The proof chain is:

1. **Recursive feasibility from (C3).** The witness policy is directly verified:
   $$
   \forall x\in\Omega_c,\ \forall d\in\mathcal D:
   Q_\theta(x,\pi^\flat_\varphi(x),d)
   \ge
   \beta(V_\theta(x))+\varepsilon.
   $$
   Therefore
   $$
   \min_{d\in\mathcal D}Q_\theta(x,\pi^\flat_\varphi(x),d)
   \ge
   \beta(V_\theta(x)),
   $$
   so the feasible action set is nonempty.

2. **Robust invariance from (C2).** For any feasible control the deployed filter may choose,
   $$
   \Phi(x,u_t)=1,
   $$
   and for any realized disturbance $d_t\in\mathcal D$, (C2) gives
   $$
   V_\theta(x_{t+1})
   =
   V_\theta(f(x_t,u_t,d_t))
   \ge
   c+\varepsilon
   \ge
   c.
   $$
   Hence $x_{t+1}\in\Omega_c$.

3. **Safety from (C1).** Since (C1) gives $\Omega_c\subseteq\mathcal K$ and (C2) keeps the trajectory inside $\Omega_c$,
   $$
   g(x_t)\ge0
   \qquad
   \forall t,\ \forall \mathbf d\in\mathcal D^\infty.
   $$

The theorem is deliberately stronger than checking a single selected optimizer $u^\star(x)$: (C2) quantifies over **all feasible** controls. This makes the certificate robust to tie-breaking and implementation details of the QP, at the cost of conservatism.

**Paper memory point.**

$$
\boxed{
\text{Theorem A certifies the learned object directly; it never assumes }Q_\theta\approx Q^\star.
}
$$

### C2. Sound Full-$\mathcal D$ Discharge and Recursive-Feasibility Machinery

Theorem A is only useful if (C1)--(C3) can be checked soundly. The technical machinery from `theory_core.md` must be presented as part of the main method:

**Witness compression (T6 / Lemma W).** Native recursive feasibility has the quantifier form

$$
\forall x\in\Omega_c\;\exists u\in\mathcal U\;\forall d\in\mathcal D:
\quad
Q_\theta(x,u,d)\ge\beta(V_\theta(x)).
$$

Pinning the deployed witness $u=\pi^\flat_\varphi(x)$ turns this into the pure universal condition (C3):

$$
\forall x\in\Omega_c\;\forall d\in\mathcal D:
\quad
Q_\theta(x,\pi^\flat_\varphi(x),d)
\ge
\beta(V_\theta(x))+\varepsilon.
$$

This is sound for **any** witness network. A suboptimal witness only makes the certificate harder to pass; it cannot make it unsound.

**Full-$\mathcal D$ inner-game discharge (Prop T2).** The disturbance $d$ is treated as an additional bounded verifier input block. Sound interval / linear relaxation plus branching over the $d$-box bounds the inner game without an HJI-style grid over a joint $(x,d)$ state. This is not free and not claimed to scale independently of dimension; the important soundness point is:

$$
\text{loose bounds}
\Longrightarrow
\Omega_c\text{ shrinks},
\qquad
\text{not false certification}.
$$

**Sound implication encoding (Lemma E).** The hard condition in (C2) has an antecedent:

$$
\Phi(x,u):
\quad
\min_{d'\in\mathcal D}Q_\theta(x,u,d')
\ge
\beta(V_\theta(x)).
$$

The verifier must prove

$$
\Phi(x,u)
\Longrightarrow
V_\theta(f(x,u,d))\ge c+\varepsilon
\qquad
\forall d\in\mathcal D.
$$

This is encoded with sound bounds on an antecedent margin and a consequent margin. This is one of the places where the paper is more than "run CROWN on another network": the verified predicate is a robust state-action filter implication, not a scalar safety label.

**Model-bounded certification (Corollary M).** The deterministic certificate composes the dynamics in (C2). If model error is present, it must be covered by an augmented uncertainty set

$$
\mathcal D_{\mathrm{aug}},
$$

so every true transition is represented by the certified transition model. The correct phrase is:

$$
\boxed{
\text{model-free synthesis / model-free runtime, model-bounded certification}.
}
$$

**Monotone shrink-refinement (Prop T4).** Because the certificate is for the deployed filter and any feasible selection, refinement uses the all-feasible predecessor

$$
\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S)
:=
\left\{
x:
\{u:\Phi(x,u)\}\ne\varnothing,\ 
\forall u\text{ with }\Phi(x,u),\ 
\forall d\in\mathcal D,\ 
f(x,u,d)\in S
\right\}.
$$

The refinement operator

$$
\mathcal T_\Phi(S)
:=
S\cap\mathcal K\cap\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S)
$$

shrinks toward a certified fixed point. Unknown cells are removed or left uncertified; they are never counted as safe.

### C3. Why the Learned-Adversary Shortcut Is Structurally Unsound (Prop U / Theorem S)

This section should stay sharp, but its role is motivation and necessity, not the main contribution.

For a continuous learned adversary $d_\psi(x,u)$, define

$$
q_\psi(z):=Q_\theta(z,d_\psi(z)),
\qquad
q_{\min}(z):=\min_{d\in\mathcal D}Q_\theta(z,d),
\qquad
z=(x,u).
$$

The learned-adversary feasible set and the full robust feasible set are

$$
\mathcal F_\psi(x)
:=
\left\{
u:
Q_\theta(x,u,d_\psi(x,u))
\ge
\beta(V_\theta(x))
\right\},
$$

$$
\mathcal F_{\mathrm{rob}}(x)
:=
\left\{
u:
\min_{d\in\mathcal D}Q_\theta(x,u,d)
\ge
\beta(V_\theta(x))
\right\}.
$$

Proposition U gives the basic inclusion

$$
\mathcal F_{\mathrm{rob}}(x)\subseteq\mathcal F_\psi(x),
$$

so certifying feasibility against $d_\psi$ over-approximates robust feasibility.

Theorem S strengthens this from a set-inclusion warning to a structural obstruction. If the true worst-case disturbance selector switches between two distinct strict minimizers $v^-\ne v^+$ across abutting regions, then no continuous plug-in adversary can represent that selector across the switching surface. Therefore, for every continuous $d_\psi$, there is a nonempty open set on which

$$
q_\psi(z)>q_{\min}(z).
$$

If the threshold

$$
\eta(x):=\beta(V_\theta(x))
$$

lands inside that value gap with margin,

$$
q_{\min}(x,u)<\eta(x)<q_\psi(x,u),
$$

then the same action is learned-adversary feasible but not robust feasible:

$$
\Phi_\psi(x,u)=1,
\qquad
\Phi_{\mathrm{rob}}(x,u)=0.
$$

This preserves the good part of the original Theorem S story: false feasibility is not merely undertraining. But the conclusion now feeds the core certificate:

$$
\boxed{
\text{Because }d_\psi\text{-certification is structurally unsound, the certificate must discharge full }\mathcal D.
}
$$

If the threshold-landing regime is weak on a particular checkpoint, the main deterministic certificate still stands. Theorem S is not a kill gate for the project; it is the clean explanation of why the learned-adversary shortcut cannot be the certificate.

### C4. High-Dimensional Conformal Feasibility-Survival Extension (Theorem F)

The deterministic certificate is the strong result, but it can become impractical or vacuous at deployment dimension because it composes $f$ through learned networks and verifies universal conditions over $(x,u,d)$ cells. The conformal lane is the principled extension when exact verification exits.

Freeze the deployed closed loop:

$$
\mathfrak C_\theta
:=
\left(
V_\theta,Q_\theta,\pi^\flat_\varphi,u^\star,
\widehat{\mathcal D}_{\mathrm{dep}},
\widehat{\mathcal D}_{\mathrm{cal}},
\text{environment/opponent policy}
\right).
$$

The calibration-time set satisfies

$$
\widehat{\mathcal D}_{\mathrm{dep}}
\subseteq
\widehat{\mathcal D}_{\mathrm{cal}}
\subseteq
\mathcal D.
$$

The trajectory score is

$$
S_{\mathrm{feas}}(\tau)
:=
\max_{0\le t<T}
\left[
\beta(V_\theta(x_t))
-
\sup_{u\in\mathcal U}
\inf_{\xi\in\widehat{\mathcal D}_{\mathrm{cal}}(x_t)}
Q_\theta(x_t,u,\xi)
\right]_+ .
$$

The fallback-pinned sufficient score is

$$
S_{\flat}(\tau)
:=
\max_{0\le t<T}
\left[
\beta(V_\theta(x_t))
-
\inf_{\xi\in\widehat{\mathcal D}_{\mathrm{cal}}(x_t)}
Q_\theta(x_t,\pi^\flat_\varphi(x_t),\xi)
\right]_+ .
$$

For exchangeable calibration rollouts $\tau_1,\ldots,\tau_N$ from the frozen deployment distribution, split conformal gives

$$
\Pr_{\tau\sim\mathbb P_{\mathrm{deploy}}}
\left[
S(\tau)\le q_\alpha
\right]
\ge
1-\alpha.
$$

If $S=S_{\mathrm{feas}}$ and $q_\alpha=0$, then

$$
\Pr_{\tau\sim\mathbb P_{\mathrm{deploy}}}
\left[
\mathrm{Feas}_{\mathrm{dep}}(x_t)\ \forall\,0\le t<T
\right]
\ge
1-\alpha.
$$

Honest reading:

- If $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$, this is a true-robust feasibility-survival certificate.
- If $\widehat{\mathcal D}_{\mathrm{cal}}\subsetneq\mathcal D$, this certifies the deployed/calibration feasible event, not full robust safety.
- The theorem is standard split conformal; the paper's novelty is the certified object and the score's connection to robust $Q$-CBF feasibility, not conformal prediction itself.

---

## 4. Methodology

### 4.1 Training / Synthesis

Training is not the certificate. It produces the frozen object to be verified:

$$
(V_\theta,Q_\theta,\pi^\flat_\varphi).
$$

Useful training losses may improve non-vacuity, for example by encouraging witness margin

$$
\min_{d\in\mathcal D}
Q_\theta(x,\pi^\flat_\varphi(x),d)
-
\beta(V_\theta(x))
\ge
m,
$$

but these losses are never proof assumptions. The only soundness path is direct verification of (C1)--(C3).

Report:

- architecture and activation class for $V_\theta,Q_\theta,\pi^\flat_\varphi$;
- the deployed decay $\beta$ and certified threshold $c$;
- the disturbance set $\mathcal D$ or augmented set $\mathcal D_{\mathrm{aug}}$;
- whether the verified $\pi^\flat_\varphi$ is learned, distilled, or optimized offline and then frozen.

### 4.2 Deterministic Verification Protocol

The deterministic protocol is the central experiment-method loop:

1. Freeze $(V_\theta,Q_\theta,\pi^\flat_\varphi)$ and the dynamics/uncertainty model.
2. Choose candidate $c$ and $\varepsilon$.
3. Verify (C1): $\Omega_c\subseteq\mathcal K$.
4. Verify (C3): the witness is feasible over all $d\in\mathcal D$.
5. Verify (C2): every feasible $u$ sends every $d\in\mathcal D$ successor back into $\Omega_c$.
6. If verification fails or is inconclusive, shrink/refine using $\mathcal T_\Phi$ and report uncertified cells honestly.
7. Report the certified set size:
   $$
   \frac{\mathrm{Vol}(\Omega_c^{\mathrm{cert}})}{\mathrm{Vol}(\Omega^\star)}
   $$
   where grid-HJI ground truth is available, or a task-relevant coverage proxy otherwise.

The key binary soundness metric is:

$$
\text{certified-but-violated states under full }\mathcal D = 0.
$$

### 4.3 Theorem S / False-Feasibility Diagnostic

Run the false-feasibility diagnostic to justify why learned-adversary certification is insufficient.

Compute or approximate:

$$
q_{\min}(x,u)=\min_{d\in\mathcal D}Q_\theta(x,u,d),
\qquad
q_\psi(x,u)=Q_\theta(x,u,d_\psi(x,u)).
$$

Report states/actions satisfying

$$
q_{\min}(x,u)
<
\beta(V_\theta(x))
<
q_\psi(x,u),
$$

and summarize with

$$
\mathrm{FFR}
:=
\Pr_{(x,u)}
\left[
q_{\min}(x,u)
<
\beta(V_\theta(x))
<
q_\psi(x,u)
\right].
$$

This is not the main result. It is a diagnostic / motivation figure showing the shortcut that Theorem A avoids.

### 4.4 Conformal Protocol

Use the conformal lane only after the frozen loop is fixed.

Forbidden leakage:

- calibration rollouts cannot tune $V_\theta,Q_\theta,\pi^\flat_\varphi$;
- calibration rollouts cannot select $\beta,c,\widehat{\mathcal D}_{\mathrm{cal}}$;
- calibration rollouts cannot choose the score variant;
- calibration rollouts cannot tune opponent/environment policies.

If the inner $\inf_\xi$ or outer $\sup_u$ is approximate, state the certificate as a certificate for the **computed score** unless the optimizer is itself conservatively bounded.

### 4.5 Strategic / Performative Drift

Distribution shift is a stress test, not a theorem. The conformal guarantee requires exchangeability:

$$
\tau_1,\ldots,\tau_N,\tau
\quad\text{exchangeable under the same frozen closed loop.}
$$

If the opponent or environment adapts after calibration, the guarantee no longer applies. Report coverage degradation under

$$
\mathbb P_1\ne\mathbb P_0
$$

as empirical robustness only.

---

## 5. Experiments

### E0. Deterministic Certificate on Toy / F1TENTH

**Purpose:** demonstrate the main theorem on a system where full-$\mathcal D$ verification is affordable and auditable.

Report:

- certified set $\Omega_c^{\mathrm{cert}}$;
- $\mathrm{Vol}(\Omega_c^{\mathrm{cert}})/\mathrm{Vol}(\Omega^\star)$ where grid-HJI is available;
- verification times and branch counts for (C1), (C2), (C3);
- zero certified-but-violated states under dense/random/adversarial full-$\mathcal D$ falsification;
- ablation showing what fails if (C3) or full-$\mathcal D$ discharge is replaced by learned-adversary checking.

This is the main evidence that Theorem A is not vacuous.

### E1. False-Feasibility Diagnostic

**Purpose:** show why learned-adversary feasibility is not enough.

Use the existing Theorem S / FFR machinery to show:

$$
\Phi_\psi(x,u)=1,
\qquad
\Phi_{\mathrm{rob}}(x,u)=0,
$$

or equivalently

$$
q_{\min}(x,u)
<
\beta(V_\theta(x))
<
q_\psi(x,u).
$$

Success is not "large FFR at all costs." Success is a clean, refinement-stable demonstration that the shortcut can admit actions the full-$\mathcal D$ certificate rejects.

### E2. F1TENTH Hardware or High-Fidelity Simulation

**Purpose:** show the certified filter has operational meaning.

Run certified and uncertified variants:

- deployed learned-adversary filter;
- full-$\mathcal D$ deterministic certified filter where affordable;
- fallback-pinned certified filter;
- task controller without safety filter;
- robust RL / domain randomization baseline without certificate.

Report:

$$
\text{intervention rate},\quad
\text{minimum boundary margin},\quad
\text{fallback activations},\quad
\text{violations under stress}.
$$

### E3. High-Dimensional Deterministic-Exit Contrast

**Purpose:** justify why conformal is needed beyond the deterministic core.

Do not assert that full-$\mathcal D$ sound verification is impossible. Measure it. As dimension or network size grows, plot:

$$
\underbrace{
\text{full-}\mathcal D\text{ deterministic verifier: runtime}\uparrow,\ \text{bound looseness}\uparrow,\ \Omega_c\to\varnothing
}_{\text{measured deterministic exit}}
$$

against

$$
\underbrace{
S_{\mathrm{feas}}\text{ or }S_{\flat}\text{ calibration: runs with non-vacuous }q_\alpha
}_{\text{conformal extension}}.
$$

This answers the reviewer who says: "If Theorem A is sound, why not use it everywhere?"

### E4. Conformal Feasibility-Survival Certificate

**Purpose:** show Theorem F gives useful finite-sample deployment assurance when deterministic verification exits.

Report:

$$
q_\alpha,\qquad
\widehat{\mathrm{Cov}},\qquad
\text{score histograms},\qquad
\text{margin scale}.
$$

The strong operating point is

$$
q_\alpha=0.
$$

If $q_\alpha>0$, report the certified margin shortfall and whether it is operationally tolerable.

### E5. Low-Dimensional Cross-Validation of the Conformal Score

**Purpose:** make the high-dimensional conformal reading credible.

On toy / F1TENTH, where full-$\mathcal D$ ground truth exists, compare $S_{\mathrm{feas}}$ and $S_{\flat}$ against native robust-feasibility failures. Show they outperform generic collision-only, value-shortfall, or learned-adversary-only scores.

This is the "meet in the middle" experiment:

$$
\text{deterministic ground truth}
\quad\Longleftrightarrow\quad
\text{conformal score meaning}.
$$

---

## 6. Go / No-Go Gates

### Gate 1: Non-Vacuous Deterministic Certificate

The project's core depends on showing $\Omega_c^{\mathrm{cert}}$ is not empty and not trivial.

Minimum success:

$$
\mathrm{Vol}(\Omega_c^{\mathrm{cert}})>0
$$

and, where $\Omega^\star$ is available,

$$
\frac{\mathrm{Vol}(\Omega_c^{\mathrm{cert}})}{\mathrm{Vol}(\Omega^\star)}
$$

is large enough to matter for the task.

If this fails, the paper becomes a theory/framework paper with a vacuity problem.

### Gate 2: Direct Verification Beats Ideal-Object Assumptions

The paper must repeatedly demonstrate that soundness comes from direct verification of $Q_\theta,V_\theta$, not from assuming ideal approximation. Explicitly report:

$$
\text{no }\|Q_\theta-Q^\star\|\text{ assumption used},
\qquad
\text{no }V_\theta=V^\star\text{ assumption used}.
$$

### Gate 3: Learned-Adversary Shortcut Is Shown Unsafe or Insufficient

Theorem S is proved regardless of experiments, but at least one diagnostic should show the practical shortcut can be optimistic:

$$
\mathcal F_\psi\setminus\mathcal F_{\mathrm{rob}}\ne\varnothing.
$$

If this is weak on a given checkpoint, do not let it derail the core certificate. Present it as a structural motivation and use a controlled example if needed.

### Gate 4: High-Dimensional Extension Is Non-Vacuous

The conformal lane must not collapse to a useless bound. Report whether

$$
q_\alpha=0
$$

or, if not, whether $q_\alpha$ is small relative to the feasibility-margin scale.

---

## 7. Reviewer Attacks and Rebuttals

- **"Isn't Theorem A just verifying a neural network with CROWN?"**  
  No. The verifier backend may use CROWN-style relaxations, but the object and proof obligation are different: deployed robust state-action $Q$-CBF recursive feasibility, all-feasible robust transition, witness feasibility, and full-$\mathcal D$ inner-game discharge. Lemma E / Prop T2 / T6 are the problem-specific encoding.

- **"Why not just rely on the learned adversary?"**  
  Proposition U gives $\mathcal F_{\mathrm{rob}}\subseteq\mathcal F_\psi$, so learned-adversary feasibility is an over-approximation. Theorem S shows that when worst-case selectors switch, a continuous $d_\psi$ can structurally miss the full-$\mathcal D$ worst case. This is exactly why the deterministic certificate verifies over $\mathcal D$.

- **"Are you assuming $Q_\theta\approx Q^\star$?"**  
  No. Theorem A never uses $Q^\star$ or an approximation bound. It checks inequalities on the deployed $Q_\theta,V_\theta$ directly. Bad training reduces $\Omega_c$; it does not create false soundness.

- **"Strong verification will not scale."**  
  Correct in many deployment regimes. That is why E3 measures deterministic exit and Theorem F supplies a high-dimensional conformal extension. The core sound theorem remains valuable because it is the exact certificate wherever full-$\mathcal D$ discharge is affordable and the ground-truth validator for the high-dimensional score.

- **"The conformal theorem is standard."**  
  Yes. The conformal step is intentionally standard. The novelty is the deployed robust $Q$-CBF object, the sound deterministic certificate, and the object-native score; conformal is the high-dimensional deployment wrapper.

- **"This is not fully model-free."**  
  The deterministic certificate is model-bounded. Synthesis and runtime can be model-free, but (C2) must certify the true transition or an uncertainty-augmented model. A fully model-free deterministic invariance certificate is impossible in the worst case; the conformal lane is the model-free deployment alternative.

---

## 8. Venue and Impact

The paper's impact should be stated as:

$$
\boxed{
\text{the first sound post-hoc certification framework for deployed learned robust state-action }Q\text{-CBF filters}
}
$$

not as a generic conformal-safety or false-feasibility paper.

---

## 9. Collaboration and Asset Plan

Coordinate with the home line on:

- available $V_\theta,Q_\theta,\pi^\flat_\varphi$ checkpoints;
- whether checkpoints expose learned-adversary false feasibility;
- which low-dimensional platform gives the cleanest full-$\mathcal D$ certificate;
- which high-dimensional environment best demonstrates deterministic verifier exit;
- whether model-error bounds can support a deterministic $\mathcal D_{\mathrm{aug}}$ or whether that lane must be conformal.

The contribution split should be explicit:

$$
\text{home line}
=
\text{synthesis of robust state-action }Q\text{-CBF},
$$

$$
\text{this project}
=
\text{sound certification of the deployed learned robust }Q\text{-CBF filter}.
$$

Theorem S / false-feasibility diagnostics support this split by showing why synthesis with a learned adversary is not, by itself, a sound certificate.

---

## 10. Version Log

- **v9 - sound-certificate reframe.**
  - Promoted Theorem A and its full proof chain to the project spine: direct verification of $(V_\theta,Q_\theta,\pi^\flat_\varphi)$, (C1)--(C3), recursive feasibility, robust forward invariance, and safety.
  - Made explicit that no $Q_\theta\approx Q^\star$ or $V_\theta\approx V^\star$ hypothesis is used. The ideal $V^\star,Q^\star$ chain is lineage; soundness comes from verified inequalities on the deployed networks.
  - Promoted the verifier machinery (T6 / Lemma W, Prop T2, Lemma E, Cor. M, Prop T4) from background to core technical contribution.
  - Repositioned Theorem S / Cor. S3 as the structural reason learned-adversary certification is unsound and full-$\mathcal D$ discharge is necessary, not as the headline contribution.
  - Repositioned split conformal (Theorem F) as the high-dimensional feasibility-survival extension and cross-validated score lane, not as the main certificate.
  - Rewrote experiments and gates around non-vacuous deterministic certification first, false-feasibility diagnostics second, and conformal deployment extension third.

- **v8 - Story B unification + evidentiary tightening.**
  - Unified the optimistic deployed set $\widehat{\mathcal D}_{\mathrm{dep}}$, calibration set $\widehat{\mathcal D}_{\mathrm{cal}}$, and full set $\mathcal D$ around the false-feasibility gap. Useful for the conformal lane, but too dominant for the full project spine.

- **v7 - false-feasibility + conformal deployment reframe.**
  - Replaced deterministic-verifier-first framing with Theorem S, $S_{\mathrm{feas}}$, and split conformal as one spine. Superseded by v9 because it underweighted the sound deterministic certificate.

- **v6 and earlier - deterministic verifier line.**
  - Sound deployed-object certificate, Theorem A, witness compression, full-$\mathcal D$ bounding, monotone refinement, and F1TENTH platform. v9 restores this as the core, with sharper deployed-network and full-$\mathcal D$ emphasis.

---

*Standing reminders: (1) The main theorem is Theorem A, not Theorem S or Theorem F. (2) The certificate is on $(V_\theta,Q_\theta,\pi^\flat_\varphi)$ directly; never imply an unverified $Q_\theta\approx Q^\star$ assumption. (3) Robust means full-$\mathcal D$ unless explicitly in the conformal deployed/calibration lane. (4) Theorem S explains why the learned-adversary shortcut is unsound; keep it, but do not let FFR become the paper. (5) Conformal is valuable high-dimensional deployment assurance, but it is the extension, not the core. (6) Non-vacuity of $\Omega_c^{\mathrm{cert}}$ is the central empirical burden.*
