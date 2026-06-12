# False-Feasibility and Deployment Certificates for Learned Robust Q-CBF Filters
### Theoretical Core — full derivation and deployment-assurance lane

*A single chain, derived step by step at the granularity of the source papers: from Hamilton–Jacobi–Isaacs reachability, through the control-barrier and control-barrier-value-function constructions and the continuous-to-discrete bridge, to the deployed-object robustness certificate of this work. Prior results are reproduced as the lemmas we build on (with explicit equation-level proofs), not summarized. Steps that are still open are flagged `[OPEN]`; proof obligations not yet discharged are flagged `[TODO]`.*

> **v7 dependency map (read first).** The current project has one spine: **false-feasibility certification for deployed robust state-action $Q$-CBF filters**. The main theory hook is **Theorem S** and Corollary S3: continuous plug-in adversaries can structurally miss switching worst cases and thereby create false feasibility. The scalable deployment certificate is the split-conformal feasibility-survival result in **§5.11**. The old deterministic certificate — **Theorem A** and its hypotheses (C1)–(C3) — is still proved and useful, but its role is now the **low-dimensional diagnostic / ground-truth lane** for F1TENTH and toy systems, not the main high-dimensional method.
>
> **Story B notation contract (v8 — read with the framework).** Three disturbance sets are kept distinct everywhere below:
> $$
> \widehat{\mathcal D}_{\mathrm{dep}}(x)\ \subseteq\ \widehat{\mathcal D}_{\mathrm{cal}}(x)\ \subseteq\ \mathcal D ,
> $$
> where $\widehat{\mathcal D}_{\mathrm{dep}}$ is the **deployed (optimistic)** set the runtime filter uses (a learned adversary $d_\psi$, or a small ball), $\widehat{\mathcal D}_{\mathrm{cal}}$ is the **calibration-time** set the score is evaluated against offline (the richest affordable worst case), and $\mathcal D$ is the **true full** set. The deployed predicate $\Phi_\theta$ (§5.1, §5.11) is the *optimistic* one $\Phi_\psi$ when $\widehat{\mathcal D}_{\mathrm{dep}}=\{d_\psi\}$. The **deployment certificate (Theorem F) certifies survival of the deployed/calibration feasible set $\mathcal F_\psi$, not true robust safety against $\mathcal D$**; it becomes a true-robustness statement **iff** $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$ — the low-dimensional lane, where Theorem A and full-$\mathcal D$ discharge are affordable. The object linking the necessity result and the certificate is the **false-feasibility gap** $\mathcal F_\psi\setminus\mathcal F_{\mathrm{rob}}$: Theorem S / Cor. S3 prove it can have positive measure; the low-dimensional lane (§5.2–§5.10) measures it against ground truth; the high-dimensional regime can only *validate* (not certify) that the certificate is tracking it, via an offline partial worst-case probe (§5.11, Remark F4). This is **Story B**: certify the deployed object, and separately characterize — structurally and empirically — how optimistic it is.
>
> **Load-bearing vs. lineage.** The deterministic deployed certificate — **Theorem A** and (C1)–(C3) — rests *only* on the **discrete-time** results: the robust-DCBF forward invariance of §2.5, the maximal-robust-Q-CBF backbone of §3, and the verification machinery of §5. Everything in §1.2 (the HJI PDE/VI), §2.1–§2.4 (continuous CBF/CBVF and the continuous-to-discrete bridge) is **origin material**: it explains *where* the discrete objects come from and inherits the standard assumptions of HJ reachability and sampled-data CBF (viscosity solutions, a.e. differentiability, inter-sample hold). These origin sections are flagged inline wherever they carry an undischarged standard assumption; **none** of those assumptions is a proof obligation for Theorem A. When auditing deterministic rigor, the airtight set to scrutinize is **§2.5 + §3 + §5.1–§5.10**. When auditing the new deployment-assurance guarantee, scrutinize **§5.11** and the exchangeability / leakage-free freeze assumptions.

---

## 0. Notation and standing assumptions

| Symbol | Meaning |
|---|---|
| $x\in\mathcal X\subseteq\mathbb R^{n}$ | state |
| $u\in\mathcal U\subseteq\mathbb R^{m}$ | control input; $\mathcal U$ compact |
| $d\in\mathcal D\subseteq\mathbb R^{p}$ | disturbance; $\mathcal D$ **compact**, the *entire* admissible set |
| $f:\mathcal X\times\mathcal U\times\mathcal D\to\mathcal X$ | discrete-time dynamics, $x_{t+1}=f(x_t,u_t,d_t)$; **known** in the deterministic lane (§5.8 relaxes) |
| $F:\mathcal X\times\mathcal U\times\mathcal D\to\mathbb R^{n}$ | continuous-time field, $\dot x=F(x,u,d)$ (used only in §1–§2 for the HJ/CBVF origin) |
| $g:\mathcal X\to\mathbb R$ | safety margin; **constraint (safe) set** $\mathcal K:=\{x:g(x)\ge 0\}$; **failure set** $\mathcal F:=\{x:g(x)<0\}$ |
| $\mathbf d=(d_0,d_1,\dots)\in\mathcal D^{\infty}$ | disturbance signal |

**Sign convention (Fisac/Oh).** The safety value is **non-negative on safe states**; "stay safe" $\Leftrightarrow$ "keep the value $\ge 0$." (CMU's verifier uses the opposite sign — safe $\Leftrightarrow Q\le 0$; we note the flip wherever their result is invoked.) Everything is **discrete-time** and **avoid-only** unless stated.

**Comparison ($\mathcal K$/$\mathcal{KL}$) functions.** A continuous $\alpha:[0,a)\to[0,\infty)$ is **class $\mathcal K$** if it is strictly increasing with $\alpha(0)=0$. A continuous $\beta:[0,a)\times[0,\infty)\to[0,\infty)$ is **class $\mathcal{KL}$** if $\beta(\cdot,t)$ is class $\mathcal K$ for each fixed $t$ and $\beta(r,\cdot)$ is decreasing with $\beta(r,t)\to0$ as $t\to\infty$. Throughout, the **decay function** $\beta:\mathbb R_{\ge0}\to\mathbb R_{\ge0}$ is class $\mathcal K$ with the additional property
$$
\beta(r)\le r\qquad\forall r\ge 0 .
\tag{0.1}
$$
The canonical linear instance is $\beta(r)=(1-\alpha)\,r$ with $\alpha\in(0,1]$; HCSF writes this as $\gamma:=1-\alpha$, i.e. $\beta(r)=\gamma r$.

---

## 1. Hamilton–Jacobi–Isaacs reachability: the safety value function

### 1.1 Continuous-time robust safety as a differential game

Fix the continuous-time uncertain system $\dot x=F(x,u,d)$. We want the set of states from which **some** control can keep the trajectory in $\mathcal K$ **for all time** against **every** admissible disturbance. To make "every disturbance, reacting to control" precise, the disturbance is granted the instantaneous informational advantage through a **nonanticipative strategy** $\mathsf d:\mathcal U\to\mathcal D$ (it may react to the current control but not to its future); $\Xi$ denotes the set of such strategies.

The **safety value function** measures the worst-case-over-time margin under best control against the worst reacting disturbance:
$$
V(x)\;=\;\min_{\mathsf d\in\Xi}\;\max_{\mathsf u}\;\inf_{s\ge 0}\;g\!\left(\xi^{\,\mathsf u,\mathsf d}_{x}(s)\right),
\tag{1.1}
$$
where $\xi^{\,\mathsf u,\mathsf d}_{x}(\cdot)$ is the trajectory from $x$. The interpretation is direct: $V(x)\ge 0$ iff the controller can keep $g\ge0$ along the whole trajectory whatever the disturbance does. The corresponding set is studied in §1.4.

### 1.2 The HJI characterization (PDE / variational inequality)

$V$ in (1.1) is the (viscosity) solution of a Hamilton–Jacobi–Isaacs equation. Two equivalent stationary forms are used in the literature; writing the **Hamiltonian:** Derivitive of the system, equal to how agent's safety changes under control and disturbance.
$$
\mathcal H^{\star}(x)\;:=\;\max_{u\in\mathcal U}\;\min_{d\in\mathcal D}\;\big\langle \nabla V(x),\,F(x,u,d)\big\rangle ,
$$
the **single-boundary PDE** forces contraction,
$$
\min\big\{\,0,\;\mathcal H^{\star}(x)\big\}=0,
\tag{1.2}
$$
and the **variational inequality (VI)** keeps the trajectory inside $\mathcal K$,
$$
\min\big\{\,g(x)-V(x),\;\mathcal H^{\star}(x)\big\}=0 .
\tag{1.3}
$$
With the natural boundary data these coincide. *(Standalone, (1.2) reduces to $\mathcal H^{\star}\ge0$ everywhere and drops $g$ entirely; it matches (1.3) only once the obstacle/boundary data that re-injects $g$ is supplied. (1.3) is the self-contained statement and the one we actually use.)* Equation (1.3) is the continuous-time template we will see again, in three forms: as the CBVF (§2.3), as the discrete Isaacs equation (§1.3), and as the refinement operator (§5.9).

### 1.3 Discrete time: the safety Bellman–Isaacs equation

The object of this project is the **discrete-time** system $x_{t+1}=f(x_t,u_t,d_t)$. The discrete analog of (1.1) is
$$
V^{\star}(x)\;=\;\max_{\mathsf u}\;\min_{\mathbf d\in\mathcal D^{\infty}}\;\inf_{t\ge 0}\;g(x_t),
\qquad x_0=x,
\tag{1.4}
$$
where $\mathsf u$ is the controller's policy and $\mathbf d=(d_0,d_1,\ldots)$ is the disturbance signal. The quantity $\inf_{t\ge0} g(x_t)$ is the worst safety margin encountered along the whole future trajectory; hence $V^{\star}(x)\ge0$ means that the controller can keep $g(x_t)\ge0$ for all time against the worst admissible disturbance.

The dynamic-programming step is a one-step decomposition of this infinite-horizon statement. Split the worst-over-time margin into the present time and everything after the first transition:
$$
\inf_{t\ge0} g(x_t)
\;=\;
\min\Big\{\,g(x_0),\;\inf_{t\ge1} g(x_t)\Big\}.
\tag{1.4a}
$$
Since $x_0=x$ and, after choosing $u$ and seeing the worst disturbance $d$, the next state is $x_1=f(x,u,d)$, the tail problem starting at $t=1$ is the same safety problem restarted from $x_1$. Its optimal robust value is therefore $V^{\star}(f(x,u,d))$. Optimizing this one-step tail gives the **safety Bellman--Isaacs equation**
$$
\boxed{\;
V^{\star}(x)=\min\Big\{\,g(x),\;\max_{u\in\mathcal U}\min_{d\in\mathcal D}\,V^{\star}\!\big(f(x,u,d)\big)\Big\}.
\;}
\tag{1.5}
$$
Reading (1.5): you are *exactly* as safe as the worse of (i) your present margin $g(x)$ and (ii) the best worst-case future you can still secure. The outer $\min\{\cdot,\cdot\}$ appears because safety is not an accumulated reward; one bad time is enough to determine the trajectory margin. The per-step ordering $\max_u\min_d$ grants the disturbance the one-step informational advantage -- commit $u$, then take the worst $d$ -- which is the discrete mirror of the nonanticipative $\Xi$ and is therefore the **conservative**, hence sound, choice.

This is the Bellman principle of optimality in safety form. Bellman's idea is not a new safety assumption; it is a self-consistency property of an optimal value function. If the first step takes the system to $x_1$, then the remaining tail policy should itself be optimal for the same problem starting from $x_1$. If it were not optimal, replacing only the tail by a better tail policy would improve the original policy, contradicting optimality. Thus an infinite-horizon problem can be written as a one-step backup plus the value-to-go:
$$
\text{value at }x
\;=\;
\text{aggregate}\Big(\text{current margin at }x,\;\text{optimal value-to-go from }x_1\Big).
$$
For ordinary discounted reward maximization this aggregation is additive,
$$
J^{\star}(x)
=
\max_{u}\Big\{r(x,u)+\gamma J^{\star}(f(x,u))\Big\}.
$$
For safety reachability the trajectory score is the worst margin over time, so the aggregation is instead a minimum:
$$
V^{\star}(x)
=
\min\Big\{g(x),\;\text{robust value-to-go after one step}\Big\}.
$$
The Isaacs part is the zero-sum game backup inside that value-to-go: the controller chooses $u$ to maximize safety, while the disturbance chooses $d$ to minimize it.

> **Operator-order note (no inconsistency with (1.1)).** The outer operator visibly *flips* between (1.1) and (1.4): continuous time carries the disturbance's advantage in the **strategy class** $\Xi$ (a nonanticipative $\mathsf d$ reacting to $\mathsf u$), so the disturbance-favorable value is the lower value $\min_{\mathsf d\in\Xi}\max_{\mathsf u}$; discrete time carries the *same* advantage in the **per-step** ordering $\max_u\min_d$ (commit $u$, then worst $d$), with $\mathbf d$ a signal. The two forms disagree on the surface yet both encode the conservative, disturbance-advantaged value — the discrepancy is the strategy-vs-signal encoding, not an error.

**Constructive solution (used as ground truth).** Define the operator
$$
(\mathcal B V)(x):=\min\Big\{g(x),\,\max_u\min_d V\big(f(x,u,d)\big)\Big\},
\qquad V^{(0)}:=g,\quad V^{(k+1)}:=\mathcal B V^{(k)} .
\tag{1.6}
$$
Here $\mathcal B$ is a one-step safety backup: given a candidate future-value function $V$, it computes the present value obtained by checking the current margin $g(x)$ and then backing up the best robust value of the next state. Starting from $V^{(0)}(x)=g(x)$, the iterates have a finite-horizon interpretation: $V^{(1)}$ certifies safety for the present and one future step, $V^{(2)}$ for two future steps, and so on. Each backup can only reveal additional future failure modes, because the outer $\min\{g,V\}$ never allows the value to become more optimistic than the present margin. Thus the sequence is monotone non-increasing and converges, under the standard dynamic-programming assumptions, to the infinite-horizon value $V^{\star}$.

Equivalently, the iteration can be read as expanding the safety horizon one step at a time:
$$
V^{(k)}(x)
=
\max_{\mathsf u}
\min_{\mathbf d}
\min_{0\le t\le k} g(x_t),
\qquad x_0=x,
\tag{1.6a}
$$
with the disturbance sequence truncated to the horizon $0,\ldots,k$. The base case $k=0$ gives $V^{(0)}=g$. Applying one Bellman backup replaces the tail after the first transition by the already-computed $k$-step value, producing the $(k+1)$-step value. In the limit $k\to\infty$, the finite-horizon values converge to the infinite-horizon safety value in (1.4).

The fixed-point condition
$$
\mathcal B V^{\star}=V^{\star}
\tag{1.6b}
$$
means that the value function is self-consistent: after all future danger has already been propagated backward through the dynamics, one more Bellman--Isaacs backup changes nothing. If $g(x)<0$, then $(\mathcal B V)(x)\le g(x)<0$ for every $V$, so an already failed state remains failed; if $g(x)\ge0$ but every admissible control can eventually be driven to failure by some disturbance sequence, repeated backups propagate that future failure backward until $V^{\star}(x)<0$. This is what grid-HJI / OptimizedDP computes on a low-dimensional mesh, and what we use to obtain the reference $\Omega^{\star}$ in §6.

### 1.4 Two structural consequences and the maximal robust safe set

Directly from the outer minimum in (1.5). To avoid a common notational confusion, the two appearances of $V^{\star}$ below are the **same function** evaluated at different states: $V^{\star}(x)$ is the value of the current state, while $V^{\star}(f(x,u,d))$ is the value of the next state reached after applying $u$ and disturbance $d$.

Define the robust one-step future term
$$
A(x)
:=
\max_{u\in\mathcal U}\min_{d\in\mathcal D}
V^{\star}\!\big(f(x,u,d)\big).
\tag{1.6c}
$$
Then (1.5) is simply
$$
V^{\star}(x)=\min\{\,g(x),A(x)\,\}.
\tag{1.6d}
$$
For any two scalars $a,b$, if $y=\min\{a,b\}$, then $y\le a$ and $y\le b$. Applying this elementary fact with $y=V^{\star}(x)$, $a=g(x)$, and $b=A(x)$ gives:

> **(P1)** $\;V^{\star}(x)\le g(x)\quad\forall x\in\mathcal X.$ &nbsp;&nbsp;*(the value never exceeds the present margin)*
>
> **(P2)** $\;\displaystyle\max_{u}\min_{d}V^{\star}\!\big(f(x,u,d)\big)\;\ge\;V^{\star}(x)\quad\forall x\in\mathcal X.$ &nbsp;&nbsp;*(the future term is $\ge$ the min of the two terms)*

Interpretation of (P1): $g(x)$ is the instantaneous safety margin, while $V^{\star}(x)$ is the worst margin over the whole future trajectory under optimal robust play. Since the present time is included in that trajectory, the long-run worst margin cannot be larger than the present margin:
$$
V^{\star}(x)\le g(x).
$$
Thus $g(x)>V^{\star}(x)$ is not paradoxical; it means the state may look safe now, but future dynamics or disturbances make the robust long-run margin smaller.

Interpretation of (P2): the robustly best next-state value is at least the current value,
$$
\max_{u}\min_{d}V^{\star}\!\big(f(x,u,d)\big)\ge V^{\star}(x).
$$
When the maximum is attained by some $u^{\star}(x)$, this implies
$$
\forall d\in\mathcal D:\qquad
V^{\star}\!\big(f(x,u^{\star}(x),d)\big)\ge V^{\star}(x).
\tag{1.6e}
$$
This is the control-invariance content hidden inside the Bellman fixed point: there is a control action that prevents the value from decreasing under any admissible one-step disturbance.

Define the **maximal robust safe (control-invariant) set**
$$
\Omega^{\star}\;:=\;\{x:\,V^{\star}(x)\ge 0\}.
\tag{1.7}
$$

> **Fact 1 ($\Omega^{\star}\subseteq\mathcal K$).**
> If $x\in\Omega^{\star}$ then $V^{\star}(x)\ge 0$, and by (P1) $g(x)\ge V^{\star}(x)\ge 0$, so $x\in\mathcal K$. $\blacksquare$

$\Omega^{\star}$ is the largest set on which control can maintain safety against every $\mathbf d\in\mathcal D^{\infty}$; it is the denominator against which any certified set is measured.

---

## 2. Barrier certificates: from "is $x$ safe?" to "which $u$ is safe?"

$V^{\star}$ answers whether a state *is* safe-controllable. A **barrier certificate** repackages that into a **pointwise constraint on the control**, which is what a runtime filter can evaluate. We build the certificate up in the order it was historically solved: CBF $\to$ robust CBF $\to$ CBVF (the construction) $\to$ the continuous-to-discrete bridge $\to$ robust DCBF (the object we certify).

> **Notation checkpoint ($g$ vs. $V^{\star}$ vs. $h$).**
>
> | object | superlevel set | answers | role |
> |---|---|---|---|
> | $g:\mathcal X\to\mathbb R$ | $\mathcal K=\{x:g(x)\ge0\}$ | "is $x$ safe now?" | primitive constraint / instantaneous safety margin |
> | $V^{\star}:\mathcal X\to\mathbb R$ | $\Omega^{\star}=\{x:V^{\star}(x)\ge0\}$ | "can control keep $g(x_t)\ge0$ forever against all disturbances?" | maximal robust safe value; satisfies $V^{\star}(x)\le g(x)$ |
> | $h:\mathcal X\to\mathbb R$ | $H=\{x:h(x)\ge0\}$ | "which control inputs keep this certified set invariant?" | generic barrier certificate; may be hand-picked, or later chosen as a value function such as $h=V^{\star}$ / $h=B$ |
>
> Thus a certified barrier set is normally compared through $H\subseteq\Omega^{\star}\subseteq\mathcal K$; choosing the value function itself as the barrier is the constructive route to avoid an overly conservative hand-picked $h$.

### 2.1 Continuous-time CBF and Nagumo's condition

For $\dot x=F(x,u)$ (drop $d$ momentarily), let $h:\mathcal X\to\mathbb R$ be $C^{1}$ with superlevel set $H:=\{x:h(x)\ge 0\}\subseteq\mathcal K$, Lie derivative $L_Fh(x,u):=\langle\nabla h(x),F(x,u)\rangle$, and Hamiltonian $L_F^{\star}h(x):=\sup_{u}L_Fh(x,u)$.

> **Nagumo's theorem.** Assuming $\nabla h(x)\ne0$ on $\partial H=\{h=0\}$, $H$ is control-invariant **iff**
> $$
> L_F^{\star}h(x)\ge 0\qquad\forall x\in\partial H.
> \tag{2.1}
> $$

> **Definition 1 (CBF).** $h$ is a **control barrier function** on a set $C\supseteq H$ if there is a class-$\mathcal K$ (extended class-$\mathcal K$) function $\alpha$ with
> $$
> L_F^{\star}h(x)\;\ge\;-\alpha\big(h(x)\big)\qquad\forall x\in C.
> \tag{2.2}
> $$

(2.2) extends Nagumo's boundary condition (2.1) to the interior with a Lyapunov-like rate: the closer to $\partial H$, the smaller the permitted inflow. Any $u$ in the admissible set
$$
G_h(x):=\big\{u\in\mathcal U:\,L_Fh(x,u)+\alpha(h(x))\ge 0\big\}
$$
keeps the system in $H$ for all time (comparison lemma). The **construction problem**: a hand-picked $h$ rarely satisfies (2.2) on all of $H$.

### 2.2 Robust CBF (the inner $\inf$ over $d$)

Restore the disturbance, $\dot x=F(x,u,d)$.

> **Definition 2 (robust CT-CBF; Oh et al. 2026, Def. 1).** $h$ is a **robust CBF** on $\Omega=\{h\ge0\}\subseteq\mathcal K$ if there is a class-$\mathcal K$ $\alpha$ with
> $$
> \sup_{u\in\mathcal U}\;\inf_{d\in\mathcal D}\;\big\langle \nabla h(x),F(x,u,d)\big\rangle\;\ge\;-\alpha\big(h(x)\big)\qquad\forall x\in\Omega .
> \tag{2.3}
> $$

The structural object is the **game** $\sup_u\inf_d$: the guaranteeing control must dominate the **worst** disturbance. Two routes appear downstream. The *compensation* route upper-bounds the disturbance effect by a term $\sigma(x,u)$ and enforces $L_fh+L_gh\,u-\sigma\ge-\alpha(h)$; it needs explicit structure (worst-case bounds, observer error) and is typically conservative ($\Omega\subsetneq\Omega^{\star}$). The *explicit-min* route keeps $\inf_d$ and is less conservative but needs a tractable $\mathcal D$-representation. The certificate of §5 is the discrete, learned, **explicit-$\mathcal D$** route — we keep the full game and discharge $\inf_d$ soundly (§5.7).

### 2.3 CBVF: constructing a valid robust CBF *as a value function*

The construction problem in §2.1–§2.2 is that a hand-picked $h$ may define a plausible-looking set $H=\{h\ge0\}$ but fail the robust CBF inequality (2.3). CBVF reverses the order: start from the primitive safety margin $g$, solve the reachability game that already knows about the dynamics and disturbance, and then use the resulting value as the barrier. In symbols, the desired route is
$$
g
\quad\Longrightarrow\quad
B_{\lambda}
\quad\Longrightarrow\quad
h:=B_{\lambda}.
$$
Because $B_{\lambda}$ is constructed by dynamic programming, its barrier inequality is inherited from the HJI variational inequality rather than checked after the fact.

Fix the continuous-time disturbed system $\dot x=F(x,u,d)$ and the safe set $\mathcal K=\{g\ge0\}$. Choi–Lee–Sreenath–Tomlin–Herbert (CDC 2021) define, for time $t\le0$, decay rate $\lambda\ge0$, and trajectory $\xi^{\,\mathsf u,\mathsf d}_{x,t}$ starting from $x$ at time $t$, the **control barrier-value function**
$$
B_{\lambda}(x,t)\;=\;\min_{\mathsf d\in\Xi}\;\max_{\mathsf u}\;\min_{s\in[t,0]}\;e^{\lambda(s-t)}\,g\!\left(\xi^{\,\mathsf u,\mathsf d}_{x,t}(s)\right).
\tag{2.4}
$$
Read (2.4) from inside out:

- $\min_{s\in[t,0]}$ records the worst safety margin encountered on the whole horizon;
- $\max_{\mathsf u}$ lets the controller choose the policy that makes that worst margin as large as possible;
- $\min_{\mathsf d\in\Xi}$ gives the disturbance its nonanticipative, worst-case strategy;
- $e^{\lambda(s-t)}>0$ changes the rate/magnitude of the value but not the sign of safety.

Therefore
$$
B_{\lambda}(x,t)\ge0
\quad\Longleftrightarrow\quad
\text{the controller can keep }g\!\left(\xi(s)\right)\ge0
\text{ on }[t,0]\text{ against every admissible disturbance strategy.}
$$
At $\lambda=0$, (2.4) is exactly the reachability value (1.1) on a finite horizon. As $|t|\to\infty$, the infinite-horizon $\lambda=0$ CBVF recovers the maximal robust safe set,
$$
\{x:B_0(x)\ge0\}=\Omega^{\star}.
$$

The HJI-VI comes from applying the same one-step / short-time dynamic-programming split as in §1, but in continuous time. Over an infinitesimal interval, either the current obstacle value $g(x)$ is already the active worst margin, or the value is transported by the dynamics. Formally this gives the time-dependent variational inequality
$$
\min\Big\{\,
g(x)-B_{\lambda}(x,t),\;
D_tB_{\lambda}(x,t)
+\max_{u\in\mathcal U}\min_{d\in\mathcal D}
\big\langle\nabla B_{\lambda}(x,t),F(x,u,d)\big\rangle
+\lambda B_{\lambda}(x,t)
\Big\}=0 .
\tag{2.4a}
$$
The first term is the **obstacle condition** $B_{\lambda}\le g$: the worst-over-time safety value cannot be more optimistic than the current margin. The second term is the **game Hamiltonian condition**: away from the obstacle, the controller must be able to move the value according to the robust game $\max_u\min_d$. The $\lambda B_{\lambda}$ term is the continuous-time trace of the exponential factor in (2.4).

At convergence ($D_tB_{\lambda}=0$), the stationary HJI-VI is
$$
\min\Big\{\,
g(x)-B_{\lambda}(x),\;
\max_{u\in\mathcal U}\min_{d\in\mathcal D}
\big\langle \nabla B_{\lambda}(x),F(x,u,d)\big\rangle
+\lambda B_{\lambda}(x)
\Big\}=0 .
\tag{2.5}
$$
The special case $\lambda=0$ reduces to the reachability VI used in §1.2. For $\lambda>0$, (2.5) gives a CBF-type inequality with offline rate $\lambda$.

> **Lemma C (the CBVF is a valid robust CBF; from (2.5)).** Let $B_{\lambda}$ solve (2.5) and let $H_{\lambda}=\{B_{\lambda}\ge0\}$. Then for every online rate $\gamma\ge\lambda$,
> $$
> \sup_{u}\inf_{d}\big\langle\nabla B_{\lambda}(x),F(x,u,d)\big\rangle
> \;\ge\;
> -\lambda B_{\lambda}(x)
> \;\ge\;
> -\gamma B_{\lambda}(x)
> \qquad\forall x\in H_{\lambda}.
> $$
> Hence $h=B_{\lambda}$ is a robust CBF on $H_{\lambda}$ with $\alpha(r)=\gamma r$. In particular, if $\lambda=0$, the value is valid for every $\gamma\ge0$.
> *(Stated at points where $B_{\lambda}$ is differentiable; on the measure-zero non-differentiable set the inequality is read in the viscosity sub/supersolution sense — cf. Remark C2. $\nabla B_{\lambda}$ is interpreted accordingly throughout. `[inherited HJ-reachability regularity; §2 is origin material — see the dependency map]`)*

**Proof.** From $\min\{a,b\}=0$ in (2.5) we read off **both** inequalities:
$$
g(x)-B_{\lambda}(x)\ge 0
\qquad\text{and}\qquad
\max_{u}\min_{d}\big\langle\nabla B_{\lambda}(x),F(x,u,d)\big\rangle
+\lambda B_{\lambda}(x)\ge0 .
\tag{i}
$$
The right inequality of (i) gives
$$
\sup_{u}\inf_{d}\big\langle\nabla B_{\lambda}(x),F(x,u,d)\big\rangle
\ge
-\lambda B_{\lambda}(x).
$$
On $H_{\lambda}$, $B_{\lambda}(x)\ge0$. If $\gamma\ge\lambda$, then
$$
-\lambda B_{\lambda}(x)\ge-\gamma B_{\lambda}(x).
$$
Chaining the two inequalities yields the robust CBF condition (2.3) with $h=B_{\lambda}$ and $\alpha(r)=\gamma r$. $\blacksquare$

The corresponding robust admissible-control set is therefore
$$
G_{B_{\lambda}}^{\gamma}(x)
:=
\left\{
u\in\mathcal U:
\inf_{d\in\mathcal D}
\big\langle\nabla B_{\lambda}(x),F(x,u,d)\big\rangle
+\gamma B_{\lambda}(x)\ge0
\right\}.
\tag{2.5a}
$$
Lemma C says this set is nonempty on $H_{\lambda}$ whenever $\gamma\ge\lambda$ (under the usual compactness/attainment assumptions). This is the precise sense in which the value function is a barrier **by construction**.

Two consequences we reuse:

> **Remark C1 (decouple offline $\lambda$ from online $\gamma$; Tonkens et al. 2026, Prop. 1).** If $B_{\lambda}$ is constructed with offline rate $\lambda$ and enforced online with rate $\gamma\ge\lambda$, control-invariance of $H_{\lambda}$ is preserved, because the admissible set only grows, $G_{B_{\lambda}}^{\gamma}(x)\supseteq G_{B_{\lambda}}^{\lambda}(x)\ne\varnothing$, and on $\partial H_{\lambda}$ ($B_{\lambda}=0$) both rate terms vanish so the sets coincide. Hence one may take $\lambda=0$ (infinite-time CBVF, superlevel set $=\Omega^{\star}$) and pick any $\gamma>0$ online.

> **Remark C2 (CBF $\subsetneq$ CBVF).** Every CBF is a CBVF; the converse fails (a CBVF is differentiable only a.e.). This is why "compute the value function, it is automatically a (robust) CBF" is the constructive escape from the validity-checking problem of §2.1.

### 2.4 The continuous-to-discrete bridge (class-$\mathcal K$ $\alpha$ $\mapsto$ class-$\mathcal{KL}$ $\beta$)

The certificate runs in discrete time, so we convert the continuous rate $\alpha$ into a one-step decay $\beta$ rigorously (Oh et al. 2026, §II-C; Khalil).

**Step 1 — associate a class-$\mathcal{KL}$ flow to $\alpha$.** Let $\alpha$ be class $\mathcal K$ and locally Lipschitz. For each $r\ge0$ define $\beta_{\alpha}(r,\cdot)$ as the solution of the scalar initial-value problem
$$
\dot y=-\alpha(y),\qquad y(0)=r .
\tag{2.6}
$$
Then $\beta_{\alpha}$ is class $\mathcal{KL}$: increasing in $r$, decreasing to $0$ in $t$.

**Step 2 — sample.** Under sampling period $\Delta t$, the continuous robust-CBF condition (2.3) integrates to the **one-step** inequality
$$
h(x_{t+1})\;\ge\;\beta_{\alpha}\big(h(x_t),\,\Delta t\big).
\tag{2.7}
$$
The "integrates to" is the standard sampled-data passage: it requires (2.3) to hold along the *whole* inter-sample interval $[t,t+\Delta t]$ (zero-order hold on $u$), or a Lipschitz-margin tightening of it, after which the comparison lemma against $\dot y=-\alpha(y)$ yields (2.7). `[inherited sampled-data assumption; §2.4 is the CBVF origin and is NOT load-bearing for the discrete certificate of §5 — see the dependency map up front]`

**Step 3 — fix $\Delta t$.** For a fixed $\Delta t$, the map $r\mapsto\beta_{\alpha}(r,\Delta t)$ is itself class $\mathcal K$; we denote it $\beta(r)$. Because $\dot y=-\alpha(y)\le0$ along (2.6), the flow is non-increasing, giving exactly the property (0.1):
$$
\beta(r)=\beta_{\alpha}(r,\Delta t)\le r .
\tag{2.8}
$$

This is the precise origin of the decay function $\beta$ and of the constraint $\beta(r)\le r$ used everywhere below.

### 2.5 Robust discrete-time CBF and forward invariance

> **Definition 3 (robust DCBF; Oh et al. 2026, Def. 2).** $B:\mathcal X\to\mathbb R$ is a **robust discrete-time CBF** for $x_{t+1}=f(x,u,d)$ on $C=\{B\ge0\}\subseteq\mathcal K$ if there is a class-$\mathcal K$ $\beta$ with $\beta(r)\le r$ such that
> $$
> \sup_{u\in\mathcal U}\;\inf_{d\in\mathcal D}\;B\!\big(f(x,u,d)\big)\;\ge\;\beta\big(B(x)\big)\qquad\forall x\in C .
> \tag{2.9}
> $$

The additive HCSF form $\Delta B(x,u)\ge-\alpha B(x)$ is the special case $\beta(r)=(1-\alpha)r=\gamma r$.

> **Theorem 1 (robust forward invariance).** If $B$ is a robust DCBF on $C$, then any controller that at each step selects $u_t$ with $\inf_{d}B(f(x_t,u_t,d))\ge\beta(B(x_t))$ renders $C$ **robustly forward invariant**:
> $$
> x_0\in C\;\Longrightarrow\;x_t\in C\;\;\forall t\ge0,\;\forall\mathbf d\in\mathcal D^{\infty}.
> $$

**Proof.** Induction on $t$.

*Base.* $x_0\in C$ by hypothesis, hence
$$
B(x_0)\ge 0 .
$$

*Inductive step.* Assume $x_t\in C$, i.e. $B(x_t)\ge0$. We chain four one-line facts.

(a) The decay floor is non-negative — $\beta$ is class $\mathcal K$ ($\beta(0)=0$, nondecreasing) and $B(x_t)\ge0$:
$$
\beta\big(B(x_t)\big)\;\ge\;\beta(0)\;=\;0 .
$$
(b) The selected control meets the constraint — by construction of the controller:
$$
\inf_{d\in\mathcal D}B\!\big(f(x_t,u_t,d)\big)\;\ge\;\beta\big(B(x_t)\big).
$$
(c) Worst-case next value is non-negative — combine (b) and (a):
$$
\inf_{d\in\mathcal D}B\!\big(f(x_t,u_t,d)\big)\;\ge\;\beta\big(B(x_t)\big)\;\ge\;0 .
$$
(d) The realized successor is safe — the realized $d_t\in\mathcal D$ is one feasible point of the infimum:
$$
B(x_{t+1})\;=\;B\!\big(f(x_t,u_t,d_t)\big)\;\ge\;\inf_{d\in\mathcal D}B\!\big(f(x_t,u_t,d)\big)\;\ge\;0 ,
$$
so $x_{t+1}\in C$. Since $d_t\in\mathcal D$ was arbitrary, $C$ is robustly forward invariant. $\blacksquare$

This is the payoff: an **infinite-horizon, all-disturbance** safety guarantee discharged by a **one-step pointwise** condition. The remaining sections (i) exhibit a *specific* valid robust DCBF (§3), and (ii) make the condition *machine-checkable on a learned network for the deployed filter* (§5).

---

## 3. The safety value function is the maximal robust DCBF

We now exhibit a *specific* valid robust DCBF — the Isaacs value $V^{\star}$ — and lift it to state–action form so the runtime constraint needs no model. This section reproduces Oh et al. (2026, Lemmas 1–2, Theorem 1, Prop. 1) at the equation level, in our sign convention.

### 3.1 The state–action lift (Q-function)

The formula below is not an extra assumption. It is the Bellman--Isaacs fixed point (1.5) rewritten so that the first control and disturbance are exposed as arguments of a \(Q\)-function.

Start from the trajectory score used throughout §1:
$$
\inf_{t\ge0}g(x_t).
$$
If the present state is \(x_0=x\), and the first-step pair \((u,d)\) is fixed, then the successor is
$$
x_1=f(x,u,d).
$$
The worst safety margin over the whole future splits into the present margin and the tail margin:
$$
\inf_{t\ge0}g(x_t)
=
\min\Big\{g(x_0),\inf_{t\ge1}g(x_t)\Big\}.
$$
The tail beginning at \(t=1\) is the same robust safety problem restarted from \(x_1\). If all future controls are chosen optimally against all future disturbances, its value is
$$
V^{\star}(x_1)
=
V^{\star}\!\big(f(x,u,d)\big).
$$
Therefore, once the first pair \((u,d)\) is fixed, the best robust safety value of the resulting one-step branch is
$$
\min\Big\{g(x),V^{\star}\!\big(f(x,u,d)\big)\Big\}.
$$
This is the safety analogue of the usual reward-form identity
$$
Q^{\star}_{\mathrm{reward}}(x,u)
=
r(x,u)+\gamma V^{\star}(x^+),
$$
except that safety reachability does not add rewards over time. It records the smallest margin ever encountered, so the additive backup is replaced by a minimum backup. The term \(g(x)\) is included because failure at the current state counts immediately; the term \(V^{\star}(f(x,u,d))\) already includes the successor margin \(g(f(x,u,d))\) and all later margins.

> **Definition 4 (safety Q-function).**
> $$
> Q^{\star}(x,u,d)\;:=\;\min\big\{\,g(x),\;V^{\star}\!\big(f(x,u,d)\big)\big\}.
> \tag{3.1}
> $$

After this lift, returning from \(Q^{\star}\) to the state value \(V^{\star}\) simply means optimizing the exposed first step: the controller chooses \(u\) to maximize the guaranteed safety value, and the disturbance chooses \(d\) to minimize it. This gives the same conservative, disturbance-advantaged one-step order as (1.5):
$$
\max_{u\in\mathcal U}\min_{d\in\mathcal D}.
$$

> **Lemma Q (the lift is exact).**
> $$
> V^{\star}(x)\;=\;\max_{u\in\mathcal U}\;\min_{d\in\mathcal D}\;Q^{\star}(x,u,d)\qquad\forall x .
> \tag{3.2}
> $$

**Proof.** Fix \(x\). Define the scalar clipping map
$$
\phi_x(z):=\min\{g(x),z\}.
$$
Then (3.1) is
$$
Q^{\star}(x,u,d)=\phi_x\!\left(V^{\star}(f(x,u,d))\right).
$$
The map \(\phi_x\) is nondecreasing. Hence it commutes with the inner worst-case minimization:
$$
\min_{d\in\mathcal D}\phi_x\!\left(V^{\star}(f(x,u,d))\right)
=
\phi_x\!\left(\min_{d\in\mathcal D}V^{\star}(f(x,u,d))\right),
$$
because applying a nondecreasing scalar map preserves which values are smaller. It also commutes with the outer maximization:
$$
\max_{u\in\mathcal U}\phi_x\!\left(\min_{d\in\mathcal D}V^{\star}(f(x,u,d))\right)
=
\phi_x\!\left(\max_{u\in\mathcal U}\min_{d\in\mathcal D}V^{\star}(f(x,u,d))\right).
$$
Substituting \(\phi_x(z)=\min\{g(x),z\}\) gives
$$
\max_{u}\min_{d}Q^{\star}(x,u,d)
=\max_{u}\min_{d}\,\min\big\{g(x),\,V^{\star}(f(x,u,d))\big\}.
$$
Pulling the same clipping map outside the inner \(\min_d\) and outer \(\max_u\) yields
$$
=\max_{u}\,\min\Big\{g(x),\,\min_{d}V^{\star}(f(x,u,d))\Big\}
=\min\Big\{g(x),\,\max_{u}\min_{d}V^{\star}(f(x,u,d))\Big\}.
$$
The right-hand side is exactly the safety Bellman–Isaacs operator (1.5) evaluated at $V^{\star}$, which equals $V^{\star}(x)$:
$$
=\;V^{\star}(x). \qquad\blacksquare
$$

The runtime value of (3.1)–(3.2) is now visible: \(Q^{\star}\) has already absorbed the one-step model \(f\). Once \(Q^{\star}\) is represented directly, the online filter can evaluate a state--action--disturbance score without composing \(f\) at runtime — this is the HCSF / Q-CBF advantage (Oh et al., RSS 2025).

### 3.2 The safe fallback (witness) policy

> **Definition 5 (fallback policy; Oh et al. 2026, eq. 14).**
> $$
> \pi^{\flat}(x)\;:=\;\arg\max_{u\in\mathcal U}\;\min_{d\in\mathcal D}\;Q^{\star}(x,u,d).
> \tag{3.3}
> $$

By construction $\pi^{\flat}$ attains the outer maximum in (3.2):
$$
\min_{d\in\mathcal D}Q^{\star}\!\big(x,\pi^{\flat}(x),d\big)\;=\;\max_{u}\min_{d}Q^{\star}(x,u,d)\;=\;V^{\star}(x).
\tag{3.4}
$$

### 3.3 Backbone Lemma B1 — the fallback is feasible on $\Omega^{\star}$

> **Lemma B1 (Oh et al. 2026, Lemma 1).** For all $x\in\Omega^{\star}$ and every decay $\beta$ (class $\mathcal K$, $\beta(r)\le r$), there exists $u\in\mathcal U$ with $\min_{d}Q^{\star}(x,u,d)\ge\beta(V^{\star}(x))$ — namely $u=\pi^{\flat}(x)$.

**Proof.** By the attainment identity (3.4),
$$
\min_{d\in\mathcal D}Q^{\star}\!\big(x,\pi^{\flat}(x),d\big)\;=\;V^{\star}(x).
$$
Since $x\in\Omega^{\star}$,
$$
V^{\star}(x)\;\ge\;0 .
$$
Hence, using $\beta(r)\le r$ at $r=V^{\star}(x)$,
$$
\min_{d\in\mathcal D}Q^{\star}\!\big(x,\pi^{\flat}(x),d\big)\;=\;V^{\star}(x)\;\ge\;\beta\big(V^{\star}(x)\big).
$$
Thus $u=\pi^{\flat}(x)$ satisfies the constraint. $\blacksquare$

### 3.4 Backbone Lemma B2 — the Q-CBF constraint equals the $V$-form

> **Lemma B2 (Oh et al. 2026, Lemma 2).** For all $x\in\Omega^{\star}$, all $u\in\mathcal U$, and every decay $\beta$,
> $$
> \min_{d\in\mathcal D}Q^{\star}(x,u,d)\;\ge\;\beta\big(V^{\star}(x)\big)
> \qquad\Longleftrightarrow\qquad
> \inf_{d\in\mathcal D}V^{\star}\!\big(f(x,u,d)\big)\;\ge\;\beta\big(V^{\star}(x)\big).
> $$

**Proof.** Recall the consequence (P1) of the Isaacs equation:
$$
V^{\star}(x)\;\le\;g(x)\qquad\forall x. \tag{P1}
$$

*($\Rightarrow$)* Assume $\min_{d}Q^{\star}(x,u,d)\ge\beta(V^{\star}(x))$. A minimum over $d$ bounded below means every term is, so
$$
Q^{\star}(x,u,d)\;\ge\;\beta\big(V^{\star}(x)\big)\qquad\forall d\in\mathcal D .
$$
Substitute the definition (3.1):
$$
\min\big\{g(x),\,V^{\star}(f(x,u,d))\big\}\;\ge\;\beta\big(V^{\star}(x)\big)\qquad\forall d\in\mathcal D .
$$
A minimum of two terms is $\ge$ a bound only if each term is; in particular the second:
$$
V^{\star}\!\big(f(x,u,d)\big)\;\ge\;\beta\big(V^{\star}(x)\big)\qquad\forall d\in\mathcal D .
$$
Take the infimum over $d$:
$$
\inf_{d\in\mathcal D}V^{\star}\!\big(f(x,u,d)\big)\;\ge\;\beta\big(V^{\star}(x)\big).
$$

*($\Leftarrow$)* Assume $\inf_{d}V^{\star}(f(x,u,d))\ge\beta(V^{\star}(x))$, i.e.
$$
V^{\star}\!\big(f(x,u,d)\big)\;\ge\;\beta\big(V^{\star}(x)\big)\qquad\forall d\in\mathcal D .
$$
Because $x\in\Omega^{\star}$ we have $V^{\star}(x)\ge0$, so by (P1) and $\beta(r)\le r$,
$$
g(x)\;\ge\;V^{\star}(x)\;\ge\;\beta\big(V^{\star}(x)\big).
$$
The two displayed inequalities bound both arguments of the minimum, hence
$$
\min\big\{g(x),\,V^{\star}(f(x,u,d))\big\}\;\ge\;\beta\big(V^{\star}(x)\big)\qquad\forall d\in\mathcal D ,
$$
which by (3.1) is $Q^{\star}(x,u,d)\ge\beta(V^{\star}(x))$ for all $d$. Taking the minimum over $d$,
$$
\min_{d\in\mathcal D}Q^{\star}(x,u,d)\;\ge\;\beta\big(V^{\star}(x)\big). \qquad\blacksquare
$$

### 3.5 Backbone Theorem B — $V^{\star}$ is the maximal robust DCBF

> **Theorem B (Maximal Robust Q-CBF; Oh et al. 2026, Theorem 1).** $V^{\star}$ is a valid robust DCBF (Def. 3) whose $0$-superlevel set is $\Omega^{\star}$. Equivalently, for every decay $\beta$ the **robust Q-CBF constraint**
> $$
> \min_{d\in\mathcal D}Q^{\star}(x,u,d)\;\ge\;\beta\big(V^{\star}(x)\big)
> \tag{3.5}
> $$
> is, on $\Omega^{\star}$, equivalent to the robust DCBF condition (2.9) with $B=V^{\star}$.

**Proof.** First $\Omega^{\star}=\{V^{\star}\ge0\}\subseteq\mathcal K$ by Fact 1. Fix any decay $\beta$ and any $x\in\Omega^{\star}$.
By **Lemma B1** there exists $u\in\mathcal U$ with
$$
\min_{d\in\mathcal D}Q^{\star}(x,u,d)\;\ge\;\beta\big(V^{\star}(x)\big).
$$
By **Lemma B2** this is equivalent to
$$
\inf_{d\in\mathcal D}V^{\star}\!\big(f(x,u,d)\big)\;\ge\;\beta\big(V^{\star}(x)\big),
$$
which is exactly the robust DCBF condition (2.9) for $B=V^{\star}$ at $x$. As $x\in\Omega^{\star}$ and $\beta$ were arbitrary, $V^{\star}$ is a valid robust DCBF on $\Omega^{\star}$ for every $\beta$, and (3.5) is the equivalent state–action form. $\blacksquare$

The undiscounted Isaacs value is therefore a valid robust DCBF for **every** $\beta\le\mathrm{id}$ — the discrete mirror of Remark C1 (offline $\lambda=0$, any online $\gamma$). Tunability lives in the *choice of $\beta$ in the constraint*, not in a discounted value function.

### 3.6 The robust Q-CBF safety filter and recursive feasibility

> **Definition 6 (robust Q-CBF filter; Oh et al. 2026, Def. 3).** For $x\in\Omega^{\star}$,
> $$
> u^{\star}(x)=\arg\min_{u\in\mathcal U}\big\|u-u_{\mathrm{task}}(x)\big\|^{2}
> \quad\text{s.t.}\quad
> \min_{d\in\mathcal D}Q^{\star}(x,u,d)\ge\beta\big(V^{\star}(x)\big).
> \tag{3.6}
> $$

> **Proposition R (recursive feasibility; Oh et al. 2026, Prop. 1).** (3.6) is recursively feasible for every decay $\beta$ and every $x_0\in\Omega^{\star}$.

**Proof.** Fix $x_t\in\Omega^{\star}$.
*Feasibility at $t$.* By Theorem B (via Lemma B1) there exists $u_t$ satisfying the constraint, so (3.6) is feasible.
*Invariance under any feasible choice.* Let $u_t$ be **any** feasible solution:
$$
Q^{\star}(x_t,u_t,d_t)\;\ge\;\beta\big(V^{\star}(x_t)\big)\qquad\forall d_t\in\mathcal D .
$$
From the definition (3.1), $Q^{\star}\le V^{\star}\!\circ f$, hence
$$
V^{\star}\!\big(f(x_t,u_t,d_t)\big)\;\ge\;Q^{\star}(x_t,u_t,d_t)\;\ge\;\beta\big(V^{\star}(x_t)\big)\qquad\forall d_t\in\mathcal D .
$$
Since $x_t\in\Omega^{\star}$, $V^{\star}(x_t)\ge0$, and $\beta$ class $\mathcal K$ gives $\beta(V^{\star}(x_t))\ge0$, so
$$
V^{\star}\!\big(f(x_t,u_t,d_t)\big)\;\ge\;0\qquad\forall d_t\in\mathcal D
\;\;\Longrightarrow\;\;
x_{t+1}\in\Omega^{\star}.
$$
The same argument applies at $t+1$; recursion closes. $\blacksquare$

### 3.7 Three distinct decay/discount quantities (a recurring confusion, settled here)

The chain involves **three** scalars that are easy to conflate; they are mathematically different and only one sets the safety margin.

1. **Offline CBVF rate $\lambda$** (continuous, §2.3, eq. 2.4): baked into the value function. May be set to $0$ for infinite-time safety (Remark C1).
2. **Deployment decay $\beta$ / $\gamma=1-\alpha$** (the runtime constraint, (3.5)–(3.6)): a *free design choice*; the undiscounted value works for every $\beta\le\mathrm{id}$ (Theorem B). This is the only knob that sets the one-step invariance margin.
3. **Training discount $\gamma_{\mathrm{ENV}}$** (numerical, §4.1): a contraction device for RL with **no** role in the invariance margin (HCSF: it "does not affect the safety guarantees").

We return to (3) in §5.10, where the certificate is shown to be insulated from $\gamma_{\mathrm{ENV}}$ because it certifies the deployed network directly.

---

## 4. The learned filter and the exact point where its guarantee stops

Theorem B is about the *exact* $V^{\star},Q^{\star}$. In practice these are neural networks trained by adversarial RL. This section states what training produces, then isolates — at the equation level — the precise gap our certificate closes (Oh et al. 2026, Prop. 2).

### 4.1 Neural synthesis (adversarial RL) and the discounted training value

Adversarial RL jointly trains a critic $Q_\theta(x,u,d)$, a control actor $\pi^{u}(x)$, and a disturbance actor $\pi^{d}_{\psi}(x,u)$ as a zero-sum game, minimizing the Bellman residual of the **time-discounted** Isaacs equation
$$
\mathcal L(\theta)=\mathbb E_{(x,u,d,g',x')\sim\mathcal B}\Big[\big(Q_\theta(x,u,d)-y\big)^2\Big],
\quad
y=(1-\gamma_{\mathrm{ENV}})\,g' + \gamma_{\mathrm{ENV}}\min\big\{g',\,Q_{\theta'}(x',u',d')\big\},
\tag{4.1}
$$
with $\gamma_{\mathrm{ENV}}\in(0,1)$ a contraction/stability device (HCSF: $\gamma_{\mathrm{ENV}}=0.992$). Two-timescale GDA (disturbance faster) converges to a *local* minimax equilibrium $(\pi^{u}_{\star},\pi^{d}_{\psi^{\star}})$. We **freeze** the result: the certifier faces static networks $V_\theta,Q_\theta,\pi^{\flat}$ (and $\pi^{d}_{\tilde\psi}$ if used) — adversarial training enters synthesis only, never the verifier (§5).

Define the learned certified set and the learned fallback exactly as in §3 but on the networks:
$$
\Omega_c:=\{x:V_\theta(x)\ge c\},\qquad
\pi^{\flat}(x):=\arg\max_{u}\min_{d}Q_\theta(x,u,d).
\tag{4.2}
$$

### 4.2 The plug-in adversary and Proposition L (local robustness only)

At runtime the nested $\min_d$ in (3.5) is expensive, so the learned disturbance is *plugged in*: $\tilde d=\pi^{d}_{\tilde\psi}(x,u)$, and the constraint is enforced as $Q_\theta(x,u,\tilde d)\ge\beta(V_\theta(x))$. The guarantee this buys is **local in disturbance-policy space**, not over $\mathcal D$.

> **Proposition L (local robustness; Oh et al. 2026, Prop. 2).** Suppose $\pi^{d}_{\tilde\psi}$ is locally optimal, i.e.
> $$
> Q_\theta\big(x,u,\pi^{d}_{\tilde\psi}(x,u)\big)\;\le\;Q_\theta\big(x,u,\pi^{d}_{\psi}(x,u)\big)
> \qquad\forall\,\psi\in B_\rho(\tilde\psi),\;\rho>0 .
> \tag{4.3}
> $$
> If the surrogate constraint $Q_\theta(x,u,\pi^{d}_{\tilde\psi}(x,u))\ge\beta(V_\theta(x))$ holds, then for **all** $\psi\in B_\rho(\tilde\psi)$,
> $$
> Q_\theta\big(x,u,\pi^{d}_{\psi}(x,u)\big)\;\ge\;\beta\big(V_\theta(x)\big).
> $$

**Proof.** For any $\psi\in B_\rho(\tilde\psi)$, chain local optimality (4.3) with the surrogate:
$$
Q_\theta\big(x,u,\pi^{d}_{\psi}(x,u)\big)
\;\ge\;
Q_\theta\big(x,u,\pi^{d}_{\tilde\psi}(x,u)\big)
\;\ge\;
\beta\big(V_\theta(x)\big). \qquad\blacksquare
$$

> **The gap, stated precisely.** Proposition L quantifies over disturbance **policies** in a ball $B_\rho(\tilde\psi)$ — *not* over disturbance **realizations** $d\in\mathcal D$. The true worst case is
> $$
> \min_{d\in\mathcal D}Q_\theta(x,u,d)\;\le\;Q_\theta\big(x,u,\pi^{d}_{\psi}(x,u)\big)\quad\forall\psi ,
> $$
> and there is **no** guarantee that any $\pi^{d}_{\psi}\in B_\rho(\tilde\psi)$ attains it. A feedforward $\pi^{d}_{\psi}(x,u)$ cannot track a worst-case vertex of $\mathcal D$ that switches across a manifold in $(x,u)$. Hence "robust against nearby learned adversaries" $\ne$ "robust against all of $\mathcal D$." Replacing the policy-ball quantifier by the **set** quantifier $\forall d\in\mathcal D$, soundly, is exactly the contribution of §5.6–§5.7; Oh et al. (2026, Remark 3) name post-hoc verification as the way to obtain it.

## 4.3 Why certifying against the learned adversary is unsound
 
Make the previous paragraph a statement about feasible-action sets.
 
> **Definition 7.** With the learned adversary $d_\psi(x,u)\in\mathcal D$ and decay $\beta$,
> $$
> \mathcal F_\psi(x):=\big\{u:\,Q_\theta(x,u,d_\psi(x,u))\ge\beta(V_\theta(x))\big\},
> \qquad
> \mathcal F_{\mathrm{rob}}(x):=\big\{u:\,\textstyle\min_{d\in\mathcal D}Q_\theta(x,u,d)\ge\beta(V_\theta(x))\big\}.
> $$
 
> **Proposition U (the learned-adversary set is too large).**
> $$
> \mathcal F_{\mathrm{rob}}(x)\;\subseteq\;\mathcal F_\psi(x)\qquad\forall x .
> \tag{4.4}
> $$
> Consequently, certifying feasibility against $d_\psi$ over-approximates the truly safe action set; it is **not** a sound robustness certificate.
 
**Proof.** Fix $x$ and take any $u\in\mathcal F_{\mathrm{rob}}(x)$, so
$$
\min_{d\in\mathcal D}Q_\theta(x,u,d)\;\ge\;\beta\big(V_\theta(x)\big).
$$
Because $d_\psi(x,u)\in\mathcal D$ is one feasible point of the minimum,
$$
Q_\theta\big(x,u,d_\psi(x,u)\big)\;\ge\;\min_{d\in\mathcal D}Q_\theta(x,u,d)\;\ge\;\beta\big(V_\theta(x)\big),
$$
hence $u\in\mathcal F_\psi(x)$. This proves (4.4). $\blacksquare$
 
Proposition U gives only the *inclusion*. The remainder of this subsection establishes when it is **strict**, and shows that strictness is **generic** — not incidental — under the natural structure in which the worst-case disturbance switches between distinct points of $\mathcal D$. This is the formal content of the motivating experiment (E0) and the precise reason a continuous plug-in adversary $d_\psi$ is insufficient. Our certificate then chooses the general sound route: discharge the inner minimum over the *full* $\mathcal D$ (§5.6–§5.7).
 
### Notation and a characterization of strictness
 
Write
$$
z=(x,u),\qquad q(z,d):=Q_\theta(x,u,d),\qquad \eta(x):=\beta\big(V_\theta(x)\big),
$$
$$
q_{\min}(z):=\min_{d\in\mathcal D}q(z,d),\qquad q_\psi(z):=q\big(z,d_\psi(z)\big).
$$
Then $\mathcal F_{\mathrm{rob}}(x)=\{u:q_{\min}(x,u)\ge\eta(x)\}$ and $\mathcal F_\psi(x)=\{u:q_\psi(x,u)\ge\eta(x)\}$, and Proposition U is the pointwise inequality $q_{\min}\le q_\psi$.
 
> **Characterization (S1).** The inclusion is **strict at $x$**,
> $$
> \mathcal F_{\mathrm{rob}}(x)\subsetneq\mathcal F_\psi(x),
> $$
> **iff** there exists $u\in\mathcal U$ with
> $$
> q_{\min}(x,u)\;<\;\eta(x)\;\le\;q_\psi(x,u).
> \tag{S1}
> $$
 
**Proof.** If (S1) holds for some $u$, then $u\notin\mathcal F_{\mathrm{rob}}(x)$ while $u\in\mathcal F_\psi(x)$, so the inclusion (4.4) is strict. Conversely, strict inclusion gives some $u\in\mathcal F_\psi(x)\setminus\mathcal F_{\mathrm{rob}}(x)$, which is exactly (S1). $\blacksquare$
 
So strictness reduces to producing a single action at which the learned adversary leaves slack *across the threshold* $\eta(x)$. The structural lemma below shows such slack occupies a **positive-measure** region whenever the worst-case identity switches and $d_\psi$ is continuous.
 
### Standing regularity (this subsection)
 
$q$ is jointly continuous on $\mathcal X\times\mathcal U\times\mathcal D$; $\mathcal D$ is **compact**; $\beta$ is continuous; and the learned adversary $d_\psi:\mathcal X\times\mathcal U\to\mathcal D$ is **continuous**. ($V_\theta,Q_\theta$ are continuous as networks with continuous activations; a feedforward $d_\psi$ is continuous *by construction* — this continuity is precisely the obstruction exploited below.) Two consequences are used:
 
- $q_\psi(z)=q(z,d_\psi(z))$ is continuous (composition of continuous maps);
- $q_{\min}(z)=\min_{d\in\mathcal D}q(z,d)$ is continuous — the minimum of a jointly continuous function over a **fixed compact** set is continuous (Berge's maximum theorem; equivalently, uniform continuity of $q$ on compacta).
The second point is needed for the open-neighborhood / positive-measure conclusion and is *not* implied by continuity of $q$ alone.
 
### The switching impossibility theorem
 
> **Theorem S (continuous plug-in adversaries miss switching worst cases).**
> Let $\mathcal N\subseteq\mathcal X\times\mathcal U$ be open and connected. Suppose there exist two **distinct** points $v^-\neq v^+\in\mathcal D$ and **disjoint nonempty open** sets $\mathcal N^-,\mathcal N^+\subseteq\mathcal N$ such that:
>
> **(i) Switching with a strict minimizer.** The worst-case disturbance is uniquely $v^-$ on $\mathcal N^-$ and $v^+$ on $\mathcal N^+$:
> $$
> \arg\min_{d\in\mathcal D}q(z,d)=v^-\ (z\in\mathcal N^-),\qquad \arg\min_{d\in\mathcal D}q(z,d)=v^+\ (z\in\mathcal N^+),
> $$
> with the minimizer **strict**, i.e.
> $$
> q(z,d)>q(z,v^-)\ \ \forall d\neq v^-\ (z\in\mathcal N^-),\qquad q(z,d)>q(z,v^+)\ \ \forall d\neq v^+\ (z\in\mathcal N^+).
> $$
>
> **(ii) Abutting regions.** The two regions meet: there exists
> $$
> z_0\in\operatorname{cl}(\mathcal N^-)\cap\operatorname{cl}(\mathcal N^+)\cap\mathcal N .
> $$
> *(Equivalently, the switching surface $\Sigma:=\mathcal N\setminus(\mathcal N^-\cup\mathcal N^+)$ has empty interior and is approached from both sides; in $\mathbb R^n$ this is automatic when $\mathcal N^\pm$ are the two sides of a separating hypersurface.)*
>
> Then, for **every** continuous $d_\psi$, there is a **nonempty open** set $\mathcal M\subseteq\mathcal N$ on which the learned adversary strictly under-realizes the worst case:
> $$
> q_\psi(z)\;>\;q_{\min}(z)\qquad\forall z\in\mathcal M .
> \tag{S2}
> $$
>
> This is the necessity-side statement: it rules out the continuous plug-in-adversary shortcut under switching. It does **not** claim that sound full-$\mathcal D$ bounding is the unique possible implementation route; exact inner minimization, a discontinuous/set-valued worst-case selector, or special monotone endpoint structure can close the same gap in special cases.
 
**Proof.**
 
*Step 1 — the gap region $\mathcal M$ is open.* Define the mismatch set
$$
\mathcal M:=\{z\in\mathcal N^-:d_\psi(z)\neq v^-\}\;\cup\;\{z\in\mathcal N^+:d_\psi(z)\neq v^+\}.
$$
The singletons $\{v^-\},\{v^+\}$ are closed in $\mathcal D$, so $\mathcal D\setminus\{v^\pm\}$ are open; by continuity of $d_\psi$ the preimages $d_\psi^{-1}(\mathcal D\setminus\{v^\pm\})$ are open. Intersecting with the open sets $\mathcal N^\mp$ shows each term is open, hence $\mathcal M$ is open.
 
*Step 2 — $\mathcal M$ is nonempty.* Suppose $\mathcal M=\varnothing$. Then $d_\psi\equiv v^-$ on $\mathcal N^-$ and $d_\psi\equiv v^+$ on $\mathcal N^+$. By hypothesis (ii) pick $z_0\in\operatorname{cl}(\mathcal N^-)\cap\operatorname{cl}(\mathcal N^+)\cap\mathcal N$ and sequences
$$
z_k^-\in\mathcal N^-\to z_0,\qquad z_k^+\in\mathcal N^+\to z_0 .
$$
Continuity of $d_\psi$ at $z_0$ forces
$$
d_\psi(z_0)=\lim_{k\to\infty}d_\psi(z_k^-)=v^-\qquad\text{and}\qquad d_\psi(z_0)=\lim_{k\to\infty}d_\psi(z_k^+)=v^+,
$$
contradicting $v^-\neq v^+$. Hence $\mathcal M\neq\varnothing$.
 
> **Where (ii) bites.** Without (ii) the contradiction in Step 2 collapses: if $\Sigma$ has nonempty interior, a continuous $d_\psi$ may interpolate from $v^-$ to $v^+$ *entirely inside $\Sigma$* — where the worst-case identity is unconstrained — matching the worst point on all of $\mathcal N^-\cup\mathcal N^+$, giving $\mathcal M=\varnothing$ with no contradiction. Hypothesis (ii) is exactly the geometry that rules this out, and it is the *only* hypothesis doing work beyond bookkeeping.
 
*Step 3 — strict under-realization on $\mathcal M$.* Fix $z\in\mathcal M$, say $z\in\mathcal N^-$ with $d_\psi(z)\neq v^-$ (the $\mathcal N^+$ case is symmetric). Strictness in (i) gives
$$
q\big(z,d_\psi(z)\big)\;>\;q(z,v^-)\;=\;\min_{d\in\mathcal D}q(z,d),
$$
i.e. $q_\psi(z)>q_{\min}(z)$. This is (S2). $\blacksquare$

> **Corollary S3 (threshold landing gives false feasibility).**
> Under Theorem S, suppose the decay threshold lands inside the value gap with margin: there exist $\bar z=(\bar x,\bar u)\in\mathcal M$ and $\mu>0$ such that
> $$
> q_{\min}(\bar z)+\mu\;\le\;\eta(\bar x)\;\le\;q_\psi(\bar z)-\mu .
> \tag{S3}
> $$
> Then
> $$
> \mathcal F_{\mathrm{rob}}(\bar x)\subsetneq\mathcal F_\psi(\bar x).
> $$
> Moreover, the strict inclusion persists on an open neighborhood of $\bar z$, and the set of falsely-feasible actions
> $$
> \big\{(x,u):u\in\mathcal F_\psi(x)\setminus\mathcal F_{\mathrm{rob}}(x)\big\}
> $$
> has **positive Lebesgue measure**.

**Proof.** The margin condition (S3) implies
$$
q_{\min}(\bar z)\;<\;\eta(\bar x)\;<\;q_\psi(\bar z),
$$
so $\bar u\notin\mathcal F_{\mathrm{rob}}(\bar x)$ and $\bar u\in\mathcal F_\psi(\bar x)$ by (S1). Hence
$$
\mathcal F_{\mathrm{rob}}(\bar x)\subsetneq\mathcal F_\psi(\bar x).
$$
Because $q_{\min}$, $q_\psi$, and $\eta$ are continuous (standing regularity), there is an open neighborhood $\mathcal O\subseteq\mathcal M$ of $\bar z$ such that, for every $z=(x,u)\in\mathcal O$,
$$
q_{\min}(z)+\frac{\mu}{2}\;<\;\eta(x)\;<\;q_\psi(z)-\frac{\mu}{2}.
$$
Thus every $(x,u)\in\mathcal O$ satisfies $u\in\mathcal F_\psi(x)\setminus\mathcal F_{\mathrm{rob}}(x)$. Since $\mathcal O$ is a nonempty open subset of $\mathcal X\times\mathcal U\subseteq\mathbb R^{n+m}$, it has positive Lebesgue measure. $\blacksquare$
 
### Reading Theorem S and Corollary S3
 
> **Vertices vs. distinct minimizers (scope, and consistency with non-convexity).** The proof uses only that $v^-,v^+$ are *distinct points of $\mathcal D$* with a strict gap — **not** that they are vertices. Box vertices are the canonical instance: when $Q_\theta$ is coordinatewise monotone in $d$ on each region, the inner minimum is attained at a corner of $\mathcal D$ whose identity is fixed by the local sign pattern of monotonicity (cf. §5.7, Remark E2); a sign flip across a manifold in $(x,u)$ is precisely a switching surface with $v^-\neq v^+$ corners. The hypothesis therefore *asserts a particular local structure*, not global convexity — it is **consistent with** the general non-convexity of $Q_\theta$ in $d$ (the very reason the inner min is hard, §2.5/§5.7), not in tension with it.
 
> **What it means for the certificate.** Theorem S upgrades Proposition U from "the learned-adversary set *can* be too large" to "continuous plug-in adversaries generically miss switching worst cases." When the threshold condition in Corollary S3 also holds, the learned-adversary feasible set is **strictly too large** on a positive-measure region. Crucially, the value gap (S2) is **not closable by training a better continuous adversary**: the obstruction is the *discontinuity* of the true worst-case selection map
> $$
> z\;\longmapsto\;\arg\min_{d\in\mathcal D}q(z,d),
> $$
> which jumps from $v^-$ to $v^+$ across $\Sigma$ and which **no continuous network can represent**. Formally,
> $$
> \boxed{\;\text{a continuous learned adversary cannot represent a switching (discontinuous) worst-case selection;}\;}
> $$
> hence Proposition L's *local* (policy-ball) robustness cannot be promoted to robustness over all of $\mathcal D$ by a continuous plug-in $d_\psi$. This is the formal reason we do not certify against the learned adversary. Our chosen sufficient route is to discharge the inner minimum over the **full** $\mathcal D$ by sound bounding (§5.6–§5.7), while special exact routes remain allowed when their assumptions hold. Theorem S also hardens the motivating experiment E0: the falsely-feasible actions in Corollary S3 are exactly the states where a $d_\psi$-certified filter admits a control that the true worst case in $\mathcal D$ defeats.
 
> **On the threshold condition (S3).** Strictness of the *value* gap (S2) is unconditional given (i)–(ii); strictness of the *feasibility* gap additionally requires the decay threshold $\eta(x)=\beta(V_\theta(x))$ to fall inside the value gap with room. The $\mu$-margin in (S3) is deliberately stronger than bare strictness: it gives a robust open neighborhood and positive-measure false feasibility. This is honest and expected: a vacuous threshold that no control meets, or a slack threshold that every control clears, washes out the difference. The interesting — and generic — regime is $\beta$ active near $\partial\Omega_c$, exactly where the filter operates. Theorem S itself holds unconditionally; the threshold-landing hypothesis (S3) is what gives it *bite*, and confirming that hypothesis on a particular trained object is an **implementation-stage** diagnostic (a controlled object suffices if a checkpoint does not exhibit it), not a planning-phase precondition.
 
---

## 5. A sound robustness certificate for the deployed filter

The guiding decision (the spine of this work): **certify the artifact that runs, not an ideal.** We do *not* assume $Q_\theta\approx Q^{\star}$, nor that $Q_\theta$ solves the Isaacs equation. We verify a few conditions about the deployed networks $(V_\theta,Q_\theta,\pi^{\flat})$ and the **true** $f$, from which safety follows by Theorem 1 applied to $V_\theta$.

### 5.1 The deployed filter and its feasibility predicate

Let $\Phi(x,u)$ be the **feasibility predicate the filter actually enforces**. The model-free-runtime choice mirrors (3.6) on the networks:
$$
\Phi(x,u)\;:\equiv\;\Big[\,\min_{d\in\mathcal D}Q_\theta(x,u,d)\;\ge\;\beta\big(V_\theta(x)\big)\Big],
\qquad
u^{\star}(x)=\arg\min_{u:\,\Phi(x,u)}\big\|u-u_{\mathrm{task}}(x)\big\|^{2}.
\tag{5.1}
$$
Every statement below is about *this* $\Phi$; if a deployment enforces a different inequality, the certificate is restated for that predicate — that substitution **is** the deployed-object principle.

> **Story B placement of this lane.** The predicate (5.1) discharges the inner worst case over the **full** $\mathcal D$, i.e. it is the case $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$. Sections 5.1–5.10 are therefore the **true-robustness (full-$\mathcal D$) deterministic lane**, affordable only at low dimension — the regime where the false-feasibility gap is *zero by construction* (the filter is certified against all of $\mathcal D$) and where the gap of the *optimistic* deployed object can be **measured** against this ground truth. The high-dimensional deployed object of §5.11 replaces $\min_{d\in\mathcal D}$ in (5.1) by $\inf_{\xi\in\widehat{\mathcal D}_{\mathrm{dep}}}$ (optimistic), and is certified only for survival, not robustness. Read §5.1–§5.10 as the microscope; read §5.11 as the scalable lane.

### 5.2 The certified set and the three verifiable conditions

Fix a threshold $c\ge0$ (a **search knob**, §5.5) and a verification margin $\varepsilon\ge0$ (§5.7). With $\Omega_c=\{x:V_\theta(x)\ge c\}$:

> **Domain convention for $\beta$.** Because $\beta:\mathbb R_{\ge0}\to\mathbb R_{\ge0}$, the certificate is stated for $c\ge0$ so that $V_\theta(x)\ge c$ implies $\beta(V_\theta(x))$ is well-defined on $\Omega_c$. If one wants to certify a negative level set, the statement must instead use an extended class-$\mathcal K$ decay or a shifted predicate in $V_\theta-c$; this core uses the non-negative superlevel convention.

> **(C1) No-failure.** $\displaystyle\;\forall x\in\Omega_c:\; g(x)\ge 0.$ &nbsp; *(i.e. $\Omega_c\subseteq\mathcal K$ — ties the learned superlevel set to the true safety function.)*
>
> **(C2) Robust transition.** $\displaystyle\;\forall x\in\Omega_c,\;\forall u\text{ with }\Phi(x,u),\;\forall d\in\mathcal D:\; V_\theta\!\big(f(x,u,d)\big)\ge c+\varepsilon.$
>
> **(C3) Witness feasibility.** $\displaystyle\;\forall x\in\Omega_c,\;\forall d\in\mathcal D:\; Q_\theta\!\big(x,\pi^{\flat}(x),d\big)\ge\beta\big(V_\theta(x)\big)+\varepsilon.$

(C2) says every action the filter *could* take keeps the *true* successor in $\Omega_c$ against every $d$. (C3) says the feasible set of (5.1) is never empty on $\Omega_c$, witnessed by $\pi^{\flat}$.

Quantifying (C2) over **all** feasible $u$ — rather than the single $u^{\star}(x)$ the filter actually selects — is a deliberate **sound over-approximation**: it makes the certificate robust to the filter's tie-breaking among feasible actions and keeps the condition verifier-friendly (no inner $\arg\min$ over $u_{\mathrm{task}}$ to encode), at the price of some conservatism in $\Omega_c$. This is one of the contributors to potential vacuity (§6).

### 5.3 Soundness firewall (read before the theorem)

(C2) composes the **known** $f$ with the network, so it checks the *true* transition. Therefore a poorly trained $Q_\theta$ that would admit an unsafe $u$ makes (C2) **fail** on those states; the response is to shrink $\Omega_c$ (raise $c$), never to certify them.

> **A bad network yields a small or empty $\Omega_c$ — never an unsound certificate.** This is exactly why no $\|Q_\theta-Q^{\star}\|$ bound appears anywhere, and why $\varepsilon$ is verification slack (§5.7), not an approximation constant.

### 5.4 Theorem A — deployed-filter robust invariance and safety

> **Theorem A.** If (C1)–(C3) hold for $(V_\theta,Q_\theta,\pi^{\flat})$ under the true dynamics $f$, then
> 1. **(Recursive feasibility)** $\forall x\in\Omega_c$, the feasible set $\{u:\Phi(x,u)\}\ne\varnothing$ (it contains $\pi^{\flat}(x)$);
> 2. **(Robust forward invariance)** the closed loop (5.1) under any feasible selection satisfies $x_0\in\Omega_c\Rightarrow x_t\in\Omega_c$ for all $t$ and all $\mathbf d\in\mathcal D^{\infty}$;
> 3. **(Safety)** consequently $g(x_t)\ge0$ for all $t$.

**Proof.**

*Part 1 (recursive feasibility).* Fix $x\in\Omega_c$. By (C3), for every $d\in\mathcal D$,
$$
Q_\theta\!\big(x,\pi^{\flat}(x),d\big)\;\ge\;\beta\big(V_\theta(x)\big)+\varepsilon\;\ge\;\beta\big(V_\theta(x)\big).
$$
Taking the minimum over $d$,
$$
\min_{d\in\mathcal D}Q_\theta\!\big(x,\pi^{\flat}(x),d\big)\;\ge\;\beta\big(V_\theta(x)\big),
$$
which is precisely $\Phi(x,\pi^{\flat}(x))$. Hence the feasible set is non-empty.

*Part 2 (robust forward invariance).* Induction on $t$.

Base: $x_0\in\Omega_c$ by hypothesis, so
$$
V_\theta(x_0)\ge c .
$$
Step: assume $x_t\in\Omega_c$. By Part 1 the filter selects some $u_t$ with $\Phi(x_t,u_t)$ (the minimizer of (5.1) over a non-empty feasible set). Apply (C2) at $(x_t,u_t)$: for **every** $d\in\mathcal D$,
$$
V_\theta\!\big(f(x_t,u_t,d)\big)\;\ge\;c+\varepsilon\;\ge\;c .
$$
In particular for the realized $d_t\in\mathcal D$,
$$
V_\theta(x_{t+1})\;=\;V_\theta\!\big(f(x_t,u_t,d_t)\big)\;\ge\;c ,
$$
so $x_{t+1}\in\Omega_c$. Since $d_t\in\mathcal D$ was arbitrary, $\Omega_c$ is robustly forward invariant.

*Part 3 (safety).* By (C1), $\Omega_c\subseteq\mathcal K$. By Part 2, $x_t\in\Omega_c$ for all $t$, hence
$$
g(x_t)\;\ge\;0\qquad\forall t . \qquad\blacksquare
$$

> **Remark A1 (what is and is not used).** Parts 2–3 use only (C1)–(C2); Part 1 is exactly what the witness (C3) buys. No step references $V^{\star},Q^{\star}$, the Isaacs equation, or any approximation bound — only the *verified inequalities* and the *true* $f$. The guarantee holds over **every** disturbance sequence. Note also that $\varepsilon$ is **not consumed by the proof**: every use is the slackening $\ge c+\varepsilon\ge c$ or $\ge\beta+\varepsilon\ge\beta$, so the theorem already holds at $\varepsilon=0$. We carry $\varepsilon>0$ purely as **verifier margin** (so sound bound-propagation can certify a strict inequality, §5.7) and as **refinement buffer** (T4) — never as an approximation constant (§5.3).

### 5.5 Decoupling the runtime decay $\beta$ from certified invariance; the $c$-search

A subtlety the Q-CBF constraint does not resolve on its own: the filter (5.1) only guarantees, roughly, $V_\theta(x_{t+1})\gtrsim\beta(V_\theta(x_t))$, and near $\partial\Omega_c$ (where $V_\theta\approx c$) one has
$$
\beta\big(V_\theta(x_t)\big)\;\approx\;\beta(c)\;\le\;c ,
$$
which is **strictly below** $c$ for $c>0$. So the runtime constraint does **not**, by itself, imply invariance of $\Omega_c$.

The deployed-object framing dissolves this without any global "$\beta\Rightarrow$ invariance" argument: **(C2) checks the landing condition $V_\theta(f)\ge c+\varepsilon$ directly** for the specific learned $V_\theta$ and predicate $\Phi$. Hence:

- $\beta$ governs only *runtime aggressiveness* (how hard the filter pushes back / how it interpolates to $u_{\mathrm{task}}$);
- *certified invariance* is whatever (C2) verifies directly;
- the **$c$-search** reconciles them: increase $c$ to shrink $\Omega_c$ and add transition margin until (C2) passes; report the **largest verifiable** $c$. The natural maximal case $c=0$ with $\beta(0)=0$ makes the runtime constraint and (C2) coincide on $\partial\Omega_0$ (mirroring Remark C1: $\lambda=0$, any online $\gamma$).

> **Remark (the coupling is two-way).** "$\beta$ governs only aggressiveness" is exact for the invariance **conclusion** — Theorem A, Part 2 uses only (C2)'s landing condition $V_\theta(f)\ge c$ and never references $\beta$. But $\beta$ does enter the invariance **hypothesis**: (C2) quantifies over $\{u:\Phi(x,u)\}$, and a more aggressive $\beta$ (higher floor) *shrinks* that feasible set, making (C2) *easier* to discharge — while simultaneously *stressing* (C3), whose witness must now clear a higher bar. So $\beta$ is a second verification knob alongside the $c$-search: it trades (C2)-ease against (C3)-feasibility, even though it leaves the conclusion untouched.

### 5.6 Witness compression: $\forall\exists\forall\to\forall\forall$ (T6)

Recursive feasibility is natively three-quantifier:
$$
\forall x\in\Omega_c\;\;\exists u\in\mathcal U\;\;\forall d\in\mathcal D:\quad Q_\theta(x,u,d)\;\ge\;\beta\big(V_\theta(x)\big).
\tag{5.2}
$$
The inner $\exists u$ nested in $\forall d$ is outside the reach of a sound bound-propagation verifier (which discharges $\forall$ by over-approximation but cannot witness $\exists$). **Pinning $u=\pi^{\flat}(x)$ collapses (5.2) to (C3)**, a pure $\forall\forall$ condition.

*Lineage and the exact soundness of pinning.* CMU (Li et al., L-CSS 2025, Def. 4) use a witness policy $\pi_\varphi$ to soundly bound the analogous $\exists u$ for the *non-robust* filter. In their convention (safe $\Leftrightarrow Q\le0$), the bound is

> **Lemma W (CMU Def. 4, soundness of pinning).** Since $\pi_\varphi(x)\in\mathcal U$ is one feasible control,
> $$
> Q\big(x,\pi_\varphi(x)\big)\le 0\;\Longrightarrow\;\min_{u\in\mathcal U}Q(x,u)\le Q\big(x,\pi_\varphi(x)\big)\le 0 ,
> $$
> i.e. checking the witness is a **sufficient** condition for the existential.

In our sign convention the same logic reads: $\pi^{\flat}(x)\in\mathcal U$ is one feasible control, so verifying $\min_dQ_\theta(x,\pi^{\flat}(x),d)\ge\beta(V_\theta(x))$ (which is (C3)) suffices for $\exists u$ in (5.2) — **regardless of how well $\pi^{\flat}$ approximates the true argmax** (a suboptimal witness only makes (C3) harder to pass, never unsound).

> **Our delta (T6).** Carry the witness to the **learned, approximate** $\pi^{\flat}$ with margin $\varepsilon$ and **verify the robust margin over all $d\in\mathcal D$** on $\Omega_c$, i.e. (C3). CMU verify the *non-robust* version ($\exists u$, no $d$); Oh et al. (Lemma B1) prove the *exact* fallback feasibility but do not verify it on a network. (C3) is the first verified, robust, recursive-feasibility witness for the *deployed approximate* filter.

> **Composition margin for the witness (training rationale, not a soundness hypothesis).** If $\pi^{\flat}=\pi^{\flat}_\varphi$ is a network, (C3) is the composed predicate
> $$
> Q_\theta\!\big(x,\pi^{\flat}_\varphi(x),d\big)\;\ge\;\beta\big(V_\theta(x)\big)+\varepsilon
> \qquad\forall (x,d)\in\Omega_c\times\mathcal D,
> $$
> and the **only soundness path** is still direct verification of this inequality. We do **not** add a proof assumption of the form $\|Q_\theta-Q^{\star}\|$, nor do we require a certified approximation error to an exact fallback selector.
>
> The following margin calculation explains why an anti-collapse / witness-feasibility loss is the right training pressure. Define
> $$
> R_\theta(x,u)
> :=
> \min_{d\in\mathcal D}Q_\theta(x,u,d)-\beta\big(V_\theta(x)\big).
> $$
> On a set $S\subseteq\Omega_c$, suppose there exists a margin-feasible robust selector $\bar\pi$ such that
> $$
> R_\theta\big(x,\bar\pi(x)\big)\ge\varepsilon+\mu\qquad\forall x\in S
> $$
> for some $\mu>0$. Suppose also that $Q_\theta(x,\cdot,d)$ is uniformly $L_u$-Lipschitz in $u$ for all $(x,d)\in S\times\mathcal D$, and that the trained witness satisfies
> $$
> \big\|\pi^{\flat}_\varphi(x)-\bar\pi(x)\big\|\le\delta,\qquad L_u\delta<\mu .
> $$
> Since the pointwise minimum of uniformly $L_u$-Lipschitz functions is again $L_u$-Lipschitz, $R_\theta(x,\cdot)$ is $L_u$-Lipschitz. Hence
> $$
> R_\theta\big(x,\pi^{\flat}_\varphi(x)\big)
> \ge
> R_\theta\big(x,\bar\pi(x)\big)-L_u\big\|\pi^{\flat}_\varphi(x)-\bar\pi(x)\big\|
> \ge
> \varepsilon+\mu-L_u\delta
> >
> \varepsilon .
> $$
> Thus (C3) would hold on $S$. This is **not** used by Theorem A; it is only the training-side explanation for why a witness-margin loss can make the directly verified composition $Q_\theta\!\circ\!\pi^{\flat}_\varphi$ non-vacuous. It relocates, rather than removes, the non-vacuity burden: the learned object must contain a margin-feasible selector on the set one hopes to certify.

> **Remark (well-posedness — a bonus of the deployed object).** The *exact* fallback $\pi^{\flat}(x)=\arg\max_u\min_d Q^{\star}$ may be set-valued or discontinuous in $x$ (the $\arg\max$ can jump across switching surfaces), so as a feedback law it is only guaranteed a *measurable selection* (existence via compact $\mathcal U$ + continuity). The *deployed* $\pi^{\flat}_\varphi$ is a network, hence continuous by construction, so the closed-loop trajectory is automatically well-defined. This is one more place where certifying the object that runs is cleaner than certifying the ideal.

### 5.7 Sound discharge of the inner game over the full $\mathcal D$ (T2)

The inner extremization over $\mathcal D$ inside the quantifiers of (C2)/(C3) is the technical core of the robustness axis. Two facts shape the method:

1. **$Q_\theta(\cdot,\cdot,d)$ is generally non-convex in $d$** (it approximates $\min\{g,V_\theta\!\circ f\}$ with $V_\theta$ a nonlinear net), so the inner $\min/\max$ over $\mathcal D$ is **not** an LP, **not** closed-form, **not** free. *(This corrects an earlier affine-in-$d$ assumption.)*
2. The sound, **architecture-agnostic** route treats $d$ as an input dimension and bounds the inner extremum over the box $\mathcal D$ by interval / linear-bound propagation with branch-and-bound ($\alpha,\!\beta$-CROWN / auto-LiRPA), composing the known $f$ for (C2).

> **Proposition T2 (incremental full-$\mathcal D$ discharge).** Verifying (C2)/(C3) over the full $\mathcal D$ does **not** require an HJI-style grid over the product state-disturbance space. In the verifier, $d$ is treated as an additional bounded input block and is handled by sound interval / linear relaxation plus optional branching in the $d$-box. Thus, relative to the same neural-verification problem over $(x,u)$, the robust certificate adds bounding/branching burden in the disturbance coordinates. This is **not** a claim of scalability in $\dim(x)$, nor a claim that the runtime is independent of coupling among $(x,u,d)$; it is only the defensible distinction from gridding the joint HJI state $(x,d)$. Looser inner bounds only **shrink** $\Omega_c$; soundness is never at risk.

The encoding has one genuinely subtle point: the predicate $\Phi$ in (C2) contains a $\min_d$ in the **antecedent**. Written out,
$$
\forall x,u,d:\quad
\underbrace{\Big[\min_{d'\in\mathcal D}Q_\theta(x,u,d')\ge\beta(V_\theta(x))\Big]}_{\text{antecedent }\Phi:\ \min_{d'}}
\;\Longrightarrow\;
\underbrace{\Big[V_\theta\!\big(f(x,u,d)\big)\ge c+\varepsilon\Big]}_{\text{consequent}} .
\tag{5.3}
$$

> **Lemma E (sound implication encoding).** Define the antecedent and consequent margins
> $$
> a(x,u):=\min_{d'\in\mathcal D}Q_\theta(x,u,d')-\beta(V_\theta(x)),
> \qquad
> b(x,u,d):=V_\theta(f(x,u,d))-(c+\varepsilon).
> \tag{5.4}
> $$
> Then (5.3) is exactly $a(x,u)\ge0\Rightarrow b(x,u,d)\ge0$. Over a cell $(\mathsf X,\mathsf U)\subseteq\Omega_c\times\mathcal U$:
> 1. compute a sound **upper** bound $\overline a(\mathsf X,\mathsf U)\ge\sup_{(x,u)\in\mathsf X\times\mathsf U}a(x,u)$. If $\overline a<0$, then the antecedent is false everywhere on the cell and the implication holds vacuously;
> 2. otherwise the cell may contain feasible controls, so compute a sound **lower** bound $\underline b(\mathsf X,\mathsf U,\mathcal D)\le\inf_{(x,u,d)\in\mathsf X\times\mathsf U\times\mathcal D}b(x,u,d)$. If $\underline b\ge0$, the implication holds on the whole cell;
> 3. if neither test succeeds, the cell is undecided: branch/refine or report it uncertified. Unknown cells are never counted as safe.
>
> **Soundness.** Case 1 is sound because $\overline a<0$ and $a\le\overline a$ imply $a(x,u)<0$ for every $(x,u)$ in the cell, so no feasible $u$ has been skipped. Case 2 is sound because $\underline b\ge0$ and $\underline b\le b$ imply $b(x,u,d)\ge0$ for every $(x,u,d)$ in the cell. A practical construction of $\overline a$ is:
> partition $\mathcal D$ into sub-boxes $\{\mathcal D_k\}$, take sound scalar upper bounds
> $$
> \overline Q_k\;\ge\;\sup_{(x,u,d')\in\mathsf X\times\mathsf U\times\mathcal D_k}Q_\theta(x,u,d'),
> $$
> set
> $$
> \overline q_{\min}:=\min_k\overline Q_k,
> $$
> and combine it with a sound lower bound
> $$
> \underline\beta\;\le\;\inf_{x\in\mathsf X}\beta(V_\theta(x))
> $$
> to obtain
> $$
> \overline a:=\overline q_{\min}-\underline\beta .
> $$
> Indeed, for every $(x,u)$,
> $$
> \min_{d'\in\mathcal D}Q_\theta(x,u,d')\le\overline q_{\min},
> \qquad
> \beta(V_\theta(x))\ge\underline\beta,
> $$
> hence $a(x,u)\le\overline a$. The sub-box bounds must enclose $Q_\theta$ over the whole $\mathcal D_k$; sampling a single $d'_k$ is not sound. Branch-and-bound termination is a **completeness** question, not a soundness requirement: under positive margins and bound gaps that vanish with cell diameter, refinement eventually decides the cell; otherwise the verifier may return unknown.

> **Remark E1 (design fork).** Choosing $\Phi$ to use $V_\theta\!\circ f$ instead of $Q_\theta$ makes (C2) *trivial* (it becomes $\Phi$ itself) and removes the antecedent-$\min$, **but** requires composing $f$ at runtime (model-based runtime), forfeiting the model-free-runtime property of §3.1. We keep the Q-CBF $\Phi$ and pay for it with Lemma E.

> **Remark E2 (monotone shortcut — optional, not relied upon).** If $Q_\theta$ is monotone in a component $d_i$, that component's inner extremum is at a box endpoint (a monotone scalar map on an interval attains its extrema at the ends) — exact and cheap for that component. Proposition T2 covers all cases without it; (C2)'s soundness rests on uniform bounding over the $(x,u,d)$ box, *not* on any vertex argument for how $d$ enters $f$.

### 5.8 Model error as a disturbance channel (T5)

> **Corollary M.** Suppose only a nominal model is known. Let $f_{\mathrm{cert}}$ be the dynamics used by the verifier and let $\mathcal D_{\mathrm{aug}}$ be a **combined** compact uncertainty set such that every true one-step transition is realizable by some admissible verifier disturbance:
> $$
> f_{\mathrm{true}}(x,u,\omega)\in\big\{f_{\mathrm{cert}}(x,u,\delta):\,\delta\in\mathcal D_{\mathrm{aug}}\big\}
> \qquad\forall(x,u,\omega),
> \tag{5.5}
> $$
> where $\omega$ denotes whatever real uncertainty is present. Then verifying (C1)–(C3) over $\mathcal D_{\mathrm{aug}}$ implies, via Theorem A, robust invariance and safety under the true (unmodeled) dynamics.

**Proof.** Theorem A instantiates the disturbance only at the realized transition. By (5.5), every realized true transition is represented by $f_{\mathrm{cert}}(x,u,\delta)$ for some $\delta\in\mathcal D_{\mathrm{aug}}$; that $\delta$ is covered by the verified (C2)/(C3). Hence the invariance induction of Theorem A goes through verbatim with $f=f_{\mathrm{cert}}$ and the combined uncertainty set $\mathcal D_{\mathrm{aug}}$. $\blacksquare$

> **Remark M1 (do not use a naive union).** If physical disturbance and model error can occur simultaneously, the combined set is typically a product or image set, e.g.
> $$
> \delta=(d_{\mathrm{env}},e_{\mathrm{model}})\in
> \mathcal D_{\mathrm{env}}\times\mathcal E_{\mathrm{model}},
> $$
> or, in an additive channel, the corresponding Minkowski-style image. A simple union $\mathcal D_{\mathrm{env}}\cup\mathcal E_{\mathrm{model}}$ is sound only in the special case where at most one channel is active at a time, which is not the default modeling assumption.

This is why the deterministic lane should be framed as **"model-free synthesis / model-free runtime, model-bounded certification"** — and it is the field norm (CMU's certification composes $f$ in its MIQCP). A fully model-free *deterministic* invariance certificate is impossible in the worst case. The model-free deployment alternative is therefore probabilistic: the conformal feasibility-survival lane in §5.11 calibrates real frozen-loop rollouts directly. **[OPEN]** source of the deterministic set $\mathcal D_{\mathrm{aug}}$ (Lipschitz / residual-learning / interval residual bound) — decides whether Corollary M stays deterministic-sound. A conformal residual source instead downgrades the deterministic lane to probabilistic coverage.

### 5.9 Monotone refinement of the certified set (T4)

For a generic controller-synthesis statement, the usual **existential** robust predecessor is
$$
\mathrm{Pre}_{\exists}(S):=\big\{x:\exists u\in\mathcal U,\;\forall d\in\mathcal D,\;f(x,u,d)\in S\big\}.
\tag{5.6}
$$
This computes a controlled-invariant subset for *some* policy. For the deployed filter (5.1), however, Theorem A certifies safety for **any feasible selection** of the QP, so the refinement operator must use the stronger all-feasible predecessor
$$
\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S):=
\big\{x:\{u:\Phi(x,u)\}\ne\varnothing,\;
\forall u\text{ with }\Phi(x,u),\;\forall d\in\mathcal D,\;f(x,u,d)\in S\big\},
\tag{5.7}
$$
and
$$
\mathcal T_{\Phi}(S):=S\cap\mathcal K\cap\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S).
\tag{5.8}
$$
If the implementation certifies a deterministic optimizer $u^\star(x)$ rather than arbitrary feasible tie-breaking, replace $\mathrm{Pre}_{\Phi}^{\mathrm{all}}$ by the corresponding one-control predecessor for that deployed selection.

> **Proposition T4 (monotone shrink-refinement).** $\mathcal T_{\Phi}$ is order-preserving ($S\subseteq S'\Rightarrow\mathcal T_{\Phi}(S)\subseteq\mathcal T_{\Phi}(S')$) and contractive ($\mathcal T_{\Phi}(S)\subseteq S$). Iterating from $S_0=\Omega_c$ yields a nested decreasing sequence
> $$
> S_0\supseteq S_1\supseteq S_2\supseteq\cdots,\qquad S_{k+1}=\mathcal T_{\Phi}(S_k),
> $$
> where finite $S_k$ are **$k$-step viability approximants**: $x\in S_{k+1}$ guarantees one robust deployed-filter step into $S_k$, not invariance of $S_{k+1}$ itself. Any fixed point $S=\mathcal T_{\Phi}(S)$ is robustly forward invariant under the deployed filter; on a finite verification partition, the descending iteration terminates at the greatest fixed point representable on that partition below $\Omega_c$.

**Proof.** Contractivity is immediate from the leading intersection with $S$. If $S\subseteq S'$, then any successor lying in $S$ also lies in $S'$, hence $\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S)\subseteq\mathrm{Pre}_{\Phi}^{\mathrm{all}}(S')$ and $\mathcal T_{\Phi}$ is order-preserving. The sequence is therefore nested. If $S=\mathcal T_{\Phi}(S)$, then every $x\in S$ is safe ($x\in\mathcal K$), the feasible set is non-empty, and every feasible $u$ sends every disturbance successor back into $S$; the induction proof of Theorem A applies with $S$ in place of $\Omega_c$. Thus fixed points, not arbitrary finite iterates, are the invariant sets. On the finite-cell lattice used by a verifier, monotone descent reaches a fixed point in finitely many cell removals; by Tarski, that fixed point is the greatest post-fixed certified subset representable on the partition. $\blacksquare$

**Realized verifier version.** With the learned $V_\theta$, $\mathrm{Pre}_{\Phi}^{\mathrm{all}}$ is not exactly computable; we compute a **sound inner approximation** $\underline{\mathrm{Pre}}_{\Phi}^{\mathrm{all}}$ via §5.7. Replacing $\mathcal T_{\Phi}$ by
$$
\underline{\mathcal T}_{\Phi}(S):=S\cap\mathcal K\cap\underline{\mathrm{Pre}}_{\Phi}^{\mathrm{all}}(S)
$$
keeps the sequence nested and sound but possibly conservative. Unknown cells are removed or left uncertified; they are never counted as safe.

> **Scope.** Only the shrinking direction is sound. Counterexample-guided **re-training to grow** $\Omega_c$ is empirical (retraining can break verified cells) and carries no soundness claim.

### 5.10 The discounted-training vs. undiscounted-certificate gap (made concrete)

The backbone (Theorem B) holds for the **undiscounted** fixed point (1.5). Training (4.1) targets a **discounted** value with $\gamma_{\mathrm{ENV}}$. The certificate is insulated from this gap, and one can see exactly where:

- Training produces some network $V_\theta$ — it need not equal $V^{\star}$ or the discounted value;
- Theorem A's hypotheses (C1)–(C3) are checked on **that** $V_\theta$ with the true $f$;
- the invariance margin in (C2)/(C3) is set by the deployment decay $\beta$, **not** by $\gamma_{\mathrm{ENV}}$.

Hence one must **not** identify $\gamma_{\mathrm{ENV}}$ with $\beta$ (nor with any $e^{-\alpha\Delta t}$): they live in different equations — $\gamma_{\mathrm{ENV}}$ in the training residual (4.1), $\beta$ in the runtime constraint (5.1) and the verified conditions (C2)/(C3). **[TODO]** pin the exact $\beta\!\leftrightarrow\!\alpha$ constant once the discrete-time CBVF form is fixed (the bridge of §2.4 with the chosen $\Delta t$); this is the only place a discount enters the *analysis*, and it connects the certificate to Herbert-style CBVF theory (§2.3).

---

### 5.11 Deployment-calibrated feasibility-survival certificate (main scalable lane)

The deterministic certificate above is the strongest statement in this document — it is a true full-$\mathcal D$ robustness guarantee — but it requires composing the dynamics $f$ through the learned certificate and discharging uniform conditions over state cells. That full-$\mathcal D$ discharge is the right microscope for F1TENTH / toy systems; at high dimension it becomes impractical or vacuous on this object class (shown empirically by the deterministic-exit contrast, framework E3 — *measured*, not asserted). The scalable lane therefore certifies the **deployed (optimistic)** object's **native failure event** on frozen closed-loop rollouts, and quantifies its optimism separately. This is the high-dimensional, Story B side of the §5.1 placement note.

Freeze the deployed closed loop before calibration:
$$
\mathfrak C_\theta
:=
\left(V_\theta,Q_\theta,\pi^\flat,u^\star,\widehat{\mathcal D}_{\mathrm{dep}},\widehat{\mathcal D}_{\mathrm{cal}},\text{environment/opponent policy}\right).
\tag{5.11}
$$
After this freeze, no part of $\mathfrak C_\theta$ may be tuned using calibration rollouts.

For a state $x$, the **deployed (optimistic)** feasibility predicate uses the deployed set $\widehat{\mathcal D}_{\mathrm{dep}}$:
$$
\Phi_\theta(x,u):
\quad
\inf_{\xi\in\widehat{\mathcal D}_{\mathrm{dep}}(x)}
Q_\theta(x,u,\xi)
\ge
\beta\!\left(V_\theta(x)\right),
\qquad\Bigl(=\Phi_\psi(x,u)\ \text{when}\ \widehat{\mathcal D}_{\mathrm{dep}}=\{d_\psi\}\Bigr)
\tag{5.12}
$$
and the safe-action-set survival event of the deployed object is
$$
\mathrm{Feas}_{\mathrm{dep}}(x)
:=
\left\{\exists u\in\mathcal U:\Phi_\theta(x,u)\right\}
\qquad(\text{i.e. }\mathcal F_\psi(x)\neq\varnothing).
\tag{5.13}
$$
The object-native trajectory score is evaluated **offline at calibration time against the affordable $\widehat{\mathcal D}_{\mathrm{cal}}$** ($\widehat{\mathcal D}_{\mathrm{dep}}\subseteq\widehat{\mathcal D}_{\mathrm{cal}}\subseteq\mathcal D$):
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
\tag{5.14}
$$
Thus $S_{\mathrm{feas}}(\tau)=0$ exactly when a $\widehat{\mathcal D}_{\mathrm{cal}}$-feasible action exists at every step of the horizon. **The meaning is set by $\widehat{\mathcal D}_{\mathrm{cal}}$:** with $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$ (low-dimensional lane) this is survival of the *true robust* feasible set $\mathcal F_{\mathrm{rob}}$; with $\widehat{\mathcal D}_{\mathrm{cal}}\subsetneq\mathcal D$ (high-dimensional lane) it is survival of the *deployed/calibration* feasible set, and the residual to $\mathcal F_{\mathrm{rob}}$ is the false-feasibility gap that Remark F4 validates. A cheaper sufficient score pins the deployed fallback witness:
$$
S_{\flat}(\tau)
:=
\max_{0\le t<T}
\left[
\beta(V_\theta(x_t))
-
\inf_{\xi\in\widehat{\mathcal D}_{\mathrm{cal}}(x_t)}
Q_\theta(x_t,\pi^\flat(x_t),\xi)
\right]_+ .
\tag{5.15}
$$
If $S_{\flat}(\tau)=0$, then the fallback itself remains feasible at every step, hence $\mathrm{Feas}_{\mathrm{dep}}(x_t)$ holds for every $t<T$.

Let
$$
\tau_1,\ldots,\tau_N
$$
be exchangeable calibration rollouts from the frozen closed-loop deployment distribution $\mathbb P_{\mathrm{deploy}}$. For either score $S\in\{S_{\mathrm{feas}},S_{\flat}\}$, define
$$
S_i:=S(\tau_i),
\qquad
k_\alpha:=\left\lceil (N+1)(1-\alpha)\right\rceil,
\qquad
q_\alpha:=S_{(k_\alpha)},
\tag{5.16}
$$
where $S_{(k)}$ is the $k$-th order statistic, with the usual convention that $q_\alpha=+\infty$ if $k_\alpha>N$.

> **Theorem F (split-conformal feasibility-survival coverage).** Under exchangeability of the calibration rollouts and a fresh deployment rollout $\tau\sim\mathbb P_{\mathrm{deploy}}$ from the same frozen closed loop,
> $$
> \Pr_{\tau\sim\mathbb P_{\mathrm{deploy}}}
> \left[
> S(\tau)\le q_\alpha
> \right]
> \ge
> 1-\alpha .
> \tag{5.17}
> $$
> In particular, if $S=S_{\mathrm{feas}}$ and $q_\alpha=0$, then
> $$
> \Pr_{\tau\sim\mathbb P_{\mathrm{deploy}}}
> \left[
> \mathrm{Feas}_{\mathrm{dep}}(x_t)\ \forall\,0\le t<T
> \right]
> \ge
> 1-\alpha .
> \tag{5.18}
> $$
> If $S=S_{\flat}$ and $q_\alpha=0$, then the same conclusion holds with the stronger event that the specific fallback $\pi^\flat(x_t)$ is feasible for every $t<T$. **Reading (Story B):** (5.18) certifies survival of the *deployed/calibration* feasible set $\mathcal F_\psi$. It upgrades to survival of the *true robust* feasible set $\mathcal F_{\mathrm{rob}}$ **iff** $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$; otherwise the gap to $\mathcal F_{\mathrm{rob}}$ is the false-feasibility residual (Remark F4).

**Proof.** This is the standard split-conformal rank argument applied to the scalar trajectory score $S(\tau)$. Exchangeability implies that the rank of the fresh score among
$$
S_1,\ldots,S_N,S(\tau)
$$
is uniform up to ties under the conservative order-statistic convention. Therefore
$$
\Pr\!\left[S(\tau)\le S_{(k_\alpha)}\right]
\ge
\frac{k_\alpha}{N+1}
\ge
1-\alpha .
$$
Equations (5.18) follows from the definition of $S_{\mathrm{feas}}$: $S_{\mathrm{feas}}(\tau)=0$ iff the $\widehat{\mathcal D}_{\mathrm{cal}}$-feasibility margin is nonnegative at every time, which is exactly $\mathrm{Feas}_{\mathrm{dep}}(x_t)$ for all $t<T$. The fallback-pinned case follows similarly from (5.15). $\blacksquare$

> **Remark F1 (guarantee type — what this certifies, Story B).** Theorem F is not a deterministic invariance theorem and does not imply safety for every disturbance sequence. It is a finite-sample, distributional certificate that the **deployed (optimistic)** safe-action set $\mathcal F_\psi$ survives the frozen deployment distribution. It is **not** a claim of true robust safety against $\mathcal D$ unless $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$. The honest object of the paper is the pair *(this survival certificate of $\mathcal F_\psi$)* **+** *(the structural and empirical characterization of the gap $\mathcal F_\psi\setminus\mathcal F_{\mathrm{rob}}$)* — Theorem S / Cor. S3 for the structure, §5.2–§5.10 (low-dim) and Remark F4 (high-dim) for the measurement. This is a feature of the scalable lane, not a replacement proof of Theorem A.

> **Remark F2 (numerical score computation).** If the inner $\inf_\xi$ or outer $\sup_u$ in (5.14) is approximated by a solver, then Theorem F applies to the **computed score**. To claim exact feasibility survival, the score computation itself must be exact or conservatively bounded. F1TENTH / toy systems are the place to audit this approximation against deterministic full-$\mathcal D$ checks.

> **Remark F3 (strategic or performative drift).** If the opponent or environment policy changes after calibration, the exchangeability assumption is broken. Theorem F then no longer guarantees coverage. Such shifts may be studied empirically as stress tests, but the theorem is only for the frozen closed loop.

> **Remark F4 (high-dimensional validation of the false-feasibility mechanism — consideration; realization deferred).** In the high-dimensional lane, $\widehat{\mathcal D}_{\mathrm{cal}}\subsetneq\mathcal D$ and there is **no full ground truth** for $\mathcal F_{\mathrm{rob}}$ — that is the regime's defining feature. Two consequences follow that are *not* covered by Theorem F and must be handled by validation rather than by the theorem. (i) **Soundness of the meaning:** because $\widehat{\mathcal D}_{\mathrm{cal}}\subseteq\mathcal D$, the computed $S_{\mathrm{feas}}$ can *under*-state the true feasibility shortfall, so coverage of the computed score is not automatically coverage of the true-robust event. (ii) **Attribution:** without a check, one cannot assert the certificate is tracking the *false-feasibility* mechanism (Theorem S) rather than generic feasibility loss. The remedy is an **offline, partial worst-case probe applied to a sampled subset of visited states only** (it need not scale, since it is neither run online nor over all states): a richer worst-case search over $\mathcal D$ on those samples to estimate the residual $\mathcal F_{\widehat{\mathcal D}_{\mathrm{cal}}}\setminus\mathcal F_{\mathrm{rob}}$ that $\widehat{\mathcal D}_{\mathrm{cal}}$ misses; or a platform with a known worst-case disturbance on a subspace giving partial ground truth; or injection of the offline-found worst $\xi$ to confirm flagged states collapse the fallback. The validation **succeeds** if the high-dimensional $S_{\mathrm{feas}}$ flags the states the probe confirms as falsely feasible and the missed residual is small relative to the certified margin. This is what licenses reading the high-dimensional certificate as being about the false-feasibility mechanism; absent it, the high-dimensional claim is the weaker "the certificate runs and is non-vacuous here." Concrete technical realization (probe budget, sampling, the worst-case search itself) is an implementation-stage matter and is deferred.

---

## 6. Status of the chain: proved · conditional · empirical

| Result | Statement | Status |
|---|---|---|
| Thm 1 | robust DCBF $\Rightarrow$ robust forward invariance (all $\mathbf d\in\mathcal D^{\infty}$) | **proved**, full |
| Lemma C | the CBVF solving the HJI-VI is a valid robust CBF for every online $\gamma\ge\lambda$; in particular every $\gamma\ge0$ when $\lambda=0$ | **proved**, full (from (2.5)) |
| §2.4 | continuous rate $\alpha\mapsto$ one-step decay $\beta$ with $\beta(r)\le r$ | **derived** (class-$\mathcal{KL}$ flow) |
| Lemma Q | $V^{\star}=\max_u\min_d Q^{\star}$ | **proved**, full |
| Lemma B1 | fallback $\pi^{\flat}$ is feasible on $\Omega^{\star}$ | **proved**, full |
| Lemma B2 | Q-CBF constraint $\Leftrightarrow$ $V$-form (both directions) | **proved**, full |
| **Thm B** | $V^{\star}$ is the maximal robust DCBF on $\Omega^{\star}$ | **proved**, full |
| Prop R | robust Q-CBF filter is recursively feasible | **proved**, full |
| Prop L | learned plug-in adversary $\Rightarrow$ **local** (policy-ball) robustness only | **proved**, full; the limitation we remove |
| Prop U | $\mathcal F_{\mathrm{rob}}\subseteq\mathcal F_\psi$ — learned-adversary certificate is unsound | **proved**, full |
| **Theorem S** | every continuous plug-in adversary misses switching worst cases under abutting strict-minimizer regions: $q_\psi>q_{\min}$ on a nonempty open set | **proved**, full; conditional necessity result for the continuous learned-adversary shortcut |
| **Cor. S3** | if the threshold lands in the Theorem S value gap with $\mu$-margin, then $\mathcal F_{\mathrm{rob}}\subsetneq\mathcal F_\psi$ and false-feasible actions have positive measure | **proved**, conditional on threshold landing; E0 / an appendix diagnostic should show this regime occurs on the trained object |
| **Thm A** | (C1)–(C3) $\Rightarrow$ deployed-filter recursive feasibility + robust invariance + safety | **proved**, full; **no** approximation hypothesis |
| §5.5 | runtime $\beta$ decoupled from certified invariance via direct (C2) + $c$-search | **argued**, clean |
| T6 / Lemma W | witness compresses $\forall\exists\forall\to\forall\forall$; sound for *any* $\pi^{\flat}$ directly verified by (C3) | **proved** as a quantifier reduction; composition-margin calculation is training rationale, not a soundness hypothesis |
| Prop T2 | full-$\mathcal D$ discharge adds disturbance-coordinate bounding/branching relative to the same NN verifier over $(x,u)$; no HJI-style $(x,d)$ grid | **argued**, defensible complexity framing; encoding via Lemma E |
| Lemma E | sound implication encoding via the antecedent margin $a(x,u)$ and consequent margin $b(x,u,d)$ | **proved for soundness**; termination/completeness only under positive margins and vanishing bound gaps |
| Cor. M (T5) | combined uncertainty set $\mathcal D_{\mathrm{aug}}$ covers model error $\Rightarrow$ sound under true dynamics | **proved**, full; deterministic source of $\mathcal D_{\mathrm{aug}}$ `[OPEN]` |
| Prop T4 | monotone shrink-refinement; finite iterates are viability approximants, fixed points are invariant | **proved on finite-cell verifier lattice**; sound inner-predecessor implementation is conservative |
| §5.10 | certificate insulated from $\gamma_{\mathrm{ENV}}$ (deployed-object framing) | **argued**; exact $\beta\!\leftrightarrow\!\alpha$ constant `[TODO]` |
| **Thm F** | split-conformal coverage for $S_{\mathrm{feas}}$ or $S_{\flat}$ on frozen deployment rollouts; if $q_\alpha=0$, **deployed (optimistic)** safe-action-set / fallback survival holds with prob. $\ge 1-\alpha$ | **proved**, finite-sample distributional; depends on exchangeability and leakage-free freeze, not on a dynamics model. **Certifies $\mathcal F_\psi$, not $\mathcal F_{\mathrm{rob}}$, unless $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$** (Story B; Remark F1) |

**The non-theorems.** Three usefulness/scope claims remain empirical.

1. Whether the deterministic set $\Omega_c$ is **non-vacuous** — $\mathrm{Vol}(\Omega_c)/\mathrm{Vol}(\Omega^{\star})$ large enough to matter — is not a theorem and cannot be: it depends on the trained network and on bound tightness through the $V_\theta\!\circ f$ and $Q_\theta\!\circ\pi^{\flat}_\varphi$ compositions. The T6 composition-margin calculation explains how training can help, but does not replace direct verification.
2. Whether the conformal deployment certificate is **informative** — for example $q_\alpha=0$ or $q_\alpha$ small relative to the feasibility-margin scale — is also not a theorem. Theorem F guarantees coverage of the calibrated score under exchangeability; only experiment shows that the resulting bound is operationally useful.
3. Whether the high-dimensional certificate **is about the false-feasibility mechanism** (and whether the computed $S_{\mathrm{feas}}$ under-states the true shortfall) is not a theorem either, because $\mathcal F_{\mathrm{rob}}$ is not computable there. It is established by **validation** (Remark F4), not proof. Relatedly, that full-$\mathcal D$ discharge is **impractical/vacuous at deployment dimension on this object class** is an *empirical* claim shown by the deterministic-exit contrast (framework E3), not a proposition.

Thus Theorem A guarantees deterministic (true-robust) soundness when its conditions are verified, and Theorem F guarantees finite-sample distributional coverage of the **deployed** object when the frozen-loop assumptions hold. **Only experiment establishes non-vacuity, informativeness, and — in high dimension — that the certificate tracks the false-feasibility mechanism.**

---

### Conventions inherited by the experiment section

1. Every experimental claim instantiates one row of the table above.
2. **False feasibility is the central diagnostic:** look for states/actions where
   $$
   \Phi_\psi(x,u)=1,
   \qquad
   \Phi_{\mathrm{rob}}(x,u)=0 .
   $$
3. **Deterministic soundness is binary** on the low-dimensional lane: zero certified-but-violated states under the worst case over the *full* $\mathcal D$ (verified offline; falsified online by gradient search and random/worst-case $d$).
4. **Conformal validity is distributional** on the scalable lane: calibration and evaluation rollouts must be exchangeable samples from the same frozen closed loop; otherwise Theorem F is not the claimed guarantee.
5. **Set size / score informativeness** is reported operationally: deterministic $\Omega_c$ as a fraction of grid-HJI $\Omega^{\star}$ where available, and conformal $q_\alpha$ relative to the feasibility-margin scale and fallback-collapse frequency.
6. **"Robust"** always means the game $\min_u\max_d$ (equivalently $\sup_u\inf_d$) over the entire relevant uncertainty set — never merely against a learned adversary (Prop U).
7. Neither deterministic nor conformal certification assumes $Q_\theta=Q^{\star}$. The deterministic lane verifies inequalities directly; the conformal lane calibrates the frozen deployed score directly.
8. **Story B set hierarchy (binding on every claim).** $\widehat{\mathcal D}_{\mathrm{dep}}\subseteq\widehat{\mathcal D}_{\mathrm{cal}}\subseteq\mathcal D$. The conformal certificate is for the **deployed** object ($\mathcal F_\psi$, against $\widehat{\mathcal D}_{\mathrm{cal}}$); it is a true-robustness statement **iff** $\widehat{\mathcal D}_{\mathrm{cal}}=\mathcal D$ (low-dim lane). High-dim ($\widehat{\mathcal D}_{\mathrm{cal}}\subsetneq\mathcal D$) certifies survival of the optimistic object; the false-feasibility gap to $\mathcal F_{\mathrm{rob}}$ is *validated* (Remark F4), not certified. State which set every reported feasibility/score is computed against.
