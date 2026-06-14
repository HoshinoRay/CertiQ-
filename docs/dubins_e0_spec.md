# Dubins E0 Clean Specification

## Model

The active plant is the fixed-speed Dubins car

\[
x=(p_x,p_y,\psi),\qquad
u\in[-\omega_{\max},\omega_{\max}],\qquad
d\in[-d_{\max},d_{\max}],
\]

with discrete dynamics

\[
p_x^+=p_x+\Delta t\,v\cos\psi,\qquad
p_y^+=p_y+\Delta t\,v\sin\psi,
\]

\[
\psi^+=\operatorname{wrap}\!\left(\psi+\Delta t\,(u+d)\right).
\]

The safety set is

\[
\mathcal K=\{x:g(x)\ge0\},
\]

where \(g\) is the minimum of obstacle clearance and workspace clearance.

## Ground Truth (teacher)

Two **distinct, decoupled** knobs are used:

- \(\gamma_{\mathrm{deploy}}\in(0,1]\): the deployed/certified per-step CBF decay,
  the only one the certificate and runtime use (\(\approx0.90\)).
- \(\lambda=\gamma_{\mathrm{teach}}\in(0,1)\): the *discount* in the teacher's
  discounted safety backup, used only to solve the labels (\(\approx0.92\)).

The grid oracle computes \(V_{\mathrm{HJ}}\) by the discounted safety value
iteration (Fisac / Akametalu)

\[
F[V](x)=\max_{u\in\mathcal U}\min_{d\in\mathcal D}V(f(x,u,d)),
\qquad
V^{k+1}(x)=(1-\lambda)\,g(x)+\lambda\,\min\{g(x),\,F[V^k](x)\}.
\]

The discount makes the backup a \(\lambda\)-contraction; the \((1-\lambda)g\) source
term pins \(V\) near \(g\) on the safe interior, which keeps the multilinear-interp
minimax from draining the avoid value to \(g_{\mathrm{fail}}\) (the undiscounted,
the \(\min(g,\gamma F)\), and the \(\min(g,F/\gamma)\) forms all degenerate here).
The result is a non-empty, resolution-robust \(\Omega^\star=\{V_{\mathrm{HJ}}\ge0\}\).

The teacher is an **under-approximation, not an exact CBF**: it carries a positive
deployed margin \(\sim(1-\gamma_{\mathrm{deploy}})V\) on the safe interior but can be
slightly negative on the \(V=0\) boundary shell.  This only affects which cells the
*verifier* can later certify; it never affects soundness.

The paired action value is the successor value:

\[
Q_{\mathrm{HJ}}(x,u,d)=V_{\mathrm{HJ}}(f(x,u,d)).
\]

The oracle is used for supervised labels and the reference volume only, not as a
proof assumption.

## Learned Artifact

\(V\) and \(Q\) are learned by **separate** networks (no
\(V=\max_u\min_d Q\) identity is imposed), distilling

\[
V_\theta\approx V_{\mathrm{HJ}},
\qquad
Q_\theta\approx Q_{\mathrm{HJ}},
\qquad
\pi_\theta\approx\arg\max_u\min_d Q_{\mathrm{HJ}}(x,u,d).
\]

The witness-margin fine-tuning (against \(\gamma_{\mathrm{deploy}}\)) only improves
the chance that \(C3\) certifies; it is not a safety proof.

## Certification (Theorem A)

The frozen artifact \((V_\theta,Q_\theta,\pi_\theta)\) is certified directly at
\(\gamma_{\mathrm{deploy}}\).  The verifier checks:

\[
C1:\quad \{x:V_\theta(x)\ge0\}\subseteq\mathcal K,
\]

\[
C3:\quad
\min_d Q_\theta(x,\pi_\theta(x),d)\ge\gamma_{\mathrm{deploy}}V_\theta(x)+\varepsilon,
\]

\[
C4:\quad
\min_d Q_\theta(x,u,d)
\le
\min_d V_\theta(f(x,u,d))
\quad\text{for the menu actions and the witness action.}
\]

The robust one-step decrease
\(V_\theta(f(x,u,d))\ge\gamma_{\mathrm{deploy}}V_\theta(x)\) is implied by the runtime
gate \(\min_d Q_\theta\ge\gamma_{\mathrm{deploy}}V_\theta\) together with \(C4\), so it
is not a separate obligation; the old existential \(C2\) is therefore dropped.

The certified volume is reported as an inner-cell lower bound divided by
\(\operatorname{vol}(\Omega^\star)\). Unknown cells are never counted as safe.
