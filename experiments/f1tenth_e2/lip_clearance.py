"""Hard 1-Lipschitz clearance head  C_theta(px,py)  (spectral-norm-projected ReLU).

WHY (proven by run_cert_rho.py's exact level probe): a SOFT directional-Lipschitz
penalty leaves C_theta with a worst-case directional Lipschitz constant ~1.6-1.8 in
a thin pocket (the clearance medial-axis kink), and a 15x stronger penalty does not
fix it.  Because the braking gain  D(v)-D(v+)  is analytically EXACT and equals
dt*v = ||p+ - p||_2, ANY discounted-closure deficit is exactly C_theta's directional
Lipschitz overshoot beyond 1.  So the structured object is certifiable IFF C_theta is
worst-case 1-Lipschitz -- a HARD constraint, not a soft penalty.

HOW: if every linear layer has spectral norm ||W||_2 <= 1 and the activation is
1-Lipschitz (ReLU), the whole net is 1-Lipschitz in l2:
        |C_theta(p) - C_theta(p+)|  <=  ||p - p+||_2  =  dt*v      (everywhere).
Then the cancellation  V(f)-V = [C(p+)-C(p)] + [D(v)-D(v+)] >= -dt*v + dt*v = 0 is
exact, the {V>=c} level set closes, and the throughput verifier lands rho>0.

We enforce it by PROJECTED training: after each Adam step divide each weight by its
spectral norm if it exceeds 1.  The certified object is the EXPORTED weights, on which
the per-layer spectral norms are re-verified here (training is not in the trusted
base; the 1-Lipschitz property is a checkable fact about the frozen weights).
"""
from __future__ import annotations

import numpy as np

from qcbf.nets.mlp import MLP, Adam
from qcbf.verify.bounds import SeqNet
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import clearance


def _spectral_project(W: np.ndarray) -> np.ndarray:
    """Scale W so its spectral norm (largest singular value) is <= 1."""
    s = np.linalg.norm(W, 2)            # exact largest singular value (small W)
    return W / s if s > 1.0 else W


def spectral_norms(net) -> list[float]:
    return [float(np.linalg.norm(W, 2)) for W in net.W]


def train_lip_clearance_net(cfg, seed=0, width=128, depth=2, epochs=60,
                            batch=1024, lr=1e-3, n=160_000, verbose=True):
    """Fit C_theta ~ clearance under a HARD per-layer spectral-norm<=1 constraint
    (=> globally 1-Lipschitz in l2).  Returns (SeqNet, mse, max_spectral_norm).

    Wide + shallow is deliberate: capacity (width) raises fit without raising the
    Lipschitz budget (which depends on spectral norms, not width), while few layers
    limit the sub-multiplicative Lipschitz attenuation that would over-smooth the
    clearance ridge.
    """
    rng = np.random.default_rng(seed)
    sizes = [2] + [width] * depth + [1]
    C = MLP(sizes, seed=seed)
    for i in range(len(C.W)):                       # start strictly feasible
        C.W[i] = _spectral_project(C.W[i])
    opt = Adam(C, lr=lr)
    P = rng.uniform(cfg.p_lo, cfg.p_hi, (n, 2))
    Y = clearance(cfg, P[:, 0], P[:, 1]).reshape(-1, 1)
    pb = Progress(epochs, "1-Lip C_theta") if verbose else None
    for ep in range(epochs):
        o = rng.permutation(n)
        for s in range(0, n, batch):
            idx = o[s:s + batch]
            p = P[idx]
            B = len(idx)
            out, zs, hs = C.forward(p, cache=True)
            gW, gb, _ = C.backward(zs, hs, 2.0 * (out - Y[idx]) / B)
            opt.step(gW, gb)
            for i in range(len(C.W)):               # PROJECT onto ||W||_2 <= 1
                C.W[i] = _spectral_project(C.W[i])
        if pb is not None:
            pb.update(ep + 1)
    if pb is not None:
        pb.done()
    for i in range(len(C.W)):                        # final guarantee
        C.W[i] = _spectral_project(C.W[i])
    sn = spectral_norms(C)
    assert max(sn) <= 1.0 + 1e-9, f"spectral norm constraint violated: {sn}"
    mse = float(np.mean((C(P)[:, 0] - Y.ravel()) ** 2))
    return SeqNet.from_mlp(C), mse, max(sn)
