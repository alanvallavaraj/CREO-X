"""A distributable surrogate of the rate-constrained tracking loop.

Self-contained: no private components. The plant is a seeded random linear map with
optional smooth, multimodal or saturating distortion; the controller is DE/rand/1/bin with
the three CREO additions. The point of this file is that the beta dose-response reported
for the deployed loops can be checked by anyone, on a landscape family that is defined here
rather than obtained separately.

Structure that matters, and that a looser reimplementation gets wrong:
  * the rate limiter clips each candidate to a box of half-width delta about the single
    applied control, so the whole population shares one moving operating point;
  * the carried displacement is `repaired - raw`, which points inward;
  * survivor selection is greedy and the population stores the REPAIRED point, so the
    population persists across cycles instead of being rebuilt each cycle.
The carrying rule itself is NOT load-bearing: updating the carried displacement only for
accepted members, or unconditionally for all, both reproduce the dose-response, and the
unconditional variant is marginally stronger (see `surrogate_full.py`, arm `carry_all`).
An earlier reimplementation of this loop around a controller that rebuilds its whole
population every cycle did not reproduce the effect at all.
"""
from __future__ import annotations
import numpy as np

DIM, NPOP, CYCLES, DELTA = 6, 20, 15, 0.10
F, CR, ALPHA, GAMMA = 0.7, 0.9, 0.18, 0.02
KAPPA, TOL = 0.35, 0.10


def plant_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, DIM))
    return W / np.sum(np.abs(W), axis=1, keepdims=True)


def response(u, W, landscape: str) -> np.ndarray:
    x = 2.0 * np.clip(np.asarray(u, float), 0.0, 1.0) - 1.0
    y = W @ x
    if landscape == "linear":
        pass
    elif landscape == "smooth":
        y = y + np.array([0.18*x[3]*x[4] - 0.12*x[5]**2, 0.20*x[1]*x[2] + 0.15*x[3]**2])
    elif landscape == "multimodal":
        y = y + np.array([0.22*np.sin(3*np.pi*x[3])*np.cos(2*np.pi*x[4]),
                          0.22*np.sin(3*np.pi*x[2])*np.cos(2*np.pi*x[1])])
    elif landscape == "saturating":
        y = np.tanh(1.8 * (y + np.array([0.16*x[3]*x[4], 0.18*x[1]*x[2]])))
    else:
        raise ValueError(landscape)
    return np.clip(y, -1.0, 1.0)


def _repair(raw, u_prev):
    lo = np.maximum(0.0, u_prev - DELTA)
    hi = np.minimum(1.0, u_prev + DELTA)
    rep = np.clip(raw, lo, hi)
    return rep, rep - raw


def episode(beta: float, W, landscape: str, target, seed: int, obs_sigma: float = 0.0,
            max_slew: bool = False, carry: str = "accepted") -> dict:
    rng = np.random.default_rng(seed)
    u_app = np.full(DIM, 0.5)
    z = np.clip(-0.5 * np.asarray(target, float), -1.0, 1.0)
    zhat = z.copy()
    X = rng.random((NPOP, DIM))
    X, disp = _repair(X, u_app)
    vals = None
    dists, fired, steps = [], 0, []
    for c in range(CYCLES):
        def obj(u):
            pred = np.clip((1 - KAPPA) * zhat + KAPPA * response(u, W, landscape), -1, 1)
            return float(np.linalg.norm(pred - target))
        if vals is None:
            vals = np.array([obj(x) for x in X])
            carried = disp.copy()
        gbest = X[int(np.argmin(vals))]
        raw = np.empty_like(X)
        for i in range(NPOP):
            pool = np.delete(np.arange(NPOP), i)
            r1, r2, r3 = rng.choice(pool, 3, replace=False)
            mut = X[r1] + F * (X[r2] - X[r3]) + ALPHA * (gbest - X[i])
            mut = mut + beta * carried[i] + GAMMA * rng.normal(size=DIM)
            cross = rng.random(DIM) < CR
            cross[int(rng.integers(DIM))] = True
            raw[i] = np.where(cross, mut, X[i])
        rep, d = _repair(raw, u_app)
        fired += int(np.count_nonzero(np.any(np.abs(d) > 0, axis=1)))
        new = np.array([obj(x) for x in rep])
        imp = new < vals                      # greedy survivor selection
        X[imp], vals[imp] = rep[imp], new[imp]
        if carry == "accepted":
            carried[imp] = d[imp]             # rejected proposals keep the parent's
        elif carry == "all":
            carried = d.copy()                # overwrite regardless of acceptance
        else:
            raise ValueError(carry)
        step = X[int(np.argmin(vals))] - u_app
        if max_slew:
            m = float(np.max(np.abs(step)))
            if m > 1e-12:
                step = np.clip(step * (DELTA / m), -DELTA, DELTA)
        u_app = np.clip(u_app + step, 0.0, 1.0)
        steps.append(float(np.linalg.norm(step)))
        z = np.clip((1 - KAPPA) * z + KAPPA * response(u_app, W, landscape), -1, 1)
        zhat = z + obs_sigma * rng.normal(size=2) if obs_sigma else z.copy()
        dists.append(float(np.linalg.norm(z - target)))
    return dict(distance_auc=float(np.mean(dists)), success=bool(dists[-1] <= TOL),
                repair_rate=fired / (CYCLES * NPOP), mean_step=float(np.mean(steps)))
