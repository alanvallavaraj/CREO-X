"""Repair operators of differing displacement geometry, parameterised by the box.

The CEC2017 CREO protocol uses a preferred region of [-80, 80] inside a domain of
[-100, 100] and a mean-absolute threshold of 60. Those are 0.8 and 0.6 of the box
half-width, so the box-relative parameterisation below reproduces the original
operator exactly when lo = -100, hi = 100.

What differs between operators is what dr = R(x) - x encodes:

    RadialShrinkage      R(x) = c + 0.8^k (x - c): dr is collinear with -(x - c)
                         and carries one degree of freedom. The CEC2017 operator.
    BoxProjection        axis-aligned; dr supported only on violated coordinates.
    EllipsoidProjection  exact nearest-feasible projection; dr is the true normal.
    RateLimit            trust region around the previous point; dr is the refused
                         component of the requested step (anti-windup).
    RandomResample       null operator; dr is noise with no geometric content.
"""
from typing import Optional
import numpy as np
from scipy.optimize import brentq


class RepairOperator:
    name = "base"
    informative = None

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.c = (self.lo + self.hi) / 2.0
        self.w = (self.hi - self.lo) / 2.0

    def clip_domain(self, x):
        return np.clip(x, self.lo, self.hi)

    def violation(self, x, x_prev=None) -> float:
        raise NotImplementedError

    def repair(self, x, x_prev=None):
        raise NotImplementedError

    def sample_feasible(self, rng, n=1):
        raise NotImplementedError

    def displacement(self, x, x_prev=None):
        xs = self.clip_domain(np.asarray(x, dtype=float))
        return self.repair(xs, x_prev) - xs


class RadialShrinkage(RepairOperator):
    name = "radial shrinkage (CEC2017)"
    informative = False

    def __init__(self, lo, hi, soft_frac=0.8, mean_frac=0.6, shrink=0.80, max_passes=10):
        super().__init__(lo, hi)
        self.soft = soft_frac * self.w
        self.mean_thresh = mean_frac * float(np.mean(self.w))
        self.shrink, self.max_passes = shrink, max_passes

    def violation(self, x, x_prev=None):
        z = np.asarray(x, dtype=float) - self.c
        v = float(np.sum(np.maximum(0.0, -self.soft - z) + np.maximum(0.0, z - self.soft)))
        ma = float(np.mean(np.abs(z)))
        if ma > self.mean_thresh:
            v += (ma - self.mean_thresh) * len(z)
        return v

    def repair(self, x, x_prev=None):
        y = self.clip_domain(np.array(x, dtype=float))
        if self.violation(y) <= 1e-12:
            return y
        for _ in range(self.max_passes):
            y = self.c + self.shrink * (y - self.c)
            y = self.clip_domain(y)
            if self.violation(y) <= 1e-12:
                break
        return y

    def sample_feasible(self, rng, n=1):
        D = len(self.c)
        out = np.empty((n, D)); i = 0
        while i < n:
            cand = self.c + rng.uniform(-1, 1, D) * self.soft
            if self.violation(cand) <= 1e-12:
                out[i] = cand; i += 1
        return out


class BoxProjection(RepairOperator):
    name = "box projection"
    informative = True

    def __init__(self, lo, hi, soft_frac=0.8):
        super().__init__(lo, hi)
        self.soft = soft_frac * self.w

    def violation(self, x, x_prev=None):
        z = np.asarray(x, dtype=float) - self.c
        return float(np.sum(np.maximum(0.0, -self.soft - z) + np.maximum(0.0, z - self.soft)))

    def repair(self, x, x_prev=None):
        return np.clip(x, self.c - self.soft, self.c + self.soft)

    def sample_feasible(self, rng, n=1):
        D = len(self.c)
        return self.c + rng.uniform(-1, 1, (n, D)) * self.soft


class EllipsoidProjection(RepairOperator):
    name = "ellipsoid projection"
    informative = True

    def __init__(self, lo, hi, rng, centre_frac=0.3, axis_lo=0.2, axis_hi=0.6):
        super().__init__(lo, hi)
        D = len(self.c)
        self.m = self.c + rng.uniform(-centre_frac, centre_frac, D) * self.w
        axes = rng.uniform(axis_lo, axis_hi, D) * self.w
        self.wgt = 1.0 / axes ** 2
        Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
        self.Q = Q

    def _z(self, x):
        return self.Q.T @ (np.asarray(x, dtype=float) - self.m)

    def violation(self, x, x_prev=None):
        z = self._z(x)
        return max(0.0, float(np.sum(self.wgt * z ** 2) - 1.0))

    def repair(self, x, x_prev=None):
        z = self._z(x)
        if float(np.sum(self.wgt * z ** 2)) <= 1.0 + 1e-12:
            return np.array(x, dtype=float)

        def g(lam):
            return float(np.sum(self.wgt * (z / (1.0 + lam * self.wgt)) ** 2)) - 1.0

        hi_ = 1.0
        while g(hi_) > 0 and hi_ < 1e14:
            hi_ *= 2.0
        lam = brentq(g, 0.0, hi_, xtol=1e-14, rtol=1e-15)
        return self.m + self.Q @ (z / (1.0 + lam * self.wgt))

    def sample_feasible(self, rng, n=1):
        D = len(self.c)
        u = rng.normal(size=(n, D)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        r = rng.uniform(size=(n, 1)) ** (1.0 / D)
        return self.m + (u * r / np.sqrt(self.wgt)) @ self.Q.T


class RateLimit(RepairOperator):
    name = "rate limit (E06)"
    informative = True

    def __init__(self, lo, hi, h_frac=0.10):
        super().__init__(lo, hi)
        self.h = h_frac * (self.hi - self.lo)

    def violation(self, x, x_prev=None):
        if x_prev is None:
            return 0.0
        return float(np.sum(np.maximum(0.0, np.abs(np.asarray(x) - x_prev) - self.h)))

    def repair(self, x, x_prev=None):
        if x_prev is None:
            return np.array(x, dtype=float)
        step = np.asarray(x, dtype=float) - x_prev
        return x_prev + np.clip(step, -self.h, self.h)

    def sample_feasible(self, rng, n=1):
        D = len(self.c)
        return self.c + rng.uniform(-0.8, 0.8, (n, D)) * self.w


class RandomResample(RepairOperator):
    name = "random resample (null)"
    informative = False

    def __init__(self, lo, hi, rng=None, soft_frac=0.8):
        super().__init__(lo, hi)
        self.soft = soft_frac * self.w
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def violation(self, x, x_prev=None):
        z = np.asarray(x, dtype=float) - self.c
        return float(np.sum(np.maximum(0.0, -self.soft - z) + np.maximum(0.0, z - self.soft)))

    def repair(self, x, x_prev=None):
        if self.violation(x) <= 1e-12:
            return np.array(x, dtype=float)
        return self.c + self.rng.uniform(-1, 1, len(self.c)) * self.soft

    def sample_feasible(self, rng, n=1):
        D = len(self.c)
        return self.c + rng.uniform(-1, 1, (n, D)) * self.soft


def make_battery(lo, hi, rng):
    """The five operators, scaled to the given box."""
    return [RadialShrinkage(lo, hi), BoxProjection(lo, hi),
            EllipsoidProjection(lo, hi, rng), RateLimit(lo, hi),
            RandomResample(lo, hi, rng=rng)]


OPERATOR_NAMES = ["radial shrinkage (CEC2017)", "box projection",
                  "ellipsoid projection", "rate limit (E06)",
                  "random resample (null)"]
