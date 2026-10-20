"""Host algorithms behind one component interface.

Every host proposes a candidate step; the CREO component adds exactly one term,
beta * dr, where dr is the repair displacement stored with the individual at its
last evaluation. Setting beta = 0 removes that term and draws no random numbers
differently, so the beta = 0 arm is stream-identical to the unmodified host
(asserted in test_zero_term_equivalence.py).

Selection is feasibility-aware (Deb's rules) in every host, so a matched pair
differs by the beta term and nothing else.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np


# --------------------------------------------------------------- problem API
@dataclass
class Problem:
    name: str
    dim: int
    lo: np.ndarray
    hi: np.ndarray
    objective: Callable[[np.ndarray], float]
    constraint_violation: Callable[[np.ndarray], float] = lambda x: 0.0
    optimum_x: Optional[np.ndarray] = None

    def clip(self, x):
        return np.clip(x, self.lo, self.hi)


@dataclass
class Record:
    """One evaluated individual."""
    x_stored: np.ndarray        # the point kept in the population (pre-repair, clipped)
    x_used: np.ndarray          # the point actually evaluated (post-repair)
    dr: np.ndarray              # retained repair displacement
    f: float
    viol: float

    @property
    def feasible(self):
        return self.viol <= 1e-12

    def key(self):
        return (0 if self.feasible else 1, self.viol, self.f)


def better(a: Record, b: Record) -> bool:
    return a.key() <= b.key()


class Evaluator:
    """Applies the repair operator, evaluates, and counts the budget."""

    def __init__(self, problem: Problem, repair_op, budget: int):
        self.p, self.op, self.budget = problem, repair_op, budget
        self.nfe = 0
        self.repair_fired = 0
        self.trace: List[float] = []
        self.best: Optional[Record] = None

    @property
    def exhausted(self):
        return self.nfe >= self.budget

    def __call__(self, x, x_prev=None) -> Record:
        xs = self.p.clip(np.asarray(x, dtype=float))
        xr = self.op.repair(xs, x_prev) if self.op is not None else xs
        xr = self.p.clip(xr)
        dr = xr - xs
        if np.linalg.norm(dr) > 1e-12:
            self.repair_fired += 1
        f = float(self.p.objective(xr))
        v = float(self.p.constraint_violation(xr))
        self.nfe += 1
        rec = Record(xs, xr, dr, f, v)
        if self.best is None or better(rec, self.best):
            self.best = rec
        self.trace.append(self.best.f)
        return rec


# ------------------------------------------------------------------- helpers
def _init(ev, rng, n):
    p = ev.p
    pop = [p.lo + rng.random(p.dim) * (p.hi - p.lo) for _ in range(n)]
    return [ev(x) for x in pop]


def _result(ev, pop):
    return dict(best_f=ev.best.f, best_viol=ev.best.viol, nfe=ev.nfe,
                repair_fire_rate=ev.repair_fired / max(ev.nfe, 1), trace=ev.trace)


# --------------------------------------------------------------------- hosts
def de_rand_1_bin(ev, rng, beta, N=50, F=0.7, CR=0.9):
    pop = _init(ev, rng, N)
    while not ev.exhausted:
        for i in range(N):
            if ev.exhausted:
                break
            r1, r2, r3 = rng.choice([j for j in range(N) if j != i], 3, replace=False)
            mut = (pop[r1].x_stored + F * (pop[r2].x_stored - pop[r3].x_stored)
                   + beta * pop[i].dr)
            jr = rng.integers(ev.p.dim)
            cross = rng.random(ev.p.dim) < CR
            cross[jr] = True
            trial = np.where(cross, mut, pop[i].x_stored)
            t = ev(trial, pop[i].x_stored)
            if better(t, pop[i]):
                pop[i] = t
    return _result(ev, pop)


def de_current_to_pbest(ev, rng, beta, N=50, F=0.7, CR=0.9, p_frac=0.1):
    pop = _init(ev, rng, N)
    while not ev.exhausted:
        order = sorted(range(N), key=lambda j: pop[j].key())
        top = order[:max(1, int(p_frac * N))]
        for i in range(N):
            if ev.exhausted:
                break
            pb = pop[rng.choice(top)]
            r1, r2 = rng.choice([j for j in range(N) if j != i], 2, replace=False)
            mut = (pop[i].x_stored + F * (pb.x_stored - pop[i].x_stored)
                   + F * (pop[r1].x_stored - pop[r2].x_stored) + beta * pop[i].dr)
            jr = rng.integers(ev.p.dim)
            cross = rng.random(ev.p.dim) < CR
            cross[jr] = True
            trial = np.where(cross, mut, pop[i].x_stored)
            t = ev(trial, pop[i].x_stored)
            if better(t, pop[i]):
                pop[i] = t
    return _result(ev, pop)


def lshade(ev, rng, beta, N_init=None, H=6, p_frac=0.11):
    """L-SHADE: success-history adaptation plus linear population reduction."""
    D = ev.p.dim
    N_init = N_init or 18 * D
    N_min, budget = 4, ev.budget
    pop = _init(ev, rng, N_init)
    MF = np.full(H, 0.5)
    MCR = np.full(H, 0.5)
    k = 0
    while not ev.exhausted:
        N = len(pop)
        order = sorted(range(N), key=lambda j: pop[j].key())
        top = order[:max(1, int(p_frac * N))]
        SF, SCR, dF = [], [], []
        for i in range(N):
            if ev.exhausted:
                break
            ri = rng.integers(H)
            CR = float(np.clip(rng.normal(MCR[ri], 0.1), 0, 1))
            F = 0.0
            while F <= 0:
                F = float(np.clip(MF[ri] + 0.1 * np.tan(np.pi * (rng.random() - 0.5)), -1, 1))
            pb = pop[rng.choice(top)]
            r1, r2 = rng.choice([j for j in range(N) if j != i], 2, replace=False)
            mut = (pop[i].x_stored + F * (pb.x_stored - pop[i].x_stored)
                   + F * (pop[r1].x_stored - pop[r2].x_stored) + beta * pop[i].dr)
            jr = rng.integers(D)
            cross = rng.random(D) < CR
            cross[jr] = True
            t = ev(np.where(cross, mut, pop[i].x_stored), pop[i].x_stored)
            if better(t, pop[i]):
                if t.key() < pop[i].key():
                    SF.append(F); SCR.append(CR); dF.append(abs(pop[i].f - t.f))
                pop[i] = t
        if SF:
            w = np.array(dF) + 1e-30
            w = w / w.sum()
            MF[k] = float(np.sum(w * np.array(SF) ** 2) / np.sum(w * np.array(SF)))
            MCR[k] = float(np.sum(w * np.array(SCR)))
            k = (k + 1) % H
        target = max(N_min, int(round(N_init + (N_min - N_init) * ev.nfe / budget)))
        if target < len(pop):
            pop = [pop[j] for j in sorted(range(len(pop)), key=lambda j: pop[j].key())[:target]]
    return _result(ev, pop)


def spso2011(ev, rng, beta, N=40, w=0.721, c=1.193):
    p = ev.p
    pop = _init(ev, rng, N)
    vel = [rng.uniform(-1, 1, p.dim) * (p.hi - p.lo) * 0.1 for _ in range(N)]
    pbest = list(pop)
    gi = min(range(N), key=lambda j: pop[j].key())
    gbest = pbest[gi]
    while not ev.exhausted:
        for i in range(N):
            if ev.exhausted:
                break
            r1, r2 = rng.random(p.dim), rng.random(p.dim)
            vel[i] = (w * vel[i] + c * r1 * (pbest[i].x_stored - pop[i].x_stored)
                      + c * r2 * (gbest.x_stored - pop[i].x_stored))
            cand = pop[i].x_stored + vel[i] + beta * pop[i].dr
            t = ev(cand, pop[i].x_stored)
            pop[i] = t
            if better(t, pbest[i]):
                pbest[i] = t
                if better(t, gbest):
                    gbest = t
    return _result(ev, pop)


def ga_sbx(ev, rng, beta, N=50, eta_c=20.0, eta_m=20.0, pm=None):
    p = ev.p
    pm = pm if pm is not None else 1.0 / p.dim
    pop = _init(ev, rng, N)
    while not ev.exhausted:
        new = []
        for _ in range(N // 2):
            if ev.exhausted:
                break
            par = []
            for _ in range(2):                                  # binary tournament
                a, b = rng.choice(N, 2, replace=False)
                par.append(pop[a] if better(pop[a], pop[b]) else pop[b])
            u = rng.random(p.dim)
            bq = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                          (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
            x1, x2 = par[0].x_stored, par[1].x_stored
            c1 = 0.5 * ((1 + bq) * x1 + (1 - bq) * x2)
            c2 = 0.5 * ((1 - bq) * x1 + (1 + bq) * x2)
            for child, parent in ((c1, par[0]), (c2, par[1])):
                mask = rng.random(p.dim) < pm
                delta = rng.random(p.dim)
                dq = np.where(delta < 0.5, (2 * delta) ** (1 / (eta_m + 1)) - 1,
                              1 - (2 * (1 - delta)) ** (1 / (eta_m + 1)))
                child = child + mask * dq * (p.hi - p.lo) + beta * parent.dr
                if ev.exhausted:
                    break
                new.append(ev(child, parent.x_stored))
        pool = sorted(pop + new, key=lambda r: r.key())[:N]
        pop = pool
    return _result(ev, pop)


def gwo(ev, rng, beta, N=40):
    """Grey Wolf Optimiser. Retained as a widely used host; its update is a
    documented reiteration of particle-swarm ideas (Camacho-Villalon et al.),
    and it is included because it is the strongest host on the E06 loop, not as
    a novel algorithm."""
    p = ev.p
    pop = _init(ev, rng, N)
    while not ev.exhausted:
        order = sorted(range(N), key=lambda j: pop[j].key())
        A_, B_, C_ = (pop[order[0]].x_stored, pop[order[1]].x_stored, pop[order[2]].x_stored)
        a = 2.0 * (1.0 - ev.nfe / ev.budget)
        for i in range(N):
            if ev.exhausted:
                break
            xs = []
            for leader in (A_, B_, C_):
                A = 2 * a * rng.random(p.dim) - a
                C = 2 * rng.random(p.dim)
                xs.append(leader - A * np.abs(C * leader - pop[i].x_stored))
            cand = np.mean(xs, axis=0) + beta * pop[i].dr
            t = ev(cand, pop[i].x_stored)
            if better(t, pop[i]):
                pop[i] = t
    return _result(ev, pop)


def eps_de(ev, rng, beta, N=50, F=0.7, CR=0.9, cp=5.0):
    """epsilon-constrained DE (Takahama & Sakai): the constrained-suite
    reference method. The epsilon level relaxes feasibility early and tightens
    to zero over a control generation horizon."""
    pop = _init(ev, rng, N)
    viols = sorted(r.viol for r in pop)
    eps0 = viols[int(0.2 * len(viols))]
    Tc = 0.2 * ev.budget

    def eps_at(nfe):
        if nfe >= Tc:
            return 0.0
        return eps0 * (1 - nfe / Tc) ** cp

    def eps_better(a, b, e):
        ka = (a.f, a.viol) if a.viol <= e else (np.inf, a.viol)
        kb = (b.f, b.viol) if b.viol <= e else (np.inf, b.viol)
        return (ka[1] if ka[0] == np.inf else ka[0], ka[1]) <= (kb[1] if kb[0] == np.inf else kb[0], kb[1])

    while not ev.exhausted:
        e = eps_at(ev.nfe)
        for i in range(N):
            if ev.exhausted:
                break
            r1, r2, r3 = rng.choice([j for j in range(N) if j != i], 3, replace=False)
            mut = (pop[r1].x_stored + F * (pop[r2].x_stored - pop[r3].x_stored)
                   + beta * pop[i].dr)
            jr = rng.integers(ev.p.dim)
            cross = rng.random(ev.p.dim) < CR
            cross[jr] = True
            t = ev(np.where(cross, mut, pop[i].x_stored), pop[i].x_stored)
            if eps_better(t, pop[i], e):
                pop[i] = t
    return _result(ev, pop)


def cmaes_component(ev, rng, beta, sigma0_frac=0.3):
    """CMA-ES with the component added to each sampled candidate.

    Uses the reference `cma` implementation for the distribution update; the
    CREO term is added to the sampled point before evaluation, matching how the
    term enters the other hosts.
    """
    import cma
    p = ev.p
    x0 = (p.lo + p.hi) / 2
    sigma0 = float(np.mean(p.hi - p.lo) * sigma0_frac)
    recs = []
    popsize = 4 + int(3 * np.log(p.dim))
    while not ev.exhausted:                      # IPOP-style restarts
        es = cma.CMAEvolutionStrategy(
            list(x0), sigma0,
            {'bounds': [list(p.lo), list(p.hi)], 'verbose': -9,
             'popsize': popsize,
             'seed': int(rng.integers(1, 2 ** 31 - 1)),
             'maxfevals': ev.budget - ev.nfe})
        last_dr = {}
        while not ev.exhausted and not es.stop():
            asks = es.ask()
            fits = []
            for j, x in enumerate(asks):
                if ev.exhausted:
                    break
                cand = np.asarray(x) + beta * last_dr.get(j, np.zeros(p.dim))
                # CMA-ES has no per-individual predecessor; the distribution mean
                # is the operating point the sampled step departs from.
                r = ev(cand, np.asarray(es.mean, dtype=float))
                last_dr[j] = r.dr
                recs.append(r)
                fits.append(r.f + 1e6 * r.viol)
            if len(fits) == len(asks):
                es.tell(asks, fits)
        popsize *= 2
        x0 = p.lo + rng.random(p.dim) * (p.hi - p.lo)
    return _result(ev, recs)


HOSTS = {
    'DE/rand/1/bin': de_rand_1_bin,
    'DE/current-to-pbest/1': de_current_to_pbest,
    'L-SHADE': lshade,
    'SPSO-2011': spso2011,
    'GA (SBX + polynomial)': ga_sbx,
    'GWO': gwo,
    'eps-constrained DE': eps_de,
    'CMA-ES': cmaes_component,
}


def run(host_name, problem, repair_op, beta, seed, budget):
    rng = np.random.default_rng(seed)
    ev = Evaluator(problem, repair_op, budget)
    out = HOSTS[host_name](ev, rng, beta)
    out.update(host=host_name, problem=problem.name, dim=problem.dim,
               beta=beta, seed=seed,
               repair_op=(repair_op.name if repair_op is not None else 'none'))
    return out
