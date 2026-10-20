"""The zero-term baseline convention.

Asserts that the CREO component enters only through the stored repair
displacement and consumes no random numbers, so that the beta = 0 arm is the
unmodified host and a matched pair differs by the beta term alone.

Run: python test_zero_term_equivalence.py
"""
import numpy as np
from hosts import Problem, run, HOSTS


def rastrigin(D=10):
    return Problem('Rastrigin', D, np.full(D, -5.12), np.full(D, 5.12),
                   lambda x: float(10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))),
                   optimum_x=np.zeros(D))


def test_beta_is_inert_when_displacement_is_zero():
    """repair_op=None makes dr identically zero; beta must then change nothing."""
    failures = []
    for h in HOSTS:
        a = run(h, rastrigin(10), None, beta=0.0, seed=7, budget=4000)
        b = run(h, rastrigin(10), None, beta=0.9, seed=7, budget=4000)
        if (a['best_f'], a['nfe']) != (b['best_f'], b['nfe']):
            failures.append((h, a['best_f'], b['best_f']))
    assert not failures, f"beta consumes randomness or leaks in: {failures}"


def test_runs_are_deterministic_given_a_seed():
    for h in HOSTS:
        a = run(h, rastrigin(10), None, beta=0.0, seed=3, budget=3000)
        b = run(h, rastrigin(10), None, beta=0.0, seed=3, budget=3000)
        assert a['best_f'] == b['best_f'], f"{h} is not reproducible from its seed"


if __name__ == "__main__":
    test_beta_is_inert_when_displacement_is_zero()
    test_runs_are_deterministic_given_a_seed()
    print("PASS: beta is inert when dr == 0, and every host is seed-reproducible.")
