"""
Benchmark comparison: Python calc.py vs Rust rust_calc.

Runs the identical benchmark matrix using both backends and prints
a comparison table with speedup ratios.
"""
import json
import timeit
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.calc import (
    _bsm_euro as py_bsm_euro,
    _barrier_amer as py_barrier_amer,
    _barrier_euro as py_barrier_euro,
    _euro_vanilla_greeks as py_euro_vanilla_greeks,
    _euro_barrier_amer_greeks as py_euro_barrier_amer_greeks,
    _euro_barrier_euro_greeks as py_euro_barrier_euro_greeks,
    newton_raphson as py_newton_raphson,
)

import rust_calc

# Common test parameters
TAU = 233 / 365 + 1 / 365
VOL = 0.22
S = 387.750
R = 0.0
PRODUCT = 'C'
LOTS = 10


# ── Benchmark functions ───────────────────────────────────────────────

def make_vanilla_pricing(bsm_fn):
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420, 440,
               440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    chars = ['call'] * 9 + ['put'] * 9
    def bench():
        for k, c in zip(strikes, chars):
            bsm_fn(c, TAU, VOL, k, S, R)
    return bench


def make_amer_barrier_pricing(barrier_fn):
    chars = ['put'] * 4 + ['call'] * 4
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']
    def bench():
        for i in range(len(chars)):
            barrier_fn(chars[i], TAU, VOL, strikes[i], S, R,
                       'amer', directions[i], kis[i], kos[i])
    return bench


def make_euro_barrier_pricing(barrier_fn):
    chars = ['call'] * 2 + ['put'] * 2
    directions = ['up'] * 2 + ['down'] * 2
    kis = [None, 390, None, 370]
    kos = [400, None, 350, None]
    strikes = [350, 360, 390, 400]
    def bench():
        for i in range(len(chars)):
            barrier_fn(chars[i], TAU, VOL, strikes[i], S, R,
                       'amer', directions[i], kis[i], kos[i],
                       PRODUCT, bvol=VOL, bvol2=VOL)
    return bench


def make_vanilla_greeks(greeks_fn):
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420, 440,
               440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    chars = ['call'] * 9 + ['put'] * 9
    def bench():
        for k, c in zip(strikes, chars):
            greeks_fn(c, k, TAU, VOL, S, R, PRODUCT, LOTS)
    return bench


def make_amer_barrier_greeks(greeks_fn):
    chars = ['put'] * 4 + ['call'] * 4
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']
    def bench():
        for i in range(len(chars)):
            greeks_fn(chars[i], TAU, VOL, strikes[i], S, R,
                      'amer', directions[i], PRODUCT, kis[i], kos[i], LOTS)
    return bench


def make_euro_barrier_greeks(greeks_fn):
    chars = ['call', 'call', 'put', 'put']
    directions = ['up', 'up', 'down', 'down']
    kis = [None, 390, None, 385]
    kos = [390, None, 380, None]
    strikes = [350, 380, 395, 395]
    def bench():
        for i in range(len(chars)):
            greeks_fn(chars[i], TAU, VOL, strikes[i], S, R,
                      'amer', directions[i], PRODUCT, kis[i], kos[i], LOTS,
                      bvol=VOL, bvol2=VOL)
    return bench


def make_iv_solver(nr_fn, bsm_fn):
    test_cases = [
        ('call', 387.75, bsm_fn('call', TAU, VOL, 387.75, S, R)),
        ('put', 387.75, bsm_fn('put', TAU, VOL, 387.75, S, R)),
        ('call', 350, bsm_fn('call', TAU, VOL, 350, S, R)),
        ('put', 420, bsm_fn('put', TAU, VOL, 420, S, R)),
    ]
    def bench():
        for char, k, price in test_cases:
            nr_fn(char, S, k, price, TAU, R)
    return bench


# Rust wrappers to match Python signatures
def rs_barrier_euro(char, tau, vol, k, s, r, payoff, direction, ki, ko, product, bvol=None, bvol2=None):
    return rust_calc.barrier_euro(char, tau, vol, k, s, r, payoff, direction,
                                  ki=ki, ko=ko, product=product, bvol=bvol, bvol2=bvol2)

def rs_euro_barrier_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots, bvol=None, bvol2=None):
    return rust_calc.euro_barrier_euro_greeks(char, tau, vol, k, s, r, payoff, direction, product,
                                              ki=ki, ko=ko, lots=lots, bvol=bvol, bvol2=bvol2)

def rs_barrier_amer(char, tau, vol, k, s, r, payoff, direction, ki, ko):
    return rust_calc.barrier_amer(char, tau, vol, k, s, r, payoff, direction, ki=ki, ko=ko)

def rs_amer_barrier_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots):
    return rust_calc.euro_barrier_amer_greeks(char, tau, vol, k, s, r, payoff, direction, product,
                                              ki=ki, ko=ko, lots=lots)


BENCHMARKS = {
    'vanilla_pricing': (
        make_vanilla_pricing(py_bsm_euro),
        make_vanilla_pricing(rust_calc.bsm_euro),
    ),
    'amer_barrier_pricing': (
        make_amer_barrier_pricing(py_barrier_amer),
        make_amer_barrier_pricing(rs_barrier_amer),
    ),
    'euro_barrier_pricing': (
        make_euro_barrier_pricing(py_barrier_euro),
        make_euro_barrier_pricing(rs_barrier_euro),
    ),
    'vanilla_greeks': (
        make_vanilla_greeks(py_euro_vanilla_greeks),
        make_vanilla_greeks(rust_calc.euro_vanilla_greeks),
    ),
    'amer_barrier_greeks': (
        make_amer_barrier_greeks(py_euro_barrier_amer_greeks),
        make_amer_barrier_greeks(rs_amer_barrier_greeks),
    ),
    'euro_barrier_greeks': (
        make_euro_barrier_greeks(py_euro_barrier_euro_greeks),
        make_euro_barrier_greeks(rs_euro_barrier_greeks),
    ),
    'iv_solver': (
        make_iv_solver(py_newton_raphson, py_bsm_euro),
        make_iv_solver(rust_calc.newton_raphson, rust_calc.bsm_euro),
    ),
}

ITERATIONS = [100, 1_000, 10_000]


def run_benchmarks():
    results = {}
    print(f"\n{'Group':30s} | {'Iters':>6s} | {'Python (us)':>12s} | {'Rust (us)':>12s} | {'Speedup':>8s}")
    print("-" * 80)

    for name, (py_fn, rs_fn) in BENCHMARKS.items():
        results[name] = {}
        for n in ITERATIONS:
            t_py = timeit.timeit(py_fn, number=n)
            t_rs = timeit.timeit(rs_fn, number=n)
            avg_py = (t_py / n) * 1e6
            avg_rs = (t_rs / n) * 1e6
            speedup = avg_py / avg_rs if avg_rs > 0 else float('inf')
            results[name][str(n)] = {
                'python_us': round(avg_py, 3),
                'rust_us': round(avg_rs, 3),
                'speedup': round(speedup, 2),
            }
            print(f"  {name:30s} | {n:>6d} | {avg_py:>12.3f} | {avg_rs:>12.3f} | {speedup:>7.2f}x")

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("Python vs Rust Benchmark Comparison")
    print("=" * 80)
    results = run_benchmarks()
    out_path = os.path.join(os.path.dirname(__file__), 'results_compare.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
