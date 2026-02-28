"""
Benchmark baseline for Python calc.py functions.
Profiles all major function categories using timeit.
Results written to benchmarks/results_python.json.
"""
import json
import timeit
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.calc import (
    _bsm_euro, _barrier_amer, _barrier_euro,
    _euro_vanilla_greeks, _euro_barrier_amer_greeks,
    _euro_barrier_euro_greeks, newton_raphson
)

# Common test parameters (from test_calc.py)
TAU = 233 / 365 + 1 / 365
VOL = 0.22
S = 387.750
R = 0.0
PRODUCT = 'C'
LOTS = 10


def bench_vanilla_pricing():
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420, 440,
               440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    chars = ['call'] * 9 + ['put'] * 9
    for k, c in zip(strikes, chars):
        _bsm_euro(c, TAU, VOL, k, S, R)


def bench_amer_barrier_pricing():
    chars = ['put'] * 4 + ['call'] * 4
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']
    for i in range(len(chars)):
        _barrier_amer(chars[i], TAU, VOL, strikes[i], S, R,
                      'amer', directions[i], kis[i], kos[i])


def bench_euro_barrier_pricing():
    chars = ['call'] * 2 + ['put'] * 2
    directions = ['up'] * 2 + ['down'] * 2
    kis = [None, 390, None, 370]
    kos = [400, None, 350, None]
    strikes = [350, 360, 390, 400]
    for i in range(len(chars)):
        _barrier_euro(chars[i], TAU, VOL, strikes[i], S, R,
                      'amer', directions[i], kis[i], kos[i],
                      PRODUCT, bvol=VOL, bvol2=VOL)


def bench_vanilla_greeks():
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420, 440,
               440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    chars = ['call'] * 9 + ['put'] * 9
    for k, c in zip(strikes, chars):
        _euro_vanilla_greeks(c, k, TAU, VOL, S, R, PRODUCT, LOTS)


def bench_amer_barrier_greeks():
    chars = ['put'] * 4 + ['call'] * 4
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']
    for i in range(len(chars)):
        _euro_barrier_amer_greeks(chars[i], TAU, VOL, strikes[i], S, R,
                                  'amer', directions[i], PRODUCT,
                                  kis[i], kos[i], LOTS)


def bench_euro_barrier_greeks():
    chars = ['call', 'call', 'put', 'put']
    directions = ['up', 'up', 'down', 'down']
    kis = [None, 390, None, 385]
    kos = [390, None, 380, None]
    strikes = [350, 380, 395, 395]
    for i in range(len(chars)):
        _euro_barrier_euro_greeks(chars[i], TAU, VOL, strikes[i], S, R,
                                  'amer', directions[i], PRODUCT,
                                  kis[i], kos[i], LOTS,
                                  bvol=VOL, bvol2=VOL)


def bench_iv_solver():
    # Get some option prices to solve IV for
    test_cases = [
        ('call', 387.75, _bsm_euro('call', TAU, VOL, 387.75, S, R)),
        ('put', 387.75, _bsm_euro('put', TAU, VOL, 387.75, S, R)),
        ('call', 350, _bsm_euro('call', TAU, VOL, 350, S, R)),
        ('put', 420, _bsm_euro('put', TAU, VOL, 420, S, R)),
    ]
    for char, k, price in test_cases:
        newton_raphson(char, S, k, price, TAU, R)


BENCHMARKS = {
    'vanilla_pricing': bench_vanilla_pricing,
    'amer_barrier_pricing': bench_amer_barrier_pricing,
    'euro_barrier_pricing': bench_euro_barrier_pricing,
    'vanilla_greeks': bench_vanilla_greeks,
    'amer_barrier_greeks': bench_amer_barrier_greeks,
    'euro_barrier_greeks': bench_euro_barrier_greeks,
    'iv_solver': bench_iv_solver,
}

ITERATIONS = [100, 1_000, 10_000]


def run_benchmarks():
    results = {}
    for name, func in BENCHMARKS.items():
        results[name] = {}
        for n in ITERATIONS:
            t = timeit.timeit(func, number=n)
            avg_us = (t / n) * 1e6  # microseconds per call
            results[name][str(n)] = {
                'total_s': round(t, 6),
                'avg_us': round(avg_us, 3),
            }
            print(f"  {name:30s} | {n:>6d} iters | {t:8.4f}s total | {avg_us:10.3f} us/call")
    return results


if __name__ == '__main__':
    print("=" * 80)
    print("Python calc.py Benchmark Baseline")
    print("=" * 80)
    results = run_benchmarks()
    out_path = os.path.join(os.path.dirname(__file__), 'results_python.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
