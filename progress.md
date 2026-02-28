# Rust Port Progress — `rust-calc-port` branch

## Overview

Ported all pure-numerical functions from `scripts/calc.py` (1,290 lines of Python) to ~750 lines of Rust across 7 modules, exposed via PyO3 as a drop-in replacement. The original `calc.py` is untouched; a new `scripts/calc_rs.py` wrapper provides identical function signatures with automatic fallback to pure Python if the Rust extension is not installed.

---

## 1. Branch & Toolchain Setup

- Created `rust-calc-port` branch from `master`.
- Updated Rust toolchain from 1.64.0 (2022) to 1.93.1 (2026).

## 2. Project Scaffolding

- Created `rust_calc/` directory with `Cargo.toml`, `pyproject.toml`, and full module structure.
- Updated `.gitignore` to exclude `target/` and `.venv/`.

## 3. Python Benchmark Baseline

- Created `benchmarks/bench_calc.py` profiling all 7 major function categories across 100 / 1,000 / 10,000 iterations.
- Results saved to `benchmarks/results_python.json`.

## 4. Rust Implementation

Seven modules under `rust_calc/src/`:

| Module | Contents |
|---|---|
| `types.rs` | `OptionChar` and `Direction` enums (replacing string params) |
| `multipliers.rs` | `Multipliers` struct + static `HashMap` for all 18 commodity products |
| `bsm.rs` | `_bsm_euro` vanilla European pricing + `_compute_value` dispatcher |
| `barrier_amer.rs` | Haug closed-form American barrier pricing (16-branch decision tree) + `A_B`, `C_D`, `E_f`, `F_f` helpers |
| `barrier_euro.rs` | European barrier pricing via call-spread + digital replication, `digital_option`, `call_put_spread` |
| `greeks.rs` | All greeks: `greeks_scaled`, `_euro_vanilla_greeks`, `_euro_barrier_amer_greeks`, `digital_greeks`, `_euro_barrier_euro_greeks`, `_compute_greeks`, `call_put_spread_greeks` |
| `iv.rs` | Newton-Raphson IV solver, `_compute_iv`, `compute_strike_from_delta` |
| `lib.rs` | PyO3 module registration exposing 16 Python-callable functions |

## 5. Tests

- 7/7 Rust unit tests pass (`cargo test`), validated against the same test vectors from `tests/test_calc.py`.

## 6. Python Integration Layer

- Created `scripts/calc_rs.py` — drop-in replacement wrapping Rust functions with identical signatures.
- Falls back to pure Python (`scripts/calc.py`) if `rust_calc` is not installed.
- Original `calc.py` is completely untouched.

## 7. Benchmark Comparison

Created `benchmarks/bench_compare.py`. Results at 10,000 iterations:

| Function | Python (us) | Rust (us) | Speedup |
|---|---|---|---|
| Vanilla pricing | 1,672 | 2.38 | **702x** |
| American barrier pricing | 2,279 | 2.69 | **846x** |
| European barrier pricing | 1,443 | 1.67 | **866x** |
| Vanilla greeks | 1,834 | 3.37 | **545x** |
| American barrier greeks | 13,740 | 9.56 | **1,438x** |
| European barrier greeks | 1,445 | 2.08 | **695x** |
| IV solver | 1,315 | 1.03 | **1,280x** |

## 8. Study Guide

Created `STUDY_GUIDE.md` (~1,300 lines) covering:

- **Part 0** — Glossary of terms (MRO, PyObject, boxing, LLVM, inlining, Haug variables)
- **Part 1** — Financial math (BSM derivation, Haug reflection principle with drunk-walk intuition, European barrier replication, Newton-Raphson IV, finite difference greeks, scaling conventions)
- **Part 2** — Why Python is slow (scipy `norm.cdf` overhead, function call overhead, boxing, string comparisons, full cost breakdown)
- **Part 3** — Why Rust is fast (register f64s, LLVM inlining, branch elimination, direct `norm_cdf`, CPU pipelining, FFI boundary design)
- **Part 4** — Reading the Rust code (module map, naming conventions, key patterns)
- **Part 5** — Key takeaways
- **Part 6** — Compiler magic vs code structure analysis (~200x from scipy misuse, ~7x from CPython, ~3x from LLVM)

---

## Files Created / Modified

```
.gitignore                          (modified)
rust_calc/Cargo.toml
rust_calc/pyproject.toml
rust_calc/src/lib.rs
rust_calc/src/types.rs
rust_calc/src/multipliers.rs
rust_calc/src/bsm.rs
rust_calc/src/barrier_amer.rs
rust_calc/src/barrier_euro.rs
rust_calc/src/greeks.rs
rust_calc/src/iv.rs
scripts/calc_rs.py
benchmarks/bench_calc.py
benchmarks/bench_compare.py
benchmarks/results_python.json
benchmarks/results_compare.json
STUDY_GUIDE.md
```

## What Stays in Python (DataFrame-dependent)

These functions depend on pandas DataFrames and remain in `scripts/calc.py`:

- `get_barrier_vol`
- `get_vol_from_delta`
- `get_vol_at_strike`
- `compute_delta`

## What Was Skipped (Not In Use)

These functions are not currently used and were not ported:

- `_CRRbinomial`
- `_amer_option`
- `_amer_vanilla_greeks`
- `_num_vega`
- `american_iv`
