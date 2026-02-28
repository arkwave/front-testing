# Refactor Plan: Portfolio, Option, Future Classes + Centralize Constants

## Context

The codebase has significant technical debt: no type hints, duplicated constants across 9 files, magic-index data structures, poor variable names, and complex methods. This plan targets the three core classes (Portfolio, Option, Future) and centralizes constants.

**User decisions:**
- Full rename everywhere (no backwards-compat aliases — rename `K`, `char`, etc. across ALL callers)
- Use `str, Enum` types now for option_type, barrier_style, etc.
- One PR with multiple commits (one commit per step)

**Dependencies required:** `pip install numpy pandas scipy plotly sqlalchemy joblib`

**Run tests with:** `python -m unittest discover -s tests -p "test_*.py" -v`

---

## Step 1: Create `scripts/constants.py` + Wire Up Imports Everywhere

**New file:** `scripts/constants.py`
**Modify:** all files that duplicate constants

### 1a. Contents of constants.py

**`ProductSpec` NamedTuple** — backwards-compatible with index access (`[0]`, `[-1]`, `[2:]`):
```python
from typing import NamedTuple, Dict
from enum import Enum

class ProductSpec(NamedTuple):
    dollar_mult: float
    lot_mult: float
    futures_tick: float
    options_tick: float
    pnl_mult: float

multipliers: Dict[str, ProductSpec] = {
    'LH':  ProductSpec(22.046, 18.143881, 0.025, 1, 400),
    'LSU': ProductSpec(1, 50, 0.1, 10, 50),
    'QC':  ProductSpec(1.2153, 10, 1, 25, 12.153),
    'SB':  ProductSpec(22.046, 50.802867, 0.01, 0.25, 1120),
    'CC':  ProductSpec(1, 10, 1, 50, 10),
    'CT':  ProductSpec(22.046, 22.679851, 0.01, 1, 500),
    'KC':  ProductSpec(22.046, 17.009888, 0.05, 2.5, 375),
    'W':   ProductSpec(0.3674333, 136.07911, 0.25, 10, 50),
    'S':   ProductSpec(0.3674333, 136.07911, 0.25, 10, 50),
    'C':   ProductSpec(0.393678571428571, 127.007166832986, 0.25, 10, 50),
    'BO':  ProductSpec(22.046, 27.215821, 0.01, 0.5, 600),
    'LC':  ProductSpec(22.046, 18.143881, 0.025, 1, 400),
    'LRC': ProductSpec(1, 10, 1, 50, 10),
    'KW':  ProductSpec(0.3674333, 136.07911, 0.25, 10, 50),
    'SM':  ProductSpec(1.1023113, 90.718447, 0.1, 5, 100),
    'COM': ProductSpec(1.0604, 50, 0.25, 2.5, 53.02),
    'CA':  ProductSpec(1.0604, 50, 0.25, 1, 53.02),
    'MW':  ProductSpec(0.3674333, 136.07911, 0.25, 10, 50),
}
```

**Other dicts:** `month_to_sym`, `sym_to_month`, `contract_mths`, `op_ticksize`

**Scalar constants:** `RANDOM_SEED = 7`, `DECADE = 10`, `TIMESTEP = 1/365`, `BREAKEVEN_FACTOR = 2.8`

**str Enums** (use `str, Enum` base so `BarrierStyle.AMERICAN == 'amer'` is `True`):
```python
class OptionType(str, Enum):
    CALL = 'call'
    PUT = 'put'

class BarrierStyle(str, Enum):
    AMERICAN = 'amer'
    EUROPEAN = 'euro'

class BarrierDirection(str, Enum):
    UP = 'up'
    DOWN = 'down'

class SecurityType(str, Enum):
    OPTION = 'option'
    FUTURE = 'future'

class PositionFlag(str, Enum):
    OTC = 'OTC'
    HEDGE = 'hedge'
```

### 1b. Bug fixes in contract_mths

**Missing comma in LC entry (Python string concat creates `'VZ'` instead of `'V', 'Z'`):**
- `scripts/simulation.py` line 89
- `scripts/fetch_data.py` line 33
- `demos/option_demo.py` line 58
- `tests/test_prep_data.py` (has same bug)
- `wip.py` (has same bug)
- CORRECT in: `scripts/util.py` line 66, `scripts/prep_data.py` line 61

**CT months inconsistency:** `fetch_data.py` has `['H', 'K', 'N', 'V', 'Z']` (5 months) while all other files have `['H', 'K', 'N', 'Z']` (4 months). Use the 4-month version.

### 1c. Where constants are currently duplicated

**multipliers** (9 copies):
- `scripts/simulation.py` lines 36-56
- `scripts/calc.py` lines 57-76
- `scripts/classes.py` lines 16-35
- `scripts/hedge.py` lines 14-33
- `scripts/util.py` lines 18-37
- `scripts/portfolio.py` lines 25-44
- `scripts/prep_data.py` lines 71-90
- `tests/test_calc.py` lines 9-28 (**KEEP LOCAL** — uses `LCC`/`OBM` legacy codes)
- `demos/option_demo.py` lines 10-29 (out of scope)

**month_to_sym** (4 copies): simulation.py:100, util.py:44, prep_data.py:37, option_demo.py:36
**sym_to_month** (4 copies): simulation.py:102, util.py:46, prep_data.py:39, option_demo.py:38
**contract_mths** (5+ copies): simulation.py:76-96, util.py:53-73, prep_data.py:48-68, fetch_data.py:20-40, option_demo.py:45-65
**op_ticksize** (2 copies): simulation.py:58-74, hedge.py:35-51
**seed = 7** (6 copies): simulation.py:109, calc.py:84, portfolio.py:47, util.py:39, prep_data.py:30, option_demo.py:31
**decade = 10** (4 copies): simulation.py:104, util.py:48, prep_data.py:41, option_demo.py:40

### 1d. What each file needs to import

| File | Imports needed |
|------|---------------|
| `scripts/classes.py` | `multipliers` |
| `scripts/calc.py` | `multipliers`, `RANDOM_SEED` |
| `scripts/portfolio.py` | `multipliers`, `RANDOM_SEED`, `BREAKEVEN_FACTOR` |
| `scripts/simulation.py` | `multipliers`, `contract_mths`, `month_to_sym`, `sym_to_month`, `op_ticksize`, `RANDOM_SEED`, `DECADE`, `TIMESTEP` |
| `scripts/hedge.py` | `multipliers`, `op_ticksize` |
| `scripts/util.py` | `multipliers`, `contract_mths`, `month_to_sym`, `sym_to_month`, `RANDOM_SEED`, `DECADE` |
| `scripts/prep_data.py` | `multipliers`, `contract_mths`, `month_to_sym`, `sym_to_month`, `RANDOM_SEED`, `DECADE` |
| `scripts/fetch_data.py` | `contract_mths` |

After replacing, keep `np.random.seed(RANDOM_SEED)` at module level in each file that currently has `np.random.seed(seed)`.

### Risk: LOW

---

## Step 2: Refactor `Future` class in `scripts/classes.py`

Currently lines 586-668.

### Changes:
1. **Type hints** on all methods
2. **Fix mutable default:** `instructions={}` on line 608 → remove parameter entirely (it's never read in the method body, and grep confirms no caller passes it)
3. **Use `SecurityType.FUTURE`** for `self.desc`
4. **Move calc imports to module top level** (currently lazy-imported in 4 Option methods). Verified safe: `calc.py` does NOT import `classes.py` — no circular dependency.
5. Clean up `__str__`

### Current Future API (preserve all):
- `__init__(month, price, product, shorted=None, lots=1000, ordering=None)`
- `get_price()`, `update_price(price)`, `get_month()`, `get_lots()`, `get_product()`
- `get_desc()`, `get_delta()`, `get_uid()`, `get_ordering()`, `set_ordering(i)`
- `decrement_ordering(i)`, `update_lots(lots)`

### Risk: LOW

---

## Step 3: Refactor `Option` class — Full Rename Across Codebase

Currently lines 38-584 in `scripts/classes.py`.

### 3a. Attribute renames (at source AND all callers)

| Old name | New name | Grep to find all references |
|----------|----------|-----------------------------|
| `self.K` / `.K` | `self.strike` / `.strike` | `grep -rn '\.K[^a-zA-Z]' scripts/ tests/` (careful: K appears in month codes too) |
| `self.char` / `.char` | `self.option_type` / `.option_type` | `grep -rn '\.char' scripts/ tests/` |
| `self.direc` / `.direc` | `self.direction` / `.direction` | `grep -rn '\.direc\b' scripts/ tests/` and `grep -rn 'direc=' scripts/ tests/` |
| `self.ki` / `.ki` | `self.knock_in` / `.knock_in` | `grep -rn '\.ki\b' scripts/ tests/` and `grep -rn 'ki=' scripts/ tests/` |
| `self.ko` / `.ko` | `self.knock_out` / `.knock_out` | `grep -rn '\.ko\b' scripts/ tests/` and `grep -rn 'ko=' scripts/ tests/` |
| `self.bvol` / `.bvol` | `self.barrier_vol` / `.barrier_vol` | `grep -rn '\.bvol\b' scripts/ tests/` and `grep -rn 'bvol=' scripts/ tests/` (careful: don't match bvol2) |
| `self.bvol2` / `.bvol2` | `self.barrier_vol2` / `.barrier_vol2` | `grep -rn '\.bvol2' scripts/ tests/` and `grep -rn 'bvol2=' scripts/ tests/` |
| `self.r` | Remove entirely — inline `0` | Only used inside `classes.py` in calls to calc functions |
| Constructor param `char=` | `option_type=` | Also rename in `_compute_greeks()` and `_compute_value()` param names in `calc.py` if they use `char` |

**IMPORTANT for `ki`/`ko`/`bvol`/`bvol2`:** These are also **keyword arguments** passed to `_compute_greeks()` and `_compute_value()` in `calc.py`. Check if `calc.py` function signatures use the same param names — if so, those must be renamed too, along with all callers passing them as kwargs.

### 3b. Type hints on all Option methods

Add `from __future__ import annotations` or use string quotes for forward references to `Future`.

### 3c. Extract duplicated Greeks computation

**`init_greeks()`** (lines 324-370) and **`update_greeks()`** (lines 372-419) both contain:
```python
ttms = [self.tau] if self.bullet else self.dailies
for tau in ttms:
    delta, gamma, theta, vega = _compute_greeks(self.char, self.K, tau, ...)
    d += delta; g += gamma; t += theta; v += vega
```

Extract to:
```python
def _sum_greeks_over_ttms(self, vol: float, barrier_vol, barrier_vol2) -> tuple:
    """Compute aggregate greeks across all TTM slices."""
    product = self.get_product()
    spot = self.underlying.get_price()
    ttms = [self.tau] if self.bullet else self.dailies
    total = [0.0, 0.0, 0.0, 0.0]
    for ttm in ttms:
        d, g, t, v = _compute_greeks(
            self.option_type, self.strike, ttm, vol, spot, 0, product,
            self.payoff, self.lots, ki=self.knock_in, ko=self.knock_out,
            barrier=self.barrier, direction=self.direction,
            order=self.ordering, bvol=barrier_vol, bvol2=barrier_vol2,
            dbarrier=self.dbarrier)
        total[0] += d; total[1] += g; total[2] += t; total[3] += v
    return tuple(total)
```

### 3d. Simplify `check_active()` (lines 245-313, 69 lines → ~30 lines)

**Current structure:** deeply nested if/else with state mutations scattered throughout.

**Proposed structure:**
```python
def check_active(self) -> bool:
    spot = self.underlying.get_price()
    if self.check_expired():
        return False
    if self.knockedin:
        return True
    if self.knockedout:
        return self.barrier == BarrierStyle.EUROPEAN and self.tau > 0
    if self.knock_in is not None:
        self._update_knock_in_state(spot)
    if self.knock_out is not None:
        return self._update_knock_out_state(spot)
    return True  # vanilla: active until expiry

def _update_knock_in_state(self, spot: float) -> None:
    if self.direction == 'up':
        self.knockedin = spot >= self.knock_in
    elif self.direction == 'down':
        self.knockedin = spot <= self.knock_in
    if self.barrier == 'amer' and self.knockedin:
        self.knock_in = None
        self.knock_out = None
        self.barrier = None
        self.direction = None

def _update_knock_out_state(self, spot: float) -> bool:
    if self.barrier == 'amer':
        if self.direction == 'up':
            active = spot < self.knock_out
        else:
            active = spot > self.knock_out
        self.knockedout = not active
        if self.knockedout:
            self.dailies = []
            self.expired = True
        return active
    elif self.barrier == 'euro':
        if self.direction == 'up':
            self.knockedout = spot >= self.knock_out
        else:
            self.knockedout = spot <= self.knock_out
        return True  # Euro KO stays active until expiry
    return True
```

**CRITICAL:** `check_active()` mutates `self.knockedin` and `self.knockedout` as side effects. The refactored version must preserve this exactly.

### 3e. Simplify `moneyness()` (lines 488-539, 52 lines → ~20 lines)

Extract helper:
```python
def _is_itm(self, spot: float) -> bool:
    if self.option_type == 'call':
        return self.strike < spot
    return self.strike > spot
```

Then `moneyness()` becomes:
```python
def moneyness(self) -> int:
    active = self.check_active()
    if self.knockedout:
        return -1
    self.active = active
    if not active:
        return -1
    spot = self.underlying.get_price()
    if self.strike == spot:
        return 0
    # Barrier KO check
    if self.knock_out is not None and self.knockedout:
        return -1
    # Barrier KI check — not yet knocked in means OTM
    if self.knock_in is not None and not self.knockedin:
        return -1
    return 1 if self._is_itm(spot) else -1
```

### 3f. Cleanup
- Remove debug `print()` statements in `init_greeks()` error handler (lines 348-360)
- Remove all commented-out code
- Use `SecurityType.OPTION` for `self.desc`

### Files to update for the renames

For each rename, grep and update ALL references. Key files:

- `scripts/classes.py` — source of truth
- `scripts/simulation.py` — accesses `.K`, `.char`, `.ki`, `.ko`, `.direc`, `.bvol`, `.bvol2` on Option objects throughout the simulation loop
- `scripts/hedge.py` — creates options and accesses barrier attributes
- `scripts/util.py` — factory functions pass `ki=`, `ko=`, `bvol=`, `direc=` as kwargs to Option constructor; also accesses `.K`, `.char`
- `scripts/calc.py` — function signatures use `char` param name, `ki=`, `ko=`, `bvol=`, `bvol2=` as params. **Must rename these function params too** and update all callers.
- `scripts/signals.py` — check for any attribute access
- `tests/test_options.py` — constructs Options with `ki=`, `ko=`, `direc=`, `barrier=`; accesses `.K`, `.char`
- `tests/test_calc.py` — calls calc functions with `ki=`, `ko=` kwargs
- `tests/test_portfolio.py` — constructs Options with `ki=`, `ko=`, `direc=`; accesses `.K`
- `tests/test_hedge.py` — constructs Options and accesses attributes
- `tests/test_simulation.py` — may reference option attributes

### Risk: MEDIUM-HIGH

---

## Step 4: Refactor `Portfolio` class in `scripts/portfolio.py`

Currently 1092 lines.

### 4a. Fix bugs

1. **Line 242 — `update_sec_lots` wrong list for hedge options:**
   ```python
   # BUG: else branch should be self.hedge_options, not self.hedge_futures
   ops = self.OTC_options if flag == 'OTC' else self.hedge_futures
   # FIX:
   ops = self.OTC_options if flag == 'OTC' else self.hedge_options
   ```

2. **Lines 247/249 — desc comparison case mismatch (dead code):**
   ```python
   # BUG: 'Option'/'Future' uppercase never matches get_desc() which returns lowercase
   if s.desc == 'Option' and s not in ops:
   # FIX:
   if s.desc == 'option' and s not in ops:
   ```

3. **Line 296 — `init_sec_by_month` writes to builtin `dict` instead of local `dic`:**
   ```python
   # BUG:
   dict[prod] = {}
   # FIX:
   dic[prod] = {}
   ```

4. **Line 301 — `update_greeks_by_month` missing `flag` argument:**
   ```python
   # BUG:
   self.update_greeks_by_month(prod, month, sec, True)
   # FIX:
   self.update_greeks_by_month(prod, month, sec, True, iden)
   ```

### 4b. Type hints + cleanup

- Add type hints to all methods
- Remove unused import: `from timeit import default_timer as timer` (line 13)
- Remove ~15 commented-out debug print statements (lines 323-324, 344, 431-432, 438, 442, 556, 660, 665, 679, 683)
- Remove unused `breakeven` params: `flag=None, conv=None` (line 1014, never read)

### 4c. Rename local variables

These are method-local, not public API:
- `dic` → `positions` or `target_dict` (used in ~20 methods)
- `pdt` → `product` (used in ~12 methods)
- `mth` → `month` (used in ~10 methods)
- `d3` → `net_greeks_dict` (line 580)
- `bes` → `breakevens` (line 1017)
- `op` / `ft` → `options` / `futures` (in local scope)
- `s` → `element` or `security` (lines 244, 923, 926)

### 4d. Extract flag-based dispatch

The pattern below appears ~10 times:
```python
if flag == 'OTC':
    dic = self.OTC
    op = self.OTC_options
    ft = self.OTC_futures
elif flag == 'hedge':
    dic = self.hedges
    op = self.hedge_options
    ft = self.hedge_futures
```

Extract to:
```python
def _get_position_lists(self, flag: str):
    """Return (options_deque, futures_list, positions_dict) for the given flag."""
    if flag == 'OTC':
        return self.OTC_options, self.OTC_futures, self.OTC
    return self.hedge_options, self.hedge_futures, self.hedges
```

Locations: lines 282-289, 389-394, 411-416, 490-499, 653, 725-728, 764-767, 772-777

### 4e. Simplify `compute_net_greeks()` (lines 313-383, 70 → ~20 lines)

Current code has 3 separate loops (common products, OTC-unique products, hedge-unique products) with nearly identical logic. Replace with:

```python
def compute_net_greeks(self) -> None:
    result = {}
    all_products = set(self.OTC) | set(self.hedges)
    for product in all_products:
        result[product] = {}
        otc_data = self.OTC.get(product, {})
        hedge_data = self.hedges.get(product, {})
        for month in set(otc_data) | set(hedge_data):
            otc_greeks = otc_data[month][2:] if month in otc_data and otc_data[month][0] else [0, 0, 0, 0]
            hedge_greeks = hedge_data[month][2:] if month in hedge_data and hedge_data[month][0] else [0, 0, 0, 0]
            result[product][month] = list(map(add, otc_greeks, hedge_greeks))
        if not result[product]:
            del result[product]
    self.net_greeks = result
```

### 4f. Decompose `update_sec_by_month()` (lines 474-622, 150 lines)

This method does 3 completely different things depending on parameters. Split:

```python
def update_sec_by_month(self, added, flag, update=None):
    """Thin dispatcher — preserves original call signature."""
    if update is not None:
        self._refresh_all_greeks(flag)
    elif added:
        self._add_securities_to_positions(flag)
    else:
        self._remove_securities_from_positions(flag)

def _add_securities_to_positions(self, flag: str) -> None:
    # Current lines 504-527

def _remove_securities_from_positions(self, flag: str) -> None:
    # Current lines 529-575

def _refresh_all_greeks(self, flag: str) -> None:
    # Current lines 578-622
```

### 4g. Use `BREAKEVEN_FACTOR` constant

Line 1030: replace magic `2.8` with `BREAKEVEN_FACTOR` imported from constants.

### 4h. Fix `remove_expired()` potential double-add (lines 446-472)

OTC loop uses `if`/`if` (option could be added twice), hedge loop uses `if`/`elif` (correct). Make both consistent with `elif`:
```python
# OTC loop (lines 450-460):
if sec.barrier == 'amer':
    if sec.knockedout:
        explist['OTC'].append(sec)
elif sec.check_expired():  # was 'if', should be 'elif'
    explist['OTC'].append(sec)
```

### Risk: MEDIUM-HIGH

---

## Step 5: Final Test Verification

1. Run full test suite: `python -m unittest discover -s tests -p "test_*.py" -v`
2. Verify all tests pass
3. If any test references old attribute names, update them

---

## Commit Plan (Single PR)

| Commit | Scope | Risk |
|--------|-------|------|
| 1 | `scripts/constants.py` + wire imports in all 8 consumer files | LOW |
| 2 | Future class: type hints, remove dead param, SecurityType enum | LOW |
| 3 | Option class: full rename (K→strike etc) across ~11 files, type hints, simplify check_active/moneyness, extract _sum_greeks_over_ttms | MEDIUM-HIGH |
| 4 | Portfolio class: 4 bug fixes, type hints, variable renames, extract _get_position_lists, simplify compute_net_greeks, decompose update_sec_by_month | MEDIUM-HIGH |
| 5 | Test updates + final verification | LOW |

Run full test suite after each commit.

---

## Key Reference: Option Constructor (Before → After)

**Before:**
```python
Option(strike, tau, char, vol, underlying, payoff, shorted, month,
       direc=None, barrier=None, lots=1000, bullet=True,
       ki=None, ko=None, rebate=0, ordering=1e5, settlement='futures',
       bvol=None, bvol2=None, dailies=None)
```

**After:**
```python
Option(strike, tau, option_type, vol, underlying, payoff, shorted, month,
       direction=None, barrier=None, lots=1000, bullet=True,
       knock_in=None, knock_out=None, rebate=0, ordering=1e5, settlement='futures',
       barrier_vol=None, barrier_vol2=None, dailies=None)
```

Note: `strike` stays the same (constructor param was always `strike`, just stored as `self.K`).

---

## Key Reference: calc.py Function Signatures to Rename

These functions use `ki`, `ko`, `bvol`, `bvol2` as keyword params:

- `_compute_greeks(char, K, tau, vol, s, r, product, payoff, lots, ki=, ko=, barrier=, direction=, order=, bvol=, bvol2=, dbarrier=)`
- `_compute_value(char, tau, vol, K, s, r, payoff, ki=, ko=, barrier=, d=, product=, bvol=, bvol2=, dbarrier=)`

Rename params in signatures AND all call sites.
