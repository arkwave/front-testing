# CLAUDE.md — AI Context for PortfolioSimulator

## What This Project Is

An options portfolio backtesting system for commodity futures. It simulates portfolios of vanilla and exotic options through historical price/volatility data, automatically rehedges Greeks (delta, gamma, vega, theta), and produces PnL attribution reports.

## Project Structure

```
front-testing/
├── scripts/                  # Core library modules
│   ├── simulation.py         # Main simulation loop (run_simulation)
│   ├── portfolio.py          # Portfolio class — holds positions, computes net Greeks
│   ├── classes.py            # Option and Future classes (pricing, Greeks, barriers)
│   ├── calc.py               # Black-Scholes pricing, Greeks formulas, barrier pricing
│   ├── hedge.py              # Hedge class — auto-rehedging engine
│   ├── hedge_mods.py         # HedgeModifier ABC + TrailingStop implementation
│   ├── util.py               # Factory functions (create_vanilla_option, create_barrier_option, etc.)
│   ├── fetch_data.py         # Data loading (grab_data → vdf, pdf, edf)
│   ├── prep_data.py          # Data cleaning, OHLC granularization, intraday timing fixes
│   ├── signals.py            # Trading signal application (apply_signal)
│   ├── generate_signals.py   # Signal generation from skew data
│   ├── global_vars.py        # Legacy config: paths, date ranges, multipliers
│   └── work_in_progress.py   # Experimental/WIP code
├── tests/                    # Unit tests (unittest)
│   ├── test_calc.py          # Pricing & Greeks tests
│   ├── test_options.py       # Option class tests
│   ├── test_futures.py       # Future class tests
│   ├── test_portfolio.py     # Portfolio tests
│   ├── test_hedge.py         # Hedging engine tests
│   ├── test_hedgeparser.py   # Hedge parameter parsing tests
│   ├── test_simulation.py    # End-to-end simulation tests
│   ├── test_prep_data.py     # Data preparation tests
│   ├── test_utils.py         # Utility function tests
│   ├── test_dailies.py       # Daily PnL tests
│   ├── test_trailingstop.py  # Trailing stop tests
│   └── testing_vanilla.csv   # Test fixture data
├── datasets/
│   ├── small_ct/             # Sample dataset (Cotton) used by tests & demos
│   │   ├── final_vols.csv    # Volatility surface by strike/date
│   │   ├── final_price.csv   # Settlement prices
│   │   ├── final_expdata.csv # Expiry dates
│   │   └── cleaned_*.csv     # Intermediate cleaned data
│   └── exchange_timings.csv  # Market hours per exchange
├── demos/                    # Example scripts
│   ├── option_demo.py        # Option strategy construction examples
│   ├── hedging_tmp.py        # Barrier option + hedging demo
│   ├── plotlydemo.py         # Plotly visualization demo
│   └── tkinter_demo.py       # Tkinter GUI demo
├── eod_simulation_demo.py    # PRIMARY ENTRY POINT — full end-to-end backtest demo
├── diagnostic.py             # Diagnostic/analysis script
├── specs.csv                 # Portfolio specification file
├── input_demo.xlsx           # Example input spreadsheet
└── misc/                     # Miscellaneous utilities
```

## Execution Flow

The end-to-end backtest follows this pipeline:

```
1. Load Data          fetch_data.grab_data(products, start, end)
                      → returns (vdf, pdf, edf)
                        vdf = volatility surface (strike, date, tau, implied_vol)
                        pdf = settlement prices (date, underlying_id, price)
                        edf = option expiry dates

2. Create Positions   util.create_vanilla_option(vdf, pdf, vol_id, ...)
                      util.create_barrier_option(vdf, pdf, vol_id, ...)
                      util.create_underlying(product, month, pdf, ...)
                      → returns Option / Future objects

3. Build Portfolio    portfolio.Portfolio(hedge_params, name)
                      pf.add_security([option], 'OTC')     # client positions
                      pf.add_security([future], 'hedge')   # hedging instruments

4. Attach Hedger      util.assign_hedge_objects(pf, vdf, pdf, ...)
                      → creates Hedge object, links it to the portfolio

5. Run Simulation     simulation.run_simulation(vdf, pdf, pf, ...)
                      → daily loop:
                         a. feed_data()      — update prices/vols from market data
                         b. compute PnL      — track daily Greeks PnL
                         c. handle_barriers() — check barrier breaches
                         d. apply_signal()   — execute trading signals
                         e. roll_over()      — roll expiring contracts
                         f. rebalance()      — rehedge Greeks at EOD
                         g. timestep()       — decay option time (1 day)
                         h. write_log()      — record daily metrics
                      → returns DataFrame of daily results

6. Results            DataFrame columns include:
                        value_date, px_settle, eod_pnl_net, cu_pnl_net,
                        cu_gamma_pnl, cu_vega_pnl, vol_id
                      Optional Plotly visualization
```

## Key Concepts

**Hedge Parameter Format** — passed to Portfolio constructor:
```python
from collections import OrderedDict
hedges = OrderedDict({
    'delta': [['static', 0, 1]],              # keep delta = 0, rehedge every 1 day
    'gamma': [['bound', (3800, 4200), 1]],    # gamma within bounds
    'vega':  [['bound', (50000, 100000), 1]],
    'theta': [['bound', (-500, 500), 1]],
})
# Format: [method, target, frequency]
# Methods: 'static' (exact target), 'bound' (min/max tuple), 'roll'
```

**Simulation Modes** — the `mode` parameter to `run_simulation()`:
- `HSPS` (default): Hedge @ settlement vols, PnL @ settlement vols
- `HBPS`: Hedge @ book vols, PnL @ settlement vols
- `HBPB`: Hedge @ book vols, PnL @ book vols

**Product Multipliers** — defined in `simulation.py` and `test_calc.py`:
```python
multipliers[product] = [dollar_mult, lot_mult, futures_tick, options_tick, pnl_mult]
# Supported products: W, S, C, KC, CT, SB, BO, LH, LC, SM, CC, LRC, KW, MW, COM, OBM, LSU
```

**Vol ID Format** — identifies a specific option contract:
```
"CT  Z17.Z17"
 ^^  ^^^.^^^
 |   |   └── underlying future month+year
 |   └────── option expiry month+year
 └────────── product code (space-padded to 3 chars)
```

## Running Tests

Tests use Python's built-in `unittest` framework:

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run a single test module
python -m unittest tests.test_calc
python -m unittest tests.test_simulation

# Run a specific test case
python -m unittest tests.test_options.TestOptions.test_option_creation
```

**Dependencies required**: numpy, pandas, scipy, plotly

```bash
pip install numpy pandas scipy plotly
```

## Module Dependency Graph

```
eod_simulation_demo.py
├── scripts/fetch_data     (grab_data)
├── scripts/util           (create_vanilla_option, create_underlying, assign_hedge_objects)
├── scripts/portfolio      (Portfolio)
└── scripts/simulation     (run_simulation)
    ├── scripts/prep_data  (granularize, clean_data)
    ├── scripts/calc       (pricing, Greeks)
    ├── scripts/classes    (Option, Future)
    ├── scripts/hedge      (Hedge — rebalancing)
    │   └── scripts/util   (create options/futures for hedging)
    └── scripts/signals    (apply_signal)
```

## Common Patterns

- **Option objects** always reference a `Future` object as their underlying (`option.underlying`)
- **Greeks** are stored as `[delta, gamma, theta, vega]` arrays in `portfolio.net_greeks[product][month]`
- **`shorted=True/False`** on creation determines long/short position (affects PnL sign)
- **`flag='OTC'` vs `'hedge'`** when adding securities to Portfolio separates client positions from hedging instruments
- **Data feeds** use `vdf`/`pdf` DataFrames passed throughout — these are the single source of truth for market data

## Files You Can Ignore

- `misc/` — one-off data processing scripts
- `scripts/global_vars.py` — legacy hardcoded paths (Windows-specific, not used in current flow)
- `scripts/work_in_progress.py` — experimental code
- `wip.py` — work in progress at top level
- `demos/tkinter_demo.py`, `demos/plotlydemo.py` — UI demos unrelated to core logic
