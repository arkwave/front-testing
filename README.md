# PortfolioSimulator

General-purpose module that simulates the behavior of an options portfolio on commodity futures, subject to hedging constraints and trading strategies.

## Overview

This module allows arbitrarily complex portfolios to be run through historical price/volatility series of variable length. The position is run through the historic data, and the outputs (cumulative PnL, gamma PnL, vega PnL, daily PnL and future prices) are plotted using Plotly.

Key features:
- Vanilla and exotic option support (barriers, digitals, spreads)
- Automatic Greek rehedging (delta, gamma, vega, theta) with configurable constraints
- Contract rolling for near-expiry instruments
- Detailed PnL attribution (gross/net, gamma PnL, vega PnL)
- Historical backtesting with intraday and end-of-day granularity

## Requirements

- Python >= 3.6
- numpy, pandas, scipy, plotly

```bash
pip install numpy pandas scipy plotly
```

## Quick Start

```python
from collections import OrderedDict
from scripts.fetch_data import grab_data
from scripts.util import create_vanilla_option, create_underlying
from scripts.simulation import run_simulation
from scripts.portfolio import Portfolio

# 1. Load market data
vdf, pdf, edf = grab_data(['W'], '2015-05-25', '2015-08-15')

# 2. Create an ATM call option on Wheat Z15, sized to 100k vega
option = create_vanilla_option(
    vdf, pdf, 'W  Z15.Z15', 'call', False,
    strike='atm', vol=0.26, greek='vega', greekval='100000'
)

# 3. Define hedging rules (keep delta = 0, rebalance daily)
hedges = OrderedDict({'delta': [['static', 0, 1]]})

# 4. Build the portfolio
pf = Portfolio(hedges, name='example')
pf.add_security([option], 'OTC')

# 5. Delta-hedge with futures
greeks = pf.get_net_greeks()
for pdt in greeks:
    for mth in greeks[pdt]:
        delta = round(greeks[pdt][mth][0])
        ft, _ = create_underlying(pdt, mth, pdf, shorted=(delta > 0), lots=abs(delta))
        pf.add_security([ft], 'hedge')

# 6. Run the backtest
results = run_simulation(vdf, pdf, pf, flat_vols=True)
```

See `eod_simulation_demo.py` for a full working example, and `demos/` for more.

## Project Structure

```
scripts/            Core library
  simulation.py       Main simulation loop
  portfolio.py        Portfolio class (positions, net Greeks)
  classes.py          Option and Future data classes
  calc.py             Black-Scholes pricing and Greeks
  hedge.py            Automatic rehedging engine
  util.py             Factory functions for creating positions
  fetch_data.py       Market data loading
  prep_data.py        Data cleaning and preparation
  signals.py          Trading signal application
tests/              Unit tests (unittest)
datasets/           Sample market data (Cotton)
demos/              Example scripts
```

## Running Tests

```bash
# All tests
python -m unittest discover -s tests -p "test_*.py" -v

# Single module
python -m unittest tests.test_calc
python -m unittest tests.test_simulation

# Single test
python -m unittest tests.test_options.TestOptions.test_option_creation
```

## AI Agent Context

See [CLAUDE.md](CLAUDE.md) for detailed module documentation, execution flow diagrams, dependency graphs, and key concepts — optimized for AI-assisted development.
