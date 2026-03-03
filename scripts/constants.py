"""
Centralized constants for the portfolio simulation system.

This module is the single source of truth for multipliers, contract months,
month/symbol mappings, tick sizes, and other constants that were previously
duplicated across multiple files.
"""

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

# Dictionary mapping month number to symbol and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

# Contract months for each commodity
# NOTE: LC previously had a missing-comma bug ('V' 'Z' -> 'VZ'). Fixed here.
contract_mths = {
    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'QC':  ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V', 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'CA':  ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z'],
}

# Options tick sizes (used for hedging strike selection)
op_ticksize = {
    'QC':  1,
    'CC':  1,
    'SB':  0.01,
    'LSU': 0.05,
    'KC':  0.01,
    'DF':  1,
    'CT':  0.01,
    'C':   0.125,
    'S':   0.125,
    'SM':  0.05,
    'BO':  0.005,
    'W':   0.125,
    'MW':  0.125,
    'KW':  0.125,
}

RANDOM_SEED = 7
DECADE = 10
TIMESTEP = 1 / 365
BREAKEVEN_FACTOR = 2.8


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
