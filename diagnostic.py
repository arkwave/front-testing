import numpy as np
import pandas as pd
import time
from sqlalchemy import create_engine
from scripts.portfolio import Portfolio
import scripts.prep_data as pr
from scripts.util import create_skew, create_underlying, create_vanilla_option, assign_hedge_objects
from scripts.fetch_data import grab_data
from scripts.simulation import run_simulation


multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'LCC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 10, 50],
    'C':   [0.393678571428571, 127.007166832986, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}

seed = 7
np.random.seed(seed)
pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10


# details contract months for each commodity. used in the continuation
# assignment.
contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'LCC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V' 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


# from scripts.fetch_data import pull_ohlc_data
# pdts = ['SB']
# start = '2016-10-10'
# end = '2017-01-15'

# df = pull_ohlc_data(pdts, start, end)
# vdf, pdf, edf = grab_data(pdts, start, end)

# # check if they are the same
# df.sort_values(by='value_date')
# vdf.sort_values(by='value_date')

# assert np.array_equal(vdf.value_date.unique(), df.value_date.unique())

# callop = create_vanilla_option(
#     vdf, df, 'SB  H7.H7', 'call', False, delta=25)

# pf = Portfolio(None)
# pf.add_security([callop], 'OTC')
# print('pf: ', pf)
# print('breakevens: ', pf.breakeven())

# tst = df[(df.value_date == df.value_date.min()) &
#          (df.underlying_id == 'SB  H7')]

# print('tst: ', tst)

# tst, mod = pr.reorder_ohlc_data(tst, pf)

start_date = '2017-01-01'
end_date = '2017-01-31'
pdts = ['CT']

import datetime as dt

vdf, pdf, edf = grab_data(pdts, start_date, end_date)

vdf.value_date = pd.to_datetime(vdf.value_date)
pdf.value_date = pd.to_datetime(pdf.value_date)
pdf.time = pd.to_datetime(pdf.time).dt.time


pf = Portfolio(None, name='settle_test')

callop = create_vanilla_option(vdf, pdf, 'CT  H7.H7',
                               'call', False, strike='atm')

pf.add_security([callop], 'OTC')

outputs = run_simulation(vdf, pdf, pf)
