import numpy as np
import pandas as pd
import time
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

# intraday_data = pd.read_csv('datasets/s_intraday.csv')
# intraday_data.Date = pd.to_datetime(intraday_data.Date)

# test_data = intraday_data[
#     intraday_data.Commodity.isin(['S U7 Comdty', 'S F8 Comdty'])]

# # test_data = pr.handle_intraday_conventions(test_data)

# sd, ed = test_data.Date.min().strftime('%Y-%m-%d'), \
#     test_data.Date.max().strftime('%Y-%m-%d')


# vdf.value_date = pd.to_datetime(vdf.value_date)
# pdf.value_date = pd.to_datetime(pdf.value_date)

# date = vdf.value_date.min()

# op1, op2 = create_skew('S  U7.U7', vdf, pdf, date,
#                        False, 25, greek='vega', greekval=100000)

# print('op1: ', op1)
# print('op2: ', op2)


# print('handling intraday conventions...')
# t1 = time.clock()
# test_data = pr.handle_intraday_conventions(test_data)
# print('intraday conventions handled. elapsed: ', time.clock() - t1)


# # small_df = test_data[test_data.value_date == test_data.value_date.min()]

# print('running timestep recon..')
# t = time.clock()
# tst = pr.timestep_recon(test_data)
# print('finished timestep recon. elapsed: ', time.clock() - t)


df = pd.read_csv('alt_merged_data.csv')
df.value_date = pd.to_datetime(df.value_date)
df.time = df.time.astype(pd.Timestamp)
date = df.value_date.min()
max_date = df.value_date.max()


vdf, pdf, edf = grab_data(['S'], date.strftime(
    '%Y-%m-%d'), max_date.strftime('%Y-%m-%d'), test=True)

sim_start, sim_end = pd.to_datetime('2017-02-23'), pd.to_datetime('2017-02-24')

sim_start = pd.Timestamp('2017-02-23')
sim_end = pd.Timestamp('2017-02-24')

prices = df[df.value_date.isin([sim_start, sim_end])]
vols = vdf[vdf.value_date.isin([sim_start, sim_end])]

print('price: ', prices.columns)
print('vols: ', vols.columns)

# create the portfolio
op = create_vanilla_option(vols, prices, 'S  U7.U7', 'call',
                           False, date=sim_start, strike='atm')

hedges = {'delta': [['static', 'zero', 1],
                    ['intraday', 'breakeven', {'S  U7': 0.75}]]}

pf = Portfolio(hedges, name='it_test')
pf.add_security([op], 'OTC')
pf = assign_hedge_objects(pf)

print('pf: ', pf)
print('pf.hedger: ', pf.get_hedger())

prices = prices[prices.underlying_id == 'S  U7']

# prices = prices[prices.value_date > sim_start]
# vols = vols[vols.value_date > sim_start]

log = run_simulation(vols, prices, pf, plot_results=False)
