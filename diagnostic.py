import numpy as np
import pandas as pd
# import time
from sqlalchemy import create_engine
from scripts.portfolio import Portfolio
from scripts.util import create_underlying, create_vanilla_option, create_straddle, assign_hedge_objects
import os
from scripts.fetch_data import grab_data, pull_intraday_data
from scripts.prep_data import insert_settlements
from scripts.simulation import run_simulation
from collections import OrderedDict
import datetime as dt


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


# Purpose: Small-scale runs using the trailing stop to check if it works
# properly.

start = '2017-11-20'
end = '2017-12-01'
volid = 'LH  Z7.Z7'
pdt = volid.split()[0]
ftmth = volid.split('.')[1]

cleaned = 'datasets/trailingstop_prices_cleaned.csv'

settle_vols, settle_price, _ = grab_data([pdt], start, end)

if os.path.exists(cleaned):
    fpdf = pd.read_csv(cleaned)
    fpdf.value_date = pd.to_datetime(fpdf.value_date)
    fpdf.time = pd.to_datetime(fpdf.time).dt.time

else:
    it_data = pull_intraday_data(
        [pdt], start_date=start, end_date=end, contracts=[ftmth])
    fpdf = insert_settlements(it_data, settle_price)
    fpdf.to_csv(cleaned, index=False)


# create the portfolio and hedging parameters
# be = {'CC': {'U7': 1, 'Z7': 1},
#       'QC': {'U7': 1, 'Z7': 1}}
# settle_vols.time = dt.time.max
settle_price.time = pd.to_datetime(settle_price.time).dt.time
# settle_vols.time = pd.to_datetime(settle_vols.time).dt.time
# fpdf.time = pd.to_datetime(fpdf.time).dt.time


vals = {'LH  Z7': 1.4}

intraday_params = {'tstop': {'trigger': {'LH  Z7': (1.5, 'price')},
                             'value': {'LH  Z7': (0.2, 'price')}}}

gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                    ['intraday', 'static', vals, 1,
                                     intraday_params]]})


pf = Portfolio(hedge_params=gen_hedges, name='LHZ7 Tstop')

# create the options.

ops = create_straddle(volid, settle_vols, settle_price,
                      settle_vols.value_date.min(), False, 'atm',
                      greek='theta', greekval=10000)
pf.add_security(ops, 'OTC')

# add the future to make it delta neutral.
delta = pf.get_net_greeks()['LH']['Z7'][0]
shorted = True if delta > 0 else False
lots_req = abs(round(delta))

ft, _ = create_underlying(pdt, ftmth, settle_price,
                          date=settle_price.value_date.min(),
                          shorted=shorted, lots=lots_req)

pf.add_security([ft], 'hedge')

print('pf: ', pf)

# pf = assign_hedge_objects(pf, book=False)

# h1 = pf.get_hedgeparser()

# h2 = pf.get_hedgeparser(dup=True)

# print('h1: ', h1)
# print('-'*30)
# print('h1.mod_obj: ', h1.get_mod_obj())
# print('-'*30)
# print('h2.mod_obj: ', h2.get_mod_obj())

# print('#'*30)
# prices = h1.relevant_price_move('LH  Z7', 58)
# print('-'*30)
# print('h1.mod_obj: ', h1.get_mod_obj())
# print('-'*30)
# print('h2.mod_obj: ', h2.get_mod_obj())
# print('#'*30)

results = run_simulation(settle_vols, fpdf, pf)
