# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-05 15:57:43
# @Last Modified by:   arkwave
# @Last Modified time: 2017-07-07 14:34:29

from scripts.prep_data import generate_hedges, sanity_check
import os
import numpy as np
import pandas as pd
import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import create_portfolio
from scripts.fetch_data import pull_alt_data, prep_datasets
from scripts.portfolio import Portfolio


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


#################### variables #########################
pdt = 'CT'
yr = 7
signals = True
target = 'N'
symlst = ['H', 'K', 'N', 'U', 'Z']
start_date = pd.Timestamp('2017-01-03')
end_date = pd.Timestamp('2017-03-31')
#######################################################


#################### default paths ##########################
epath = 'datasets/option_expiry.csv'
specpath = 'specs.csv'
signal_path = '../skew_lib/results/test_signal.csv'
hedgepath = 'hedging.csv'
pricedump = 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'
voldump = 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'
#############################################################


# vdf, pdf, df = pull_alt_data()


print('pulling data')
edf = pd.read_csv(epath)

# check to see if the data exists
if not os.path.exists(pricedump) or (not os.path.exists(voldump)):
    vdf, pdf, raw_daf = pull_alt_data(pdt)

# cleaning data, pulling relevant stuff from dumps.
uids = [pdt + '  ' + u + str(yr) for u in symlst]
print('uids: ', uids)
volids = [pdt + '  ' + u + str(yr) + '.' + u + str(yr) for u in symlst]
print('volids: ', volids)
pdf = pd.read_csv(pricedump)
pdf.value_date = pd.to_datetime(pdf.value_date)

# cleaning prices
pmask = pdf.underlying_id.isin(uids)
pdf = pdf[pmask]

# cleaning vols
vdf = pd.read_csv(voldump)
vmask = vdf.vol_id.isin(volids)
vdf = vdf[vmask]
vdf.value_date = pd.to_datetime(vdf.value_date)


# handling signals
signals = pd.read_csv(signal_path) if (
    os.path.exists(signal_path) and signals) else None


# prep_datasets
vdf, pdf, edf, priceDF, start_date = prep_datasets(
    vdf, pdf, edf, start_date, end_date, pdt, signals=signals, test=False, write=True)
print('finished pulling data')


# sanity check date ranges
print('sanity checking data')
sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), start_date, end_date)


# initalizing portfolio
print('creating portfolio')
pf = Portfolio()


print('portfolio: ', pf)
print('deltas: ', [op.delta/op.lots for op in pf.OTC_options])
print('vegas: ', pf.net_vega_pos())
print('start_date: ', start_date)
print('end_date: ', end_date)


print('specifying hedging logic')

# specify hedging logic
hedges = generate_hedges(hedgepath)


# run the simulation
print('running simulation')
grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values,\
    net_daily_values, net_cumul_values, log = run_simulation(vdf, pdf, edf,
                                                             pf, hedges,
                                                             brokerage=gv.brokerage,
                                                             slippage=gv.slippage,
                                                             signals=signals)

# bound = '_20_30'
bound = '_10_40'
log.to_csv('results/' + pdt.lower() + '/201' +
           str(yr) + bound + '_log_test.csv', index=False)


###################### code dump ##########################
# create 130,000 vega atm straddles

# hedge_specs = {'pdt': 'S',
#                'opmth': 'N' + str(yr),
#                'ftmth': 'N' + str(yr),
#                'type': 'straddle',
#                'strike': 'atm',
#                'shorted': True,
#                'greek': 'gamma',
#                'greekval': 'portfolio'}
