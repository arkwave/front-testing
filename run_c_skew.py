# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-12 19:19:17
# @Last Modified by:   arkwave
# @Last Modified time: 2017-07-12 19:41:19


from scripts.prep_data import generate_hedges, sanity_check
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
# import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import create_portfolio
from scripts.fetch_data import pull_alt_data, prep_datasets, grab_data
from skew_lib.skew_funcs.util import compute_skew, compute_skew_percentile
from skew_ib.skew_funcs.strategies.simple_strat import band_simple_strat
# from scripts.portfolio import Portfolio


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
pdt = 'C'
yr = 7
signals = True
target = 'U'
symlst = ['H', 'K', 'N', 'U', 'Z']
start_date = pd.Timestamp('2017-04-01')
end_date = pd.Timestamp('2017-07-12')
#######################################################


#################### default paths ##########################
epath = 'datasets/option_expiry.csv'
specpath = 'specs.csv'
signal_path = '../skew_lib/results/simple_' + pdt.lower() + '_signal.csv'
hedgepath = 'hedging.csv'
pricedump = 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'
voldump = 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'
writepath = '../skew_lib/datasets/test_sets/'
#############################################################


######### Other variables #########
volids = ['C  Z7.Z7', 'C  U7.U7']
weights = [1, -1, 1]
deltas = [25, -25, 50]


####################################
vdf, pdf, edf = grab_data([pdt], start_date, end_date,
                          writepath=writepath, volids=volids)

write_path = 'datasets/skew/' + pdt.lower() + '_' +\
    start_date.strftime('%Y%m%d') + '_' +\
    end_date.strftime('%Y%m%d')

df = compute_skew(pdf, weights, deltas)
df.to_csv(write_path + '_25_skews.csv', index=False)
vdf = compute_skew_percentile(df, deltas, 'ttm')
vdf.to_csv('datasets/skew/' + pdt.lower() + '_' +
           start_date.strftime('%Y%m%d') + '_' +
           end_date.strftime('%Y%m%d') + '_25_skew_pct.csv', index=False)
