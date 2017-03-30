"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 30/3/2017
Python version : 3.5
Description    : Script containing problematic code to be debugged.

"""

# # Imports
# from . import portfolio
# from . import classes
from scripts.prep_data import read_data
import pandas as pd
import calendar
import datetime as dt
import ast
import sys
import traceback
import numpy as np
from scipy.stats import norm
from math import log, sqrt
import time
import matplotlib.pyplot as plt
import seaborn as sns
'''
TODO:  2) read in multipliers from csv
'''
pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.
filepath = 'portfolio_specs.txt'

vdf, pdf, edf = read_data(filepath)

# composite label that has product, opmth, cont.
# vdf['label'] = vdf['vol_id'] + ' ' + \
#     vdf['cont'].astype(str) + ' ' + vdf.call_put_id


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


def compute_delta(x):
    s = x.settle_value
    K = x.strike
    tau = x.tau
    char = x.call_put_id
    vol = x.settle_vol
    r = 0
    try:
        d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
            (vol * sqrt(tau))
    except (ZeroDivisionError):
        d1 = -np.inf

    if char == 'C':
        # call option calc for delta and theta
        delta1 = norm.cdf(d1)
    if char == 'P':
        # put option calc for delta and theta
        delta1 = norm.cdf(d1) - 1

    return delta1


def vol_by_delta(voldata, pricedata):
    relevant_price = pricedata[
        ['underlying_id', 'value_date', 'settle_value', 'cont']]
    relevant_vol = voldata[['value_date', 'vol_id', 'strike',
                            'call_put_id', 'tau', 'settle_vol', 'underlying_id']]
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['value_date', 'underlying_id'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['settle_vol'] > 0)]
    merged['delta'] = merged.apply(compute_delta, axis=1)
    merged.to_csv('merged.csv')
    return merged


df = vol_by_delta(vdf, pdf)
d1c = df[(df.value_date == pd.Timestamp('2017-01-01')) & (df.call_put_id == 'C')
         ][['settle_vol', 'delta', 'strike']]
d1p = df[(df.value_date == pd.Timestamp('2017-01-01')) & (df.call_put_id == 'P')
         ][['settle_vol', 'delta', 'strike']]

# filtering
d1c = d1c[(d1c.delta > 0.05) & (d1c.delta < 0.95)]
d1p = d1p[(d1p.delta < - 0.05) & (d1p.delta > -0.95)]

plt.figure()
plt.scatter(d1p.delta, d1p.settle_vol, c='c', alpha=0.7)
plt.xlabel('delta')
plt.ylabel('settle vol')
plt.legend()
plt.show()

# conc = pd.concat(d1c, d1p)
# plt.figure()
# plt.scatter()

# label = 'C  N7.Z7 4 C'
# df = vdf[vdf.label == label]
# dates = sorted(df.value_date.unique())
# d1 = dates[0]
# d2 = dates[1]
# prev_atm_price = pdf[(pdf['value_date'] == d1)]['settle_value'].values[0]
# curr_atm_price = pdf[(pdf['value_date'] == d2)]['settle_value'].values[0]
# curr_vol_surface = df[(df['value_date'] == d2)][['strike','settle_vol']]
# prev_vol_surface = df[(df['value_date'] == d1)][['strike','settle_vol']]
# prev_atm_vol = prev_vol_surface.loc[(prev_vol_surface['strike'] == (round(prev_atm_price/10) * 10)), 'settle_vol']
# prev_atm_vol = prev_atm_vol.values[0]
# dvol = curr_vol_surface['settle_vol'] - prev_atm_vol
