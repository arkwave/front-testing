
brokerage = 1
import numpy as np
from math import log, sqrt, exp
import pandas as pd
from ast import literal_eval
from collections import OrderedDict
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

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

filepath = 'hedging.csv'
# with open(filepath) as f:
#     try:
#         d_cond = f.readline().strip('\n')
#         g_cond = f.readline().strip('\n')
#         v_cond = f.readline().strip()
#         print(d_cond)
#         print(g_cond)
#         print(v_cond)
#     except FileNotFoundError:
#         print(filepath)


seed = 7
np.random.seed(seed)

pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.

# composite label that has product, opmth, cont.
# vdf['label'] = vdf['vol_id'] + ' ' + \
#     vdf['order'].astype(str) + ' ' + vdf.call_put_id


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


# vdf = pd.read_csv('datasets/ct_data/ct_vols.csv')
# pdf = pd.read_csv('datasets/ct_data/ct_prices.csv')
# signals = pd.read_csv('datasets/signals.csv')
# signals.value_date = pd.to_datetime(signals.value_date)

# pdf.value_date = pd.to_datetime(pdf.value_date)
# vdf.value_date = pd.to_datetime(vdf.value_date)
# # getting rid of sunday/friday reporting errors.
# vdf['sunday'] = vdf.value_date.dt.weekday == 6
# pdf['sunday'] = pdf.value_date.dt.weekday == 6

# vdf.loc[vdf.sunday == True, 'value_date'] -= pd.Timedelta('2 days')
# pdf.loc[pdf.sunday == True, 'value_date'] -= pd.Timedelta('2 days')

# cal = calendar()
# holidays = cal.holidays(start=pd.Timestamp('2017-01-02'), end='2017-03-31')

# # filtering relevant dates
# vdf = vdf[(vdf.value_date > pd.Timestamp('2017-01-02')) &
#           (vdf.value_date < pd.Timestamp('2017-04-02'))]
# pdf = pdf[(pdf.value_date > pd.Timestamp('2017-01-02')) &
#           (pdf.value_date < pd.Timestamp('2017-04-02'))]


# d1 = [x for x in vdf.value_date.unique() if x not in signals.value_date.unique()]
# d1 = pd.to_datetime(d1)
# d2 = [x for x in pdf.value_date.unique() if x not in signals.value_date.unique()]
# d2 = pd.to_datetime(d2)

# d3 = [x for x in signals.value_date.unique() if x not in vdf.value_date.unique()]
# d3 = pd.to_datetime(d3)
# d4 = [x for x in signals.value_date.unique() if x not in pdf.value_date.unique()]
# d4 = pd.to_datetime(d4)

# d3_days = ['Friday' for x in d3 if x.weekday() == 4]

# ht1 = [(x in holidays) or (x.weekday() in [5, 6]) for x in d1]
# ht2 = [(x in holidays) or (x.weekday() in [5, 6]) for x in d2]


# # mask = signals.value_date.isin(d3)
# vmask = vdf.value_date.isin(d1)
# pmask = pdf.value_date.isin(d2)

# vdf = vdf[~vmask]
# pdf = pdf[~pmask]
