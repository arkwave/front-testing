import numpy as np
import pandas as pd

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

# Purpose: sanity check the new pulling functions and ensure that the two
# results are identical.
import os
from scripts.fetch_data import pull_intraday_data_new, pull_intraday_data_old

pdt = 'CT'
contract = ['H8']
start = '2017-11-20'
end = '2017-12-12'
oldpath = 'old_data.csv'


newdf = pull_intraday_data_new([pdt], start_date=start,
                               end_date=end, contracts=contract)


if os.path.exists(oldpath):
    olddf = pd.read_csv(oldpath)
    olddf.value_date = pd.to_datetime(olddf.value_date)
    olddf.time = pd.to_datetime(olddf.time).dt.time
else:
    olddf = pull_intraday_data_old(
        [pdt], start_date=start, end_date=end, contracts=contract)
    olddf.to_csv(oldpath)

# define a bunch of sanity checks.

not_matching = []

for dt, grp in olddf.groupby('value_date'):
    newdf_grp = newdf[newdf.value_date == dt]
    print('date, oldlen, newlen: ', dt, len(grp), len(newdf_grp))


ndf = newdf[newdf.value_date == newdf.value_date.min()]
odf = olddf[olddf.value_date == olddf.value_date.min()]

# def sanity_check_data(old, new):
#     # first: check length.
#     try:
#         assert len(old) == len(new)
#     except AssertionError as e:
#         raise AssertionError(len(old), len(new))

#     # check dates.
#     olddates = old.value_date.unique()
#     newdates = new.value_date.unique()
#     assert len(newdates) == len(olddates)
#     assert newdates == olddates
