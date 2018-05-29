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

from scripts.fetch_data import grab_data
from scripts.util import create_barrier_option
# from scripts.prep_data import filter_outliers

pdts = ['KC']
contract = ['N8']
start = '2018-01-01'
end = '2018-01-10'
# # oldpath = 'old_data.csv'


vdf, pdf, edf = grab_data(pdts, start, end)

# create the legs for a KCU8 118.20/128.00 Long Knight DDU b 121.85 
vol_id = 'KC  U8.U8'
acc = 118.20
ref = 121.85
char = 'call'

direction = 'up'
barriertype = 'amer'
shorted = False
bullet = False
ki = None
ko = 128


vol = 0.19032
bvol = 0.22648
bvol2 = None
date = pd.to_datetime('2018-05-21')
expiry = pd.to_datetime('2018-08-18')

op = create_barrier_option(None, None, vol_id, char, acc, shorted, date, 
                           barriertype, direction, ki, ko, bullet, 
                           lots=1, ref=ref, vol=vol, bvol=bvol, 
                           bvol2=bvol2, expiry=expiry)


value = sum([o.compute_price() for o in op]) / len([o for o in op if o.tau > 0])
print('number of ops: ', len(op))
print('-------------------- t ----------------------')
print('mkt value: ', value)
print('delta: ', sum([o.get_greek('delta') for o in op]))
print('gamma: ', sum([o.get_greek('gamma') for o in op]))
print('theta: ', sum([o.get_greek('theta') for o in op]))
print('vega: ', sum([o.get_greek('vega') for o in op]))
print('--------------------------------------------------')

for o in op:
    o.update_tau(1/365)
    o.update()

value = sum([o.compute_price() for o in op]) / len([o for o in op if o.tau > 0])
print('-------------- t + 1 --------------------')
print('mkt value: ', value)
value = sum([o.compute_price() for o in op]) / len(op)
print('delta: ', sum([o.get_greek('delta') for o in op]))
print('gamma: ', sum([o.get_greek('gamma') for o in op]))
print('theta: ', sum([o.get_greek('theta') for o in op]))
print('vega: ', sum([o.get_greek('vega') for o in op]))
print('--------------------------------------------------')
