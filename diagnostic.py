
from scripts.prep_data import generate_hedges, sanity_check

import numpy as np
import pandas as pd
from scripts.util import create_straddle, create_underlying
from scripts.portfolio import Portfolio
from scripts.fetch_data import grab_data
from scripts.hedge import Hedge
from timeit import default_timer as timer
import pprint

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


######### variables ################
start_date = '2017-05-01'
end_date = '2017-07-21'
rd1 = pd.to_datetime('2017-06-30')
rd2 = pd.to_datetime('2017-06-30')
rollover_dates = {'QC  U7.U7': [rd1], 'CC  U7.U7': [rd2]}
pdts = ['QC', 'CC']
volids = ['QC  U7.U7', 'QC  Z7.Z7', 'CC  U7.U7', 'CC  Z7.Z7']
####################################


vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          volids=volids, write=True)

sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), pd.to_datetime(start_date), pd.to_datetime(end_date))


cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
    start_date), False, 'atm', greek='vega', greekval=20000)

qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', greek='vega', greekval=20000)

# print('CC Call: ', str(cc1))
# print('CC Put: ', str(cc2))
# print('QC Call: ', str(qc1))
# print('QC Put: ', str(qc2))

pf = Portfolio()
pf.add_security([qc1, qc2, cc1, cc2], 'OTC')

# print('portfolio: ', pf)

hedges = generate_hedges('hedging.csv')

print('hedges: ', hedges)

test_pdf = pdf[pdf.value_date == pd.to_datetime(start_date)]
test_vdf = vdf[vdf.value_date == pd.to_datetime(start_date)]

b1 = [0, 20, 30, 50, 80, 90]
b2 = [0, 50, 70, 100, 150, 200]
b3 = [0, 70, 120, 170]


t = timer()
# pf = generate_portfolio()
hedge = Hedge(pf, hedges, test_vdf, test_pdf,
              desc='uid', type='straddle', brokerage=1)

for key in hedges:
    if key != 'delta':
        hedge._calibrate(key)


print(pprint.pformat(hedge.params))
print(pprint.pformat(hedge.greek_repr))
print(pprint.pformat(hedge.mappings))

print(hedge.satisfied(pf))

print('~~~~ HEDGING VEGA ~~~~~~')
x = hedge.apply(pf, 'vega')
print('~~~~ HEDGING GAMMA~~~~~~')
# y = hedge.apply(pf, 'gamma')

hedge.refresh()

print(pprint.pformat(hedge.greek_repr))
print(hedge.satisfied(pf))

print('elapsed: ', timer() - t)
