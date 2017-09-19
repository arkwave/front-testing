import numpy as np
import pandas as pd
from scripts.portfolio import Portfolio
from scripts.util import pnp_format, create_straddle, merge_dicts, merge_lists, combine_portfolios
from scripts.fetch_data import grab_data
from scripts.simulation import run_simulation
from collections import OrderedDict

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


filepath = 'C:/Users/Ananth/Desktop/sim_test/'
start_date = '2016-09-01'
end_date = '2017-06-15'
pdts = ['W']

# get the data.
vdf, pdf, edf = grab_data(pdts, start_date, end_date, writepath=filepath)


# handle types.
vdf.value_date = pd.to_datetime(vdf.value_date)
pdf.value_date = pd.to_datetime(pdf.value_date)
date = vdf.value_date.min()

# create hedges
f_1_hedges = OrderedDict(
    [('theta', [['bound', (3000, 5000), 1, 'straddle', 'strike', 'atm', 'uid']])])
f_2_hedges = OrderedDict(
    [('theta', [['bound', (7000, 9000), 1, 'straddle', 'strike', 'atm', 'uid']])])
gen_hedges = OrderedDict([('delta', [['static', 'zero', 1]])])


# create options
z_strad = create_straddle('W  Z6.Z6', vdf, pdf, date, True,
                          'atm', greek='theta', greekval=4000)

u_strad = create_straddle('W  U7.U7', vdf, pdf, date, False,
                          'atm', greek='theta', greekval=8000)


# create the portfolios
pf1 = Portfolio(f_1_hedges, name='roll_pf', roll=True,
                roll_product=None, ttm_tol=10)
pf1.add_security(z_strad, 'OTC')

pf2 = Portfolio(f_2_hedges, name='backmonth', roll=False)
pf2.add_security(u_strad, 'OTC')


# create full portfolio
pf = combine_portfolios([pf1, pf2], hedges=gen_hedges,
                        name='all', refresh=True)


# # run the simulation.
# log = run_simulation(vdf, pdf, edf, pf,
#                      flat_price=False, flat_vols=False,
#                      plot_results=False, drawdown_limit=2000000)
