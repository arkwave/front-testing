
from scripts.prep_data import generate_hedges, sanity_check

import numpy as np
import pandas as pd
from scripts.util import create_straddle, combine_portfolios, create_skew
from scripts.portfolio import Portfolio
from scripts.fetch_data import grab_data
# from scripts.hedge import Hedge
# from timeit import default_timer as timer
# import pprint
from simulation import run_simulation
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
end_date = '2017-12-31'
pdts = ['QC', 'CC']
volids = ['QC  U7.U7', 'QC  Z7.Z7', 'CC  U7.U7', 'CC  Z7.Z7']
####################################

from scripts.util import volids_from_ci
date_range = pd.bdate_range(start_date, end_date)

x = volids_from_ci(date_range, 'CC', 2)

print(pprint.pformat(x))


# vdf, pdf, edf = grab_data(pdts, start_date, end_date,
#                           volids=volids, write=True)

# sanity_check(vdf.value_date.unique(),
#              pdf.value_date.unique(), pd.to_datetime(start_date),
#              pd.to_datetime(end_date))

# callop, putop = create_skew('CC  U7.U7', vdf, pdf,
# pd.to_datetime(start_date), False, 31, greek='vega', greekval=50000,
# composites=True)


# cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
# start_date), False, 'atm', greek='vega', greekval=100000,
# composites=True)

# hedges, roll_portfolio, pf_ttm_tol, pf_roll_product, \
#     roll_hedges, h_ttm_tol, h_roll_product = generate_hedges('hedging.csv')


# # define hedges for the 25 delta call
# pf_calls = Portfolio(hedges, 1)
# pf_calls.add_security([callop], 'OTC')

# pf_puts = Portfolio(hedges, 3)
# pf_puts.add_security([putop], 'OTC')

# pf_atms = Portfolio(hedges, 2)
# pf_atms.add_security([cc1, cc2], 'OTC')

# # pf = Portfolio(hedges, 1)
# # pf.add_security([callop, putop, cc1, cc2], 'OTC')

# pf = combine_portfolios([pf_calls, pf_puts, pf_atms],
#                         hedges=hedges, name='all')

# pf.refresh()


# print('portfolio: ', pf)


# print('hedges: ', hedges)

# print('roll_portfolio: ', roll_portfolio)
# print('roll_hedges: ', roll_hedges)


# # from wip import hedge_delta_roll
# from simulation import rebalance, roll_over
# # from wip import roll_over

# date = pd.Timestamp('2017-05-02')

# test_pdf = pdf[pdf.value_date == date]
# test_vdf = vdf[vdf.value_date == date]
# counters = [1, 1, 1, 1]


# pf, counters, cost, roll_hedged = rebalance(test_vdf, test_pdf, pf, counters)
# pf.refresh()
# print('>>>>>>>>>>>> PF BEFORE ROLLOVER: ', pf.hedges)


# pf, total_cost = roll_over(pf, test_vdf, test_pdf,
# date, target_product='CC', ttm_tol=100, flag='OTC')

# pf, counters, cost, roll_hedged = rebalance(test_vdf, test_pdf, pf, counters)
# pf, cost = hedge_delta_roll(pf, test_pdf)


# log = run_simulation(vdf, pdf, edf, pf, hedges,
#                      roll_portfolio=roll_portfolio, pf_ttm_tol=pf_ttm_tol,
#                      pf_roll_product=pf_roll_product,
#                      roll_hedges=roll_hedges, h_ttm_tol=h_ttm_tol,
#                      h_roll_product=h_roll_product)


# t = timer()
# # pf = generate_portfolio()
# for dep in pf.get_families():
#     test = Hedge(dep, dep.hedge_params, test_vdf, test_pdf)
#     print('test_satisfied: ', test.satisfied())
#     for key in test.hedges:
#         if key != 'delta':
#             print('applying ' + key)
#             test.apply(key)
#     print('test_satisfied: ', test.satisfied())

# pf.refresh()

# ov_hedge = Hedge(pf, pf.hedge_params, test_vdf, test_pdf)
# print('overall satisfied: ', ov_hedge.satisfied())

# for key in ov_hedge.hedges:
#     ov_hedge.apply(key)

# print('overall satisfied: ', ov_hedge.satisfied())
