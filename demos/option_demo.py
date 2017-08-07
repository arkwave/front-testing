
from scripts.prep_data import read_data, generate_hedges, sanity_check, get_rollover_dates
import numpy as np
import pandas as pd
import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import create_portfolio, prep_datasets, pull_alt_data


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


epath = 'datasets/option_expiry.csv'
specpath = 'specs.csv'
sigpath = 'datasets/small_ct/signals.csv'
hedgepath = 'hedging.csv'

yr = 7
pnls = []

# vdf, pdf, df = pull_alt_data('KC')
log = None
target = 'N'

pdt = 'CT'
symlst = ['H', 'K', 'N', 'Z']
# ftmth = 'X2'
# opmth = 'X2'

alt = True  # if yr >= 5 else True

pricedump = 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'
voldump = 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'

start_date = pd.Timestamp('201' + str(yr) + '-05-22')
end_date = pd.Timestamp('201' + str(yr) + '-06-15')


opmth = target + str(yr)
ftmth = target + str(yr)

# vdf.to_csv('pf_test_vols.csv')
# pdf.to_csv('pf_test_prices.csv')

vdf = pd.read_csv('pf_test_vols.csv')
pdf = pd.read_csv('pf_test_prices.csv')
vdf.value_date = pd.to_datetime(vdf.value_date)
pdf.value_date = pd.to_datetime(pdf.value_date)


# pf1 - straddle.
print('1. Creating Straddle')
pf = create_portfolio(pdt, opmth, ftmth, 'straddle', vdf, pdf,
                      chars=['call', 'put'], shorted=False, atm=True,
                      greek='vega', greekval='250000')

print('############ Straddle ##############')
print(pf)
print('####################################')


print('2. Creating Long Strangle')
# pf2 - long strangle.
pf2 = create_portfolio(pdt, opmth, ftmth, 'strangle', vdf, pdf,
                       chars=['call', 'put'], shorted=False,
                       strike=[64, 58], greek='vega', greekval=25000)
print('############ L Strangle ##############')
print(pf2)
print('######################################')


print('3. Creating Short Strangle')
# pf3 - short strangle.
pf3 = create_portfolio(pdt, opmth, ftmth, 'strangle', vdf, pdf,
                       chars=['call', 'put'], shorted=True,
                       strike=[64, 58], greek='vega', greekval=-25000)

print('############ S Strangle ##############')
print(pf3)
print('######################################')


print('4. Creating Short Skew')
# pf4 - skew.
pf4 = create_portfolio(pdt, opmth, ftmth, 'skew', vdf, pdf,
                       delta=25, shorted=True, greek='vega', greekval=25000)
print('########### Short Skew ############')
print(pf4)
print('net vega position: ', pf4.net_vega_pos())
print('###################################')


print('5. Creating Bull Callspread')
# pf5 - bull callspread
pf5 = create_portfolio(pdt, opmth, ftmth, 'spread', vdf, pdf,
                       shorted=False, char='call', delta=[75, 25], greek='vega', greekval=25000)
print('########### Bull Callspread ############')
print(pf5)
print('########################################')

print('6. Creating Bear Callspread')
# pf6 - bear callspread
pf6 = create_portfolio(pdt, opmth, ftmth, 'spread', vdf, pdf,
                       shorted=True, char='call', delta=[75, 25])

print('########### Bear Callspread ############')
print(pf6)
print('########################################')

print('7. Creating Bull Putspread')
# pf7 - bull putspread
pf7 = create_portfolio(pdt, opmth, ftmth, 'spread', vdf, pdf,
                       shorted=False, char='put', delta=[25, 75])

print('########### Bull Putspread ############')
print(pf7)
print('########################################')


print('8. Creating Bear Putspread')
# pf8 - bear putspread
pf8 = create_portfolio(pdt, opmth, ftmth, 'spread', vdf, pdf,
                       shorted=True, char='put', delta=[25, 75])
print('########### Bear Putspread #############')
print(pf8)
print('########################################')

# pf9 - long call butterfly
print('9. Creating Long Call Butterfly')
pf9 = create_portfolio(pdt, opmth, ftmth, 'butterfly', vdf, pdf,
                       char='call', shorted=False, lots=[200, 200, 200, 200], delta=50, dist=2)
print('########### Long Call Butterfly #############')
print(pf9)
print('#############################################')


# pf10 - long put butterfly
print('10. Creating Long Put Butterfly')
pf10 = create_portfolio(pdt, opmth, ftmth, 'butterfly', vdf, pdf,
                        char='put', shorted=False, lots=[200, 200, 200, 200], strikes=[58, 60, 62])
print('########### Long Put Butterfly #############')
print(pf10)
print('#############################################')


# pf11 - short call butterfly
print('11. Creating short call butterfly')
pf11 = create_portfolio(pdt, opmth, ftmth, 'butterfly', vdf, pdf,
                        char='call', shorted=True, lots=[200, 200, 200, 200], strikes=[58, 60, 62])
print('########### Short Call Butterfly #############')
print(pf11)
print('##############################################')

# pf12 - short put butterfly
print('12. Creating short put butterfly')
pf12 = create_portfolio(pdt, opmth, ftmth, 'butterfly', vdf, pdf,
                        char='put', shorted=True, lots=[200, 200, 200, 200], strikes=[58, 60, 62])
print('########### Short Call Butterfly #############')
print(pf12)
print('##############################################')

# pf13 - fence
print('13. Creating Fence')
pf13 = create_portfolio(pdt, opmth, ftmth, 'skew', vdf, pdf,
                        delta=25, shorted=True, greek='vega', greekval=25000)
print('########### Fence ############')
print(pf13)
print('net vega pos: ', pf13.net_vega_pos())
print('##############################')

print('portfolio: ', pf)
print('deltas: ', [op.delta/op.lots for op in pf.OTC_options])
print('vegas: ', pf.net_vega_pos())
print('start_date: ', start_date)
print('end_date: ', end_date)

print('specifying hedging logic')
# specify hedging logic
hedges = generate_hedges(hedgepath)

print('getting rollover dates')


# get rollover dates according to opex
# rollover_dates = get_rollover_dates(priceDF)
# print('rollover dates: ', rollover_dates)

# print('running simulation')
# # run the simulation
# grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log = run_simulation(
#     vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage,
#     slippage=gv.slippage)

# pnls.append(netpnl)

# print(pnls)

# log.to_csv('results/cotton/log.csv', index=False)
