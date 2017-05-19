
from scripts.prep_data import read_data, generate_hedges, sanity_check, get_rollover_dates
import numpy as np
import pandas as pd
import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import create_portfolio, prep_datasets


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


pdt = 'S'
ftmth = 'X2'
opmth = 'X2'
alt = True

pricedump = 'datasets/small_s/cleaned_price_dump.csv'
voldump = 'datasets/small_s/cleaned_vol_dump.csv'

start_date = pd.Timestamp('2012-05-01')
end_date = pd.Timestamp('2012-07-30')

if alt:
    edf = pd.read_csv(epath)
    uid = pdt + '  ' + ftmth
    pdf = pd.read_csv(pricedump)
    pdf.value_date = pd.to_datetime(pdf.value_date)
    # cleaning prices
    pdf = pdf[pdf.underlying_id == uid]

    vdf = pd.read_csv(voldump)
    volid = pdt + '  ' + opmth + '.' + ftmth
    vdf = vdf[vdf.vol_id == volid]
    vdf.value_date = pd.to_datetime(vdf.value_date)

    vdf, pdf, edf, priceDF = prep_datasets(vdf, pdf, edf, start_date, end_date)


else:
    vdf, pdf, edf, priceDF = read_data(
        epath, '', pdt=pdt, opmth=opmth, ftmth=ftmth, start_date=start_date, end_date=end_date)


# sanity check date ranges
sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), start_date, end_date)


# create 130,000 vega atm straddles
pf = create_portfolio(pdt, opmth, ftmth, 'straddle', vdf, pdf, chars=[
    'call', 'put'], shorted=False, atm=True, greek='vega', greekval='130000')

# specify hedging logic
hedges = generate_hedges(hedgepath)

# get rollover dates according to opex
rollover_dates = get_rollover_dates(priceDF)

# run the simulation
grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log = run_simulation(
    vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage, slippage=gv.slippage)
