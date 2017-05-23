
from scripts.prep_data import read_data, generate_hedges, sanity_check, get_rollover_dates
import numpy as np
import pandas as pd
import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import create_portfolio, prep_datasets, pull_alt_data, create_underlying, create_vanilla_option
from scripts.portfolio import Portfolio


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

yrs = [5]
pnls = []

# vdf, pdf, df = pull_alt_data('W')

target = 'U'

for yr in yrs:
    pdt = 'W'
    ftlist = ['U']
    oplist = ['Q', 'U']
    # ftmth = 'X2'
    # opmth = 'X2'

    alt = True
    # alt = True if yr >= 5 else True

    pricedump = 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'
    voldump = 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'

    start_date = pd.Timestamp('201' + str(yr) + '-05-23')
    end_date = pd.Timestamp('201' + str(yr) + '-08-31')

    # vdf, pdf, df = pull_alt_data()

    if alt:
        # uids = 'U' + str(yr)
        # opmths = []
        print('pulling data')
        edf = pd.read_csv(epath)
        uids = [pdt + '  ' + u + str(yr) for u in ftlist]
        print('uids: ', uids)
        volids = [pdt + '  ' + i + str(yr) + '.' + u + str(yr)
                        for i in oplist for u in ftlist]
        print('volids: ', volids)
        pdf = pd.read_csv(pricedump)
        pdf.value_date = pd.to_datetime(pdf.value_date)
        # cleaning prices
        pmask = pdf.underlying_id.isin(uids)
        pdf = pdf[pmask]
        vdf = pd.read_csv(voldump)
        # volid = pdt + '  ' + opmth + '.' + ftmth
        vmask = vdf.vol_id.isin(volids)
        vdf = vdf[vmask]
        vdf.value_date = pd.to_datetime(vdf.value_date)
        # filter datasets before prep
        # vdf = vdf[(vdf.value_date >= start_date)
        #           & (vdf.value_date <= end_date)]
        # pdf = pdf[(pdf.value_date >= start_date)
        #           & (pdf.value_date <= end_date)]

        vdf, pdf, edf, priceDF = prep_datasets(
            vdf, pdf, edf, start_date, end_date)
        print('finished pulling data')

    else:
        opmth = target + str(yr)
        ftmth = target + str(yr)
        print('pulling data')
        vdf, pdf, edf, priceDF = read_data(
            epath, '', pdt=pdt, opmth=opmth, ftmth=ftmth, start_date=start_date, end_date=end_date)
        print('finished pulling data')

    print('sanity checking data')
    # sanity check date ranges
    sanity_check(vdf.value_date.unique(),
                 pdf.value_date.unique(), start_date, end_date)

    print('voldata: ', vdf)
    print('pricedf: ', pdf)

    print('creating portfolio')
    # create 130,000 vega atm straddles

    # hedge_specs = {'pdt': 'S',
    #                'opmth': 'N' + str(yr),
    #                'ftmth': 'N' + str(yr),
    #                'type': 'straddle',
    #                'strike': 'atm',
    #                'shorted': True,
    #                'greek': 'gamma',
    #                'greekval': 'portfolio'}

    opmth1 = target + str(yr)
    opmth2 = 'Q' + str(yr)
    ftmth = target + str(yr)

    volid1 = pdt + '  ' + opmth1 + '.' + ftmth  # W UX.UX
    volid2 = pdt + '  ' + opmth2 + '.' + ftmth  # W QX.UX

    ft, ftprice = create_underlying(pdt, ftmth, pdf, start_date)
    print('underlying: ', ft)

    op1 = create_vanilla_option(
        vdf, pdf, ft, 430, volid2, 'put', 'amer', False, opmth2, lots=900)
    op2 = create_vanilla_option(
        vdf, pdf, ft, 440, volid2, 'put', 'amer', False, opmth2, lots=500)
    op3 = create_vanilla_option(
        vdf, pdf, ft, 450, volid1, 'call', 'amer', False, opmth1, lots=450)
    op4 = create_vanilla_option(
        vdf, pdf, ft, 460, volid1, 'call', 'amer', False, opmth1, lots=175)
    op5 = create_vanilla_option(
        vdf, pdf, ft, 480, volid1, 'call', 'amer', True, opmth1, lots=231)
    fts, _ = create_underlying(pdt, ftmth, pdf, start_date, lots=509)

    pf = Portfolio()
    ops = [op1, op2, op3, op4, op5]
    fts = [fts]
    pf.add_security(ops, 'OTC')
    pf.add_security(fts, 'hedge')

    # pf = create_portfolio(pdt, opmth, ftmth, 'skew', vdf, pdf, chars=[
    #     'call', 'put'], shorted=True, delta=25, greek='vega', greekval='25000')

    # pf = create_portfolio(pdt, opmth, ftmth, 'straddle', vdf, pdf, chars=[
    #     'call', 'put'], shorted=False, atm=True, greek='vega', greekval='130000', hedges=hedge_specs)

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
    rollover_dates = get_rollover_dates(priceDF)
    print('rollover dates: ', rollover_dates)

    print('running simulation')
    # run the simulation
    grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log = run_simulation(
        vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage,
        slippage=gv.slippage)

    pnls.append(netpnl)

# bound = '_20_30'
# bound = '_10_40'
# log.to_csv('results/kc/201' + str(yr) + bound + '_log.csv', index=False)
print(pnls)
