
from scripts.prep_data import read_data, generate_hedges, sanity_check, get_rollover_dates
# from scripts.classes import Option, Future
import itertools
import numpy as np
import pandas as pd
import scripts.global_vars as gv
from simulation import run_simulation
from scripts.util import prep_datasets, create_underlying, create_vanilla_option, create_barrier_option
from scripts.portfolio import Portfolio
import os


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


yrs = [6]
pnls = []
op = None

# , 'cuo', 'cdi', 'cdo', 'pui', 'puo', 'pdi', 'pdo']
flags = ['putop']

for yr, flag in itertools.product(yrs, flags):
    # vdf, pdf, df = pull_alt_data('W')

    target = 'U'

    # for yr in yrs:
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
    end_date = pd.Timestamp('201' + str(yr) + '-06-30')

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
        vdf = vdf[(vdf.value_date >= start_date)
                  & (vdf.value_date <= end_date)]
        pdf = pdf[(pdf.value_date >= start_date)
                  & (pdf.value_date <= end_date)]

        vdf, pdf, edf, priceDF, start_date = prep_datasets(
            vdf, pdf, edf, start_date, end_date)

        print('w_position - start date: ', start_date)

        if not os.path.isdir('datasets/' + pdt.lower()):
            os.mkdir('datasets/' + pdt.lower())
        print('finished pulling data')

        vdf.to_csv('datasets/' + pdt.lower() + '/debug_vols.csv', index=False)
        pdf.to_csv('datasets/' + pdt.lower() +
                   '/debug_prices.csv', index=False)

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

    #### W Qx.Ux - Long Positions ####
    # W Qx.Ux Put 430 900 lots - 34
    if flag == 'callop':
        op = create_vanilla_option(vdf, pdf, volid2, 'call', False,
                                   start_date, lots=900, delta=34, bullet=False)

    elif flag == 'putop':
        op = create_vanilla_option(vdf, pdf, volid2, 'put', False,
                                   start_date, lots=900, delta=34, bullet=False)

    elif flag == 'cui':
        # 500 CUI 520
        op = create_barrier_option(vdf, pdf, volid2, 'call', 500, False,
                                   start_date, 'amer', 'up', ki=520, ko=None,
                                   bullet=True, lots=900)
    elif flag == 'cuo':
        # 500 CUO 520
        op = create_barrier_option(vdf, pdf, volid2, 'call', 500, False,
                                   start_date, 'amer', 'up', ki=None, ko=530,
                                   bullet=True, lots=900)
    # 500 CDI 460
    elif flag == 'cdi':
        op = create_barrier_option(vdf, pdf, volid2, 'call', 500, False,
                                   start_date, barriertype='amer', direction='down',
                                   ki=460, ko=None, bullet=True, lots=900)
    elif flag == 'cdo':
        # 500 CDO 460
        op = create_barrier_option(vdf, pdf, volid2, 'call', 500, False,
                                   start_date, barriertype='amer', direction='down',
                                   ki=None, ko=460, bullet=True, lots=900)
    elif flag == 'pui':
        # 500 PUI 520
        op = create_barrier_option(vdf, pdf, volid2, 'put', 500, False,
                                   start_date, barriertype='amer', direction='up',
                                   ki=520, ko=None, bullet=True, lots=900)

    elif flag == 'puo':
        # 500 PUO 520
        op = create_barrier_option(vdf, pdf, volid2, 'put', 500, False,
                                   start_date, barriertype='amer', direction='up',
                                   ki=None, ko=520, bullet=True, lots=900)
    elif flag == 'pdi':
        # 500 PDI 460
        op = create_barrier_option(vdf, pdf, volid2, 'put', 500, False,
                                   start_date, barriertype='amer', direction='down',
                                   ki=460, ko=None, bullet=True, lots=900)
    elif flag == 'pdo':
        # 500 PDO 460
        op = create_barrier_option(vdf, pdf, volid2, 'put', 500, False,
                                   start_date, barriertype='amer', direction='down',
                                   ki=None, ko=460, bullet=True, lots=900)

    pf = Portfolio()
    pf.add_security(op, 'OTC')

    # ops = [op1, op2, op3, op4, op5, op6, op7, op8, op9]

    pf_delta = pf.net_greeks['W']['U' + str(yr)][0]

    print('net pf delta: ', pf_delta)

    shorted = True if pf_delta > 0 else False

    ### Outstanding Future Position ###
    fts, ftprice2 = create_underlying(
        pdt, ftmth, pdf, start_date, shorted=shorted, lots=round(abs(pf_delta)))

    fts = [fts]
    pf.add_security(fts, 'hedge')

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
    # # # # run the simulation
    # grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log = run_simulation(
    #     vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage,
    #     slippage=gv.slippage)

    # pnls.append(netpnl)
    # log.to_csv('results/' + flag + '_' + str(yr) +
    #            '_daily_log.csv', index=False)


# print(pnls)
