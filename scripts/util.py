# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-05-19 20:56:16
# @Last Modified by:   Ananth
# @Last Modified time: 2017-05-22 13:36:23


from .portfolio import Portfolio
from .classes import Future, Option
from .prep_data import find_cdist, match_to_signals, get_min_start_date, clean_data, ciprice, civols, vol_by_delta
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import time


multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'LCC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 20, 50],
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


def create_portfolio(pdt, opmth, ftmth, optype, vdf, pdf, **kwargs):
    """Helper function that generates common portfolio types

    Args:
        pdt (TYPE): Product
        opmth (TYPE): Option month, e.g. K7
        ftmth (TYPE): Future month, e.g. K7
        type (TYPE): structure name. valid inputs are call, put, callspread, putspread, fence, straddle, strangle, call_butterfly and put_butterfly
        vdf (TYPE): dataframe of vols
        pdf (TYPE): dataframe of prices
        kwargs (dict): dictionary of the form {'strikes':[], 'char':[], 'shorted':[], 'lots': [], 'greek': str, 'greekvals': [], 'atm': bool}
    """
    print('kwargs: ', kwargs)
    pf = Portfolio()

    # create the underlying future
    ticksize = multipliers[pdt][-2]
    date = max(vdf.value_date.min(), pdf.value_date.min())
    ft, ftprice = create_underlying(pdt, ftmth, pdf, date)

    # create the relevant options; get all relevant information
    volid = pdt + '  ' + opmth + '.' + ftmth

    if kwargs['atm']:
        strike = round(round(ftprice / ticksize) * ticksize, 2)

    if optype == 'straddle':
        shorted = kwargs['shorted']
        op1, op2 = create_straddle(
            volid, vdf, pdf, ft, date, shorted, strike, kwargs)

    pf.add_security([op1, op2], 'OTC')

    if 'hedges' in kwargs:
        print('creating hedges')
        dic = kwargs['hedges']
        if dic['type'] == 'straddle':
            # identifying the essentials
            pdt, ftmth, opmth = dic['pdt'], dic['ftmth'], dic['opmth']
            volid = pdt + '  ' + opmth + '.' + ftmth
            shorted = dic['shorted']
            # create the underlying future object
            h_ft, h_price = create_underlying(pdt, ftmth, pdf, date)
            h_strike = round(round(h_price / ticksize) * ticksize,
                             2) if dic['strike'] == 'atm' else dic['strike']
            # create_straddle(volid, vdf, pdf, ft, date, shorted, strike,
            # kwargs):
            h1, h2 = create_straddle(
                volid, vdf, pdf, h_ft, date, shorted, h_strike, dic, pf=pf)

        pf.add_security([h1, h2], 'hedge')

    return pf


def create_underlying(pdt, ftmth, pdf, date):
    """Utility method that creates the underlying future object given a product, month, price data and date. 

    Args:
        pdt (TYPE): product (e.g. 'S')
        ftmth (TYPE): month (e.g. N7)
        pdf (TYPE): Dataframe of prices 
        date (TYPE): Date 

    Returns:
        tuple: future object, and price. 
    """
    uid = pdt + '  ' + ftmth
    try:
        ftprice = pdf[(pdf.underlying_id == uid) &
                      (pdf.value_date == date)].settle_value.values[0]
    except IndexError:
        print('util.create_underlying: cannot find price. printing outputs: ')
        print('uid: ', uid)
        print('date: ', date)
        return
    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])
    ft = Future(ftmth, ftprice, pdt, shorted=False, ordering=order)

    return ft, ftprice


def create_vanilla_option(vdf, pdf, ft, strike, lots, volid, char, payoff, shorted, mth, date=None):
    """Utility method that creates an option from the info passed in. 

    Args:
        vdf (TYPE): Description
        pdf (TYPE): Description
        ft (TYPE): Description
        strike (TYPE): Description
        lots (TYPE): Description
        volid (TYPE): Description
        cpi (TYPE): Description
        date (None, optional): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        vol_id (TYPE): Description
        call_put_id (TYPE): Description
    """
    # get tau
    date = vdf.value_date.unique()[0] if date is None else date
    cpi = 'C' if char == 'call' else 'P'
    try:
        tau = vdf[(vdf.value_date == date) &
                  (vdf.vol_id == volid)].tau.values[0]

    except IndexError:
        print('util.create_vanilla_option - cannot find tau.')
        print('inputs: ', date, volid)
    # get vol
    try:
        vol = vdf[(vdf.value_date == date) &
                  (vdf.vol_id == volid) &
                  (vdf.call_put_id == cpi) &
                  (vdf.strike == strike)].settle_vol.values[0]
    except IndexError:
        print('util.create_straddle - vol1 not found, inputs below: ')
        print('date: ', date)
        print('vol_id: ', volid)
        print('call_put_id: ', cpi)
        print('strike: ', strike)

    # (self, strike, tau, char, vol, underlying, payoff, shorted, month, direc=None, barrier=None, lots=1000, bullet=True, ki=None, ko=None, rebate=0, ordering=1e5, settlement='cash')
    newop = Option(strike, tau, char, vol, ft, payoff, shorted,
                   mth, lots=lots, ordering=ft.get_ordering())
    return newop


def create_straddle(volid, vdf, pdf, ft, date, shorted, strike, kwargs, pf=None):
    """Utility function that creates straddle given dataframes and arguments.

    Args:
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        ft (Future object): Future object underlying this straddle
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        strike (float): Strike of this straddle
        kwargs (dict): dictionary of parameters. contains info regarding greek value of straddle. 
        pf (portfolio, optional): Portfolio object. Used to determine lot-sizes of hedges on the fly. 

    Returns:
        TYPE: Description
    """

    opmth = volid.split('.')[0].split()[1]
    pdt = volid.split()[0]
    order = ft.get_ordering()
    char1, char2 = 'call', 'put'
    tau = vdf[(vdf.value_date == date) &
              (vdf.vol_id == volid)].tau.values[0]
    cpi1 = 'C' if char1 == 'call' else 'P'
    cpi2 = 'C' if char1 == 'call' else 'P'

    try:
        vol1 = vdf[(vdf.value_date == date) &
                   (vdf.vol_id == volid) &
                   (vdf.call_put_id == cpi1) &
                   (vdf.strike == strike)].settle_vol.values[0]
    except IndexError:
        print('util.create_straddle - vol1 not found, inputs below: ')
        print('date: ', date)
        print('vol_id: ', volid)
        print('call_put_id: ', cpi1)
        print('strike: ', strike)

    try:
        vol2 = vdf[(vdf.value_date == date) &
                   (vdf.vol_id == volid) &
                   (vdf.call_put_id == cpi2) &
                   (vdf.strike == strike)].settle_vol.values[0]
    except IndexError:
        print('util.create_straddle - vol2 not found, inputs below: ')
        print('date: ', date)
        print('vol_id: ', volid)
        print('call_put_id: ', cpi2)
        print('strike: ', strike)

    op1 = Option(strike, tau, char1, vol1, ft, 'amer',
                 shorted, opmth, ordering=order)
    op2 = Option(strike, tau, char2, vol2, ft, 'amer',
                 shorted, opmth, ordering=order)

    # determine number of lots based on greek and greekvalue
    if kwargs['greek'] == 'vega':

        vega_req = float(kwargs['greekval'])
        # curr_vega = op1.vega + op2.vega
        lm, dm = multipliers[pdt][1], multipliers[pdt][0]
        v1 = (op1.vega * 100) / (op1.lots * dm * lm)
        v2 = (op2.vega * 100) / (op2.lots * dm * lm)
        lots_req = round((abs(vega_req) * 100) / ((v1 + v2) * lm * dm))

    elif kwargs['greek'] == 'gamma':
        if pf is not None and kwargs['greekval'] == 'portfolio':
            gamma_req = -pf.net_gamma_pos()
        else:
            gamma_req = float(kwargs['greekval'])
        print('gamma req: ', gamma_req)
        lm, dm = multipliers[pdt][1], multipliers[pdt][0]
        # gamma = (gamma1*lots*lm)/(dm)
        g1 = (op1.gamma * dm) / (op1.lots * lm)
        g2 = (op2.gamma * dm) / (op2.lots * lm)
        lots_req = round(((gamma_req) * dm) / ((g1 + g2) * lm))

    elif kwargs['greek'] == 'theta':
        pass

    op1.update_lots(lots_req)
    op2.update_lots(lots_req)

    return op1, op2


def prep_datasets(vdf, pdf, edf, start_date, end_date, specpath='', signals=None, test=False, write=False):
    """Utility function that does everything prep_data does, but to full datasets rather than things drawn from the database. Made because i was lazy. 

    Args:
        vdf (TYPE): Dataframe of vols
        pdf (TYPE): Datafrane of prices
        edf (TYPE): Dataframe of option expiries 
    """
    # edf = pd.read_csv(epath).dropna()

    vid_list = vdf.vol_id.unique()
    p_list = pdf.underlying_id.unique()

    print('util.prep_datasets - vid_list: ', vid_list)
    print('util.prep_datasets - p_list: ', p_list)

    if os.path.exists(specpath):
        specs = pd.read_csv(specpath)
        vid_list = specs[specs.Type == 'Option'].vol_id.unique()

    # fixing datetimes
    vdf.value_date = pd.to_datetime(vdf.value_date)
    pdf.value_date = pd.to_datetime(pdf.value_date)
    vdf = vdf.sort_values('value_date')
    pdf = pdf.sort_values('value_date')

    # case 1: drawing based on portfolio.
    if signals is not None:
        signals.value_date = pd.to_datetime(signals.value_date)
        vdf, pdf = match_to_signals(vdf, pdf, signals)

    print('vid list: ', vid_list)

    # get effective start date, pick whichever is max

    # case 2: drawing based on pdt, ft and opmth
    dataset_start_date = get_min_start_date(
        vdf, pdf, vid_list, signals=signals)
    print('datasets start date: ', dataset_start_date)

    dataset_start_date = pd.to_datetime(dataset_start_date)

    start_date = dataset_start_date if (start_date is None) or \
        ((start_date is not None) and (dataset_start_date > start_date)) else start_date

    print('prep_data start_date: ', start_date)

    # filtering relevant dates
    vdf = vdf[(vdf.value_date >= start_date) &
              (vdf.value_date <= end_date)] \
        if end_date \
        else vdf[(vdf.value_date >= start_date)]

    pdf = pdf[(pdf.value_date >= start_date) &
              (pdf.value_date <= end_date)] \
        if end_date \
        else pdf[(pdf.value_date >= start_date)]

    # catch errors
    if (vdf.empty or pdf.empty):
        raise ValueError(
            '[scripts/prep_data.read_data] : Improper start date entered; resultant dataframes are empty')
    print('pdf: ', pdf)
    print('vdf: ', vdf)
    # clean dataframes
    edf = clean_data(edf, 'exp')
    vdf = clean_data(vdf, 'vol', date=start_date,
                     edf=edf)
    pdf = clean_data(pdf, 'price', date=start_date,
                     edf=edf)

    # final preprocessing steps
    final_price = ciprice(pdf)
    print('final price: ', final_price)
    final_vol = civols(vdf, final_price)

    print('sanity checking date ranges')
    if not np.array_equal(pd.to_datetime(final_vol.value_date.unique()),
                          pd.to_datetime(final_price.value_date.unique())):
        vmask = final_vol.value_date.isin([x for x in final_vol.value_date.unique()
                                           if x not in final_price.value_date.unique()])
        pmask = final_price.value_date.isin([x for x in final_price.value_date.unique()
                                             if x not in final_vol.value_date.unique()])
        final_vol = final_vol[~vmask]
        final_price = final_price[~pmask]

    if not test:
        vbd = vol_by_delta(final_vol, final_price)

        # merging vol_by_delta and price dataframes on product, underlying_id,
        # value_date and order
        vbd.underlying_id = vbd.underlying_id.str.split().str[0]\
            + '  ' + vbd.underlying_id.str.split().str[1]
        final_price.underlying_id = final_price.underlying_id.str.split().str[0]\
            + '  ' + final_price.underlying_id.str.split().str[1]
        merged = pd.merge(vbd, final_price, on=[
                          'pdt', 'value_date', 'underlying_id', 'order'])
        final_price = merged

        # handle conventions for vol_id in price/vol data.
        final_vol.vol_id = final_vol.vol_id.str.split().str[0]\
            + '  ' + final_vol.vol_id.str.split().str[1]
        final_price.vol_id = final_price.vol_id.str.split().str[0]\
            + '  ' + final_price.vol_id.str.split().str[1]

    return final_vol, final_price, edf, pdf


def pull_alt_data():
    """Utility function that draws/cleans data from the alternate data table. 

    Returns:
        TYPE: Description
    """
    print('starting clock..')
    t = time.clock()
    engine = create_engine(
        'postgresql://sumit:Olam1234@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    query = "select security_id, settlement_date, future_settlement_value, option_expiry_date,implied_vol \
            FROM table_option_settlement_data where security_id like 'S %%' and extract(YEAR from settlement_date) > 2009"

    df = pd.read_sql_query(query, connection)

    print('finished pulling data')
    print('elapsed: ', time.clock() - t)

    # cleans up data
    df['vol_id'] = df.security_id.str[:9]
    df['call_put_id'] = df.security_id.str[9:11].str.strip()
    # getting 'S' from 'S Q17.Q17'
    df['pdt'] = df.vol_id.str.split().str[0].str.strip()

    # getting opmth
    df['opmth'] = df.vol_id.str.split(
        '.').str[0].str.split().str[1].str.strip()
    df.opmth = df.opmth.str[0] + \
        (df.opmth.str[1:].astype(int) % 10).astype(str)

    df['strike'] = df.security_id.str[10:].astype(float)

    df['implied_vol'] = df.implied_vol / 100

    # Q17 -> Q7
    df['ftmth'] = df.vol_id.str.split('.').str[1].str.strip()
    df.ftmth = df.ftmth.str[0] + \
        (df.ftmth.str[1:].astype(int) % 10).astype(str)

    # creating underlying_id and vol_id
    df.vol_id = df.pdt + '  ' + df.opmth + '.' + df.ftmth
    df['underlying_id'] = df.pdt + '  ' + df.ftmth

    # selecting
    vdf = df[['settlement_date', 'vol_id',
              'call_put_id', 'strike', 'implied_vol']]
    pdf = df[['settlement_date', 'underlying_id', 'future_settlement_value']]

    vdf.columns = ['value_date', 'vol_id',
                   'call_put_id', 'strike', 'settle_vol']
    pdf.columns = ['value_date', 'underlying_id', 'settle_value']

    # removing duplicates, resetting indices
    pdf = pdf.drop_duplicates()
    vdf = vdf.drop_duplicates()

    pdf.reset_index(drop=True, inplace=True)
    vdf.reset_index(drop=True, inplace=True)

    if not os.path.isdir('datasets/data_dump'):
        os.mkdir('datasets/data_dump')

    vdf.to_csv('datasets/data_dump/s_vol_dump.csv', index=False)
    pdf.to_csv('datasets/data_dump/s_price_dump.csv', index=False)

    return vdf, pdf, df
