# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-05-19 20:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-05-19 21:28:46


from .portfolio import Portfolio
from .classes import Future, Option
from .prep_data import find_cdist, match_to_signals, get_min_start_date, clean_data, ciprice, civols, vol_by_delta
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
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
    pf = Portfolio()

    # create the underlying future
    ticksize = multipliers[pdt][-2]
    date = max(vdf.value_date.min(), pdf.value_date.min())
    uid = pdt + '  ' + ftmth
    ftprice = pdf[(pdf.underlying_id == uid) &
                  (pdf.value_date == date)].settle_value.values[0]
    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])
    ft = Future(ftmth, ftprice, pdt, shorted=False, ordering=order)

    # create the relevant options; get all relevant information
    volid = pdt + '  ' + opmth + '.' + ftmth

    if kwargs['atm']:
        strike = round(round(ftprice / ticksize) * ticksize, 2)

    if optype == 'straddle':
        char1, char2 = kwargs['chars']
        shorted = kwargs['shorted']
        tau = vdf[(vdf.value_date == date) &
                  (vdf.vol_id == volid)].tau.values[0]
        cpi1 = 'C' if char1 == 'call' else 'P'
        cpi2 = 'C' if char1 == 'call' else 'P'

        vol1 = vdf[(vdf.value_date == date) &
                   (vdf.vol_id == volid) &
                   (vdf.call_put_id == cpi1) &
                   (vdf.strike == strike)].settle_vol.values[0]

        vol2 = vdf[(vdf.value_date == date) &
                   (vdf.vol_id == volid) &
                   (vdf.call_put_id == cpi2) &
                   (vdf.strike == strike)].settle_vol.values[0]

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
            op1.update_lots(lots_req)
            op2.update_lots(lots_req)

            pf.add_security([op1, op2], 'OTC')
    return pf


def prep_datasets(vdf, pdf, edf, start_date, end_date, specpath='', signals=None, test=False, write=False):
    """Utility function that does everything prep_data does, but to full datasets rather than things drawn from the database. Made because i was lazy. 

    Args:
        vdf (TYPE): Dataframe of vols
        pdf (TYPE): Datafrane of prices
        edf (TYPE): Dataframe of option expiries 
    """
    # edf = pd.read_csv(epath).dropna()

    vid_list = vdf.vol_id.unique()

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

engine = create_engine(
    'postgresql://sumit:Olam1234@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
connection = engine.connect()
