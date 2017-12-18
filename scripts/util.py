# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-05-19 20:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-15 23:44:07

from .portfolio import Portfolio
from .classes import Future, Option
from .prep_data import find_cdist, handle_dailies
import pandas as pd
import numpy as np
import copy
import os
from .calc import compute_strike_from_delta, get_vol_from_delta
import sys

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 1, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
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
    'CA': [1.0604, 50, 0.25, 1, 53.02],
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
    'QC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V', 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'CA': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


# def create_portfolio(pdt, opmth, ftmth, optype, vdf, pdf, **kwargs):
#     """Helper function that generates common portfolio types. Delegates construction to specific
#      constructors, based on the input passed in.

#     Args:
#         pdt (TYPE): Product
#         opmth (TYPE): Option month, e.g. K7
#         ftmth (TYPE): Future month, e.g. K7
#         optype (TYPE): structure name. valid inputs are call, put, callspread, putspread, fence, straddle, strangle, call_butterfly and put_butterfly
#         vdf (TYPE): dataframe of vols
#         pdf (TYPE): dataframe of prices
#         **kwargs: dictionary of the form {'strikes':[], 'char':[], 'shorted':[], 'lots': [], 'greek': str, 'greekvals': [], 'atm': bool}

#     Returns:
#         Portfolio object: The portfolio being created.
#     """
#     print('kwargs: ', kwargs)
#     pf = Portfolio()

#     # create the underlying future
#     ticksize = multipliers[pdt][-2]
#     date = max(vdf.value_date.min(), pdf.value_date.min())
#     _, ftprice = create_underlying(pdt, ftmth, pdf, date, shorted=False)

#     # create the relevant options; get all relevant information
#     volid = pdt + '  ' + opmth + '.' + ftmth
#     shorted = kwargs['shorted']

#     if 'atm' in kwargs and kwargs['atm']:
#         strike = round(round(ftprice / ticksize) * ticksize, 2)

#     if optype == 'straddle':
#         op1, op2 = create_straddle(
#             volid, vdf, pdf, date, shorted, strike, kwargs)
#         ops = [op1, op2]

#     elif optype == 'skew':
#         delta = kwargs['delta']
#         op1, op2 = create_skew(volid, vdf, pdf, date,
#                                shorted, delta, ftprice, kwargs)
#         ops = [op1, op2]

#     elif optype == 'vanilla':
#         strike = kwargs['strike'] if 'strike' in kwargs else None
#         char = kwargs['char']
#         delta = kwargs['delta'] if 'delta' in kwargs else None
#         lots = kwargs['lots'] if 'lots' in kwargs else None
#         op1 = create_vanilla_option(vdf, pdf, volid, char, shorted,
#                                     date, lots=lots, delta=delta, strike=strike, kwargs=kwargs)
#         ops = [op1]

#     elif optype == 'spread':
#         char = kwargs['char']
#         op1, op2 = create_spread(
#             char, volid, vdf, pdf, date, shorted, kwargs)
#         ops = [op1, op2]

#     elif optype == 'fence':
#         delta = kwargs['delta']
#         op1, op2 = create_skew(volid, vdf, pdf, date,
#                                not shorted, delta, kwargs)
#         ops = [op1, op2]

#     elif optype == 'strangle':
#         op1, op2 = create_strangle(volid, vdf, pdf, date, shorted, kwargs)
#         ops = [op1, op2]

#     elif optype == 'butterfly':
#         char = kwargs['char']
#         op1, op2, op3, op4 = create_butterfly(
#             char, volid, vdf, pdf, date, shorted, kwargs)
#         ops = [op1, op2, op3, op4]

#     pf.add_security(ops, 'OTC')

#     if 'hedges' in kwargs:
#         print('creating hedges')
#         dic = kwargs['hedges']
#         if dic['type'] == 'straddle':
#             # identifying the essentials
#             pdt, ftmth, opmth = dic['pdt'], dic['ftmth'], dic['opmth']
#             volid = pdt + '  ' + opmth + '.' + ftmth
#             shorted = dic['shorted']
#             # create the underlying future object
#             h_ft, h_price = create_underlying(
#                 pdt, ftmth, pdf, date, shorted=False)
#             h_strike = round(round(h_price / ticksize) * ticksize, 2) \
#                 if dic['strike'] == 'atm' else dic['strike']
#             h1, h2 = create_straddle(
#                 volid, vdf, pdf, date, shorted, h_strike, dic, pf=pf)

#         pf.add_security([h1, h2], 'hedge')

#     return pf


def create_underlying(pdt, ftmth, pdf, date, flag='settlement', ftprice=None, shorted=False, lots=None):
    """Utility method that creates the underlying future object 
        given a product, month, price data and date. 

    Args:
        pdt (TYPE): product (e.g. 'S')
        ftmth (TYPE): month (e.g. N7)
        pdf (TYPE): Dataframe of prices 
        date (TYPE): Date 
        shorted (bool, optional): indicates if the future is shorted or not
        lots (None, optional): lot size to use for this future. 

    Returns:
        tuple: future object, and price. 
    """
    # print('pdf: ', pdf)
    # datatype = 'settlement' if settlement else 'intraday'
    flag = 'settlement' if flag == 'eod' else flag
    uid = pdt + '  ' + ftmth
    if ftprice is None:
        try:
            ftprice = pdf[(pdf.underlying_id == uid) &
                          (pdf.value_date == date)]
            if flag == 'settlement':
                ftprice = ftprice[(ftprice.datatype == flag)].price.values[0]
            else:
                ftprice = ftprice.price.values[0]

        except IndexError:
            print('util.create_underlying: cannot find price. printing outputs: ')
            print('uid: ', uid)
            print('date: ', date)
            print('flag: ', flag)
            return None, 0

    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])
    ft = Future(ftmth, ftprice, pdt, shorted=shorted, ordering=order)
    if lots is not None:
        if lots == 0:
            return None, ftprice
        ft.update_lots(lots)

    return ft, ftprice


def create_vanilla_option(vdf, pdf, volid, char, shorted, date=None,
                          payoff='amer', lots=None, delta=None,
                          strike=None, vol=None, bullet=True, **kwargs):
    """Utility method that creates an option from the info passed in.
         Each option is instantiated with its own future underlying object. 

    Args:
        vdf (dataframe): dataframe of volatilities
        pdf (dataframe): dataframe of prices and vols by delta
        volid (string): vol_id of the option
        char (string): call or put
        shorted (bool): True or False
        date (pd.Timestamp, optional): date to use when selecting vol
        payoff (string): american or european payoff
        lots (int, optional): number of lots 
        kwargs (dict, optional): dictionary containing extra delimiting factors, 
                                e.g. greek/greekvalue. 
        delta (float, optional): Delta of this option as an alternative to strike. 
        strike (float): strike of the option
        vol (float, optional): volatility of the option

    Returns:
        object: Option created according to the parameters passed in. 


    Deleted Parameters:
        mth (string): month of the option, e.g. N7

    Raises:
        ValueError: Description


    """
    # sanity checks
    # print('util.create_vanilla_option - inputs: ',
    #       volid, char, shorted, lots, delta, strike, vol, kwargs)

    if delta is None and strike is None:
        raise ValueError(
            'neither delta nor strike passed in; aborting construction.')

    lots_req = lots if lots is not None else 1000

    # naming conventions
    ftmth = volid.split('.')[1]
    pdt = volid.split()[0]
    opmth = volid.split()[1].split('.')[0]
    cpi = 'C' if char == 'call' else 'P'

    # get min start date for debugging
    min_start_date = min(vdf[vdf.pdt == pdt].value_date)

    date = min_start_date if (date is None or min_start_date > date) else date

    # create the underlying future
    ft_shorted = shorted if char == 'call' else not shorted
    # print('util.create_vanilla_option - pdf.columns: ', pdf.columns)
    ft, ftprice = create_underlying(pdt, ftmth, pdf, date,
                                    shorted=ft_shorted, lots=lots)

    ticksize = multipliers[pdt][-2]

    # if strike is set to 'atm', round closest optiontick to underlying price.
    if strike == 'atm':
        strike = round(round(ftprice / ticksize) * ticksize, 2)

    # get tau
    try:
        if 'tau' in kwargs and kwargs['tau'] is not None:
            print('tau in kwargs')
            tau = kwargs['tau']
        else:
            tau = vdf[(vdf.value_date == date) &
                      (vdf.vol_id == volid)].tau.values[0]
    except IndexError as e:
        print('debug_1: ', vdf[vdf.value_date == date])
        print('debug_2: ', vdf[vdf.vol_id == volid])
        raise IndexError(
            'util.create_vanilla_option - cannot find ttm in dataset. Inputs are: ', date, volid) from e

    # case: want to create an option with a specific breakeven. given price,
    # compute the vol
    if 'breakeven' in kwargs and kwargs['breakeven'] is not None:
        # pnl_mult = multipliers[pdt][-1]
        vol = ((252**0.5) * kwargs['breakeven'])/(ftprice)

    # Case 1 : Vol is None, but strike is specified.
    elif vol is None and strike is not None:
        # get vol
        try:
            # print("Inputs: ", date.strftime('%Y-%m-%d'), volid, cpi, strike)
            vol = vdf[(vdf.value_date == date) &
                      (vdf.vol_id == volid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike) &
                      (vdf.datatype == 'settlement')].vol.values[0]
        except IndexError as e:
            raise IndexError(
                'util.create_vanilla_option - vol not found in the dataset. inputs are: ', date, volid, cpi, strike) from e

    # Case 2: Option construction is basis delta. vol and strike are None.
    elif vol is None and strike is None:
        try:
            delta = delta/100
            vol = get_vol_from_delta(
                delta, vdf, pdf, volid, char, shorted, date)
            strike = compute_strike_from_delta(None, delta1=delta, vol=vol, s=ft.get_price(),
                                               char=char, pdt=ft.get_product(), tau=tau)
        except IndexError as e:
            raise IndexError(
                'util.create_vanilla_option - cannot find vol by delta. Inputs are: ', volid, date, delta, cpi) from e
        except ValueError as e1:
            raise ValueError(getattr(e1, 'message', repr(e1)))

    # if delta specified, compute strike appropriate to that delta
    if delta is not None:
        # delta = delta/100
        strike = compute_strike_from_delta(
            None, delta1=delta, vol=vol, s=ft.get_price(), char=char, pdt=ft.get_product(), tau=tau)

    # specifying option with information gathered thus far.
    newop = Option(strike, tau, char, vol, ft, payoff, shorted,
                   opmth, lots=lots_req, ordering=ft.get_ordering(),
                   bullet=bullet)

    # handling bullet vs daily
    if not bullet:
        tmp = {'OTC': [newop]}
        ops = handle_dailies(tmp, date)
        ops = ops['OTC']
        return ops

    # handling additional greek requirements for options.
    pdt = ft.get_product()
    if kwargs is not None and 'greek' in kwargs:
        lots_req = 0
        if kwargs['greek'] == 'theta':
            theta_req = float(kwargs['greekval'])
            print('theta req: ', theta_req)
            lm, dm = multipliers[pdt][1], multipliers[pdt][0]
            t1 = (newop.theta * 365)/(newop.lots * lm * dm)
            lots_req = round((abs(theta_req) * 365) / abs(t1 * lm * dm))

        if kwargs['greek'] == 'vega':
            vega_req = float(kwargs['greekval'])
            lm, dm = multipliers[pdt][1], multipliers[pdt][0]
            v1 = (newop.vega * 100) / (newop.lots * dm * lm)
            lots_req = round(abs(vega_req * 100) / (abs(v1) * lm * dm))
            print('lots req: ', lots_req)

        if kwargs['greek'] == 'gamma':
            gamma_req = float(kwargs['greekval'])
            lm, dm = multipliers[pdt][1], multipliers[pdt][0]
            g1 = (gamma_req * dm) / (newop.lots * lm)
            lots_req = round(abs(gamma_req) * dm) / (abs(g1) * lm)
            print('lots_req: ', lots_req)

        newop.update_lots(lots_req)
        newop.underlying.update_lots(lots_req)
    return newop


# def __init__(self, strike, tau, char, vol, underlying, payoff, shorted,
# month, direc=None, barrier=None, lots=1000, bullet=True, ki=None,
# ko=None, rebate=0, ordering=1e5, settlement='futures')

def create_barrier_option(vdf, pdf, volid, char, strike, shorted, date, barriertype,
                          direction, ki, ko, bullet, rebate=0, payoff='amer', lots=None,
                          kwargs=None, vol=None, bvol=None):
    """Helper method that creates barrier options. 

    Args:
        vdf (TYPE): dataframe of vols
        pdf (TYPE): dataframe of prices
        volid (TYPE): vol_id of this barrier option
        char (TYPE): call/put
        strike (TYPE): strike price. 
        shorted (TYPE): True or False
        date (TYPE): date of initialization
        barriertype (TYPE): amer or euro barrier. 
        direction (TYPE): up or down
        ki (TYPE): knockin value 
        ko (TYPE): knockout value
        bullet (TYPE): True if bullet, false if daily. 
        rebate (int, optional): rebate value.
        payoff (str, optional): amer or euro option
        lots (None, optional): number of lots
        kwargs (None, optional): additional parameters (greeks, etc)
        vol (None, optional): vol at strike
        bvol (None, optional): vol at barrier

    Deleted Parameters:
        delta (None, optional): specif

    Returns:
        TYPE: Description

    No Longer Raises:
        ValueError: Description
    """
    print('util.create_barrier_option - inputs: ',
          volid, char, shorted, lots, strike, vol)

    # if delta is None and strike is None:
    #     raise ValueError(
    #         'neither delta nor strike passed in; aborting construction.')

    lots_req = lots if lots is not None else 1000

    # naming conventions
    ftmth = volid.split('.')[1]
    pdt = volid.split()[0]
    opmth = volid.split()[1].split('.')[0]
    cpi = 'C' if char == 'call' else 'P'

    # create the underlying future
    ft_shorted = shorted if char == 'call' else not shorted
    # print('util.create_vanilla_option - pdf.columns: ', pdf.columns)
    ft, ftprice = create_underlying(pdt, ftmth, pdf, date,
                                    shorted=ft_shorted, lots=lots_req)

    # get tau
    try:
        tau = vdf[(vdf.value_date == date) &
                  (vdf.vol_id == volid)].tau.values[0]
    except IndexError as e:
        raise IndexError(
            'util.create_barrier_option - cannot find tau given inpits: ', date, volid) from e

    # Case 1 : Vol is None, but strike is specified.
    if vol is None and strike is not None:
        # get vol
        try:
            vol = vdf[(vdf.value_date == date) &
                      (vdf.vol_id == volid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike)].vol.values[0]
        except IndexError as e:
            raise IndexError(
                'util.create_barrier_option - strike not found, input: ', date, volid, cpi, strike)

    # get barrier vol.
    barlevel = ko if ko is not None else ki
    if bvol is None:
        try:
            bvol = vdf[(vdf.value_date == date) &
                       (vdf.vol_id == volid) &
                       (vdf.call_put_id == cpi) &
                       (vdf.strike == barlevel)].vol.values[0]
        except IndexError as e:
            raise IndexError(
                'util.create_barrier_option - bvol not found. inputs are: ', date, volid, cpi, barlevel) from e

    op1 = Option(strike, tau, char, vol, ft, payoff, shorted, opmth,
                 direc=direction, barrier=barriertype, lots=lots_req,
                 bullet=bullet, ki=ki, ko=ko, rebate=rebate, ordering=ft.get_ordering(), bvol=bvol)

    return op1


def create_butterfly(char, volid, vdf, pdf, date, shorted, **kwargs):
    """Utility method that creates a butterfly position. 

    Args:
        char (str): 'call' or 'put', specifies if this is a call or put butterfly.
        volid (str): volid of options comprising the butterfly, e.g. C  N7.N7
        vdf (pandas dataframe): dataframe of vols
        pdf (pandas dataframe): detaframe of prices
        date (pandas Timestamp): initialization date. 
        shorted (bool): determines if this is a short or long butterfly. 
        kwargs (dic): dictionary of specifications must include the following:
            > 3 strikes OR one delta and one dist 
            > 3 lot sizes. 

    Notes:
        1) Construction requires either:
            > 3 strikes. 
            > One delta and a spread. 

    Deleted Parameters:
        strike1 (TYPE): Description
        strike2 (TYPE): Description
        strike3 (TYPE): Description
        strike4 (TYPE): Description
        ft (TYPE): Description

    """
    # checks if strikes are passed in, and if there are 3 strikes.

    print('kwargs: ', kwargs)
    lower_strike, mid_strike, upper_strike, mid_delta, dist = [None] * 5

    lot1, lot2, lot3, lot4 = kwargs['lots']

    # print('delta in kwargs: ', 'delta' in kwargs)
    # print('strikes in kwargs: ', 'strikes' in kwargs)

    if ('strikes' not in kwargs) and ('delta' not in kwargs):
        raise ValueError(
            'neither strike nor mid-delta specified. aborting construction.')

    if 'strikes' in kwargs and len(kwargs['strikes']) == 3:
        print('explicit strikes construction')
        lower_strike, mid_strike, upper_strike = kwargs['strikes']

    elif 'delta' in kwargs and 'dist' in kwargs:
        print('delta/dist construction')
        mid_delta = kwargs['delta']
        # mid_delta = mid_delta/100
        dist = kwargs['dist']

    print('mid delta: ', mid_delta)
    print('lots: ', lot1, lot2, lot3, lot4)
    mid_op1 = create_vanilla_option(
        vdf, pdf, volid, char, not shorted, date, delta=mid_delta, strike=mid_strike, lots=lot2)

    mid_op2 = create_vanilla_option(
        vdf, pdf, volid, char, not shorted, date, delta=mid_delta, strike=mid_strike, lots=lot3)
    print('mid strikes: ', mid_op1.K, mid_op2.K)

    if dist is not None:
        lower_strike = mid_op2.K - dist
        upper_strike = mid_op2.K + dist

    lower_op = create_vanilla_option(
        vdf, pdf, volid, char, shorted, date, strike=lower_strike, lots=lot1)

    upper_op = create_vanilla_option(
        vdf, pdf, volid, char, shorted, date, strike=upper_strike, lots=lot4)

    ops = [lower_op, mid_op1, mid_op2, upper_op]

    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)
    return ops


def create_spread(char, volid, vdf, pdf, date, shorted, **kwargs):
    """Utility method that creates a callspread 

    Args:
        char (str): call or put, indicating if this is a callspread or a putspread. 
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        kwargs (dict): dictionary of the form 
                        {'chars': ['call', 'put'], 'strike': [strike1, strike2],
                        'greek':(gamma theta or vega), 
                        'greekval': the value used to determine lot size, 
                        'lots': lottage if greek not specified.}

    Deleted Parameters:
        ft (Future object): Future object underlying this straddle

    Returns:
        TYPE: Description
    """
    # identify if this spread is created with explicit strikes or with delta
    # values
    delta1, delta2, strike1, strike2 = None, None, None, None
    if 'delta' in kwargs:
        delta1, delta2 = kwargs['delta']
        # delta1, delta2 = delta1/100, delta2/100
    elif 'strike' in kwargs:
        strike1, strike2 = kwargs['strike']

    # long call spread : buy itm call, sell otm call.
    # short call spread: sell itm call, buy otm call

    # long put spread:   sell itm put, buy otm put,
    # short put spread:  buy itm put,  sell otm put,
    # make sure inputs are organized in the appropriate order.

    op1 = create_vanilla_option(
        vdf, pdf, volid, char, shorted, date, delta=delta1, strike=strike1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, char, not shorted, date, delta=delta2, strike=strike2)

    ops = [op1, op2]

    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)
    return ops

# update to allow for greeks to specify lottage.


def create_strangle(volid, vdf, pdf, date, shorted, pf=None, **kwargs):
    """Utility method that creates a strangle (long or short).

    Args:
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        kwargs (dict): dictionary of the form 
            {'chars': ['call', 'put'],
             'strike': [strike1, strike2],
             'greek':(gamma theta or vega),
             'greekval': the value used to determine lot size,
             'lots': lottage if greek not specified.}
        pf (portfolio, optional): Portfolio object. 

    Deleted Parameters:
        ft (Future object): Future object underlying this straddle

    Returns:
        TYPE: Description

    """
    print('kwargs: ', kwargs)
    assert 'chars' in kwargs
    assert isinstance(kwargs['chars'], list)
    assert len(kwargs['chars']) == 2

    c_delta = None

    strike1, strike2 = None, None
    delta1, delta2 = None, None
    lot1, lot2 = None, None
    pdt = volid.split()[0]
    char1, char2 = kwargs['chars'][0], kwargs['chars'][1]

    if 'delta' in kwargs and kwargs['delta'] is not None:
        # single delta value
        if isinstance(kwargs['delta'], (float, int)):
            c_delta = float(kwargs['delta'])
        else:
            delta1, delta2 = kwargs['delta'][0], kwargs['delta'][1]

    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    if 'strike' in kwargs and kwargs['strike'] is not None:
        strike1, strike2 = kwargs['strike'][0], kwargs['strike'][1]

    if 'lots' in kwargs and kwargs['lots'] is not None:
        lot1, lot2 = kwargs['lots'][0], kwargs['lots'][1]

    print('deltas: ', delta1, delta2)
    print('c_delta: ', c_delta)

    print('util.create_strangle - lot1, lot2: ', lot1, lot2)

    if c_delta is not None:
        f_delta1, f_delta2 = c_delta, c_delta
    else:
        f_delta1, f_delta2 = delta1, delta2

    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date, strike=strike1, lots=lot1, delta=f_delta1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, shorted, date, strike=strike2, lots=lot2, delta=f_delta2)

    # setting lots based on greek value passed in
    if 'greek' in kwargs:
        if kwargs['greek'] == 'vega':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                vega_req = -pf.net_theta_pos()
            else:
                vega_req = float(kwargs['greekval'])
            print('vega req: ', vega_req)
            v1 = (op1.vega * 100) / (op1.lots * dm * lm)
            v2 = (op2.vega * 100) / (op2.lots * dm * lm)
            print('v1 + v2: ', v1+v2)
            lots_req = round(abs(vega_req * 100) / (abs(v1 + v2) * lm * dm))
            print('lots_req: ', lots_req)

        elif kwargs['greek'] == 'gamma':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                gamma_req = -pf.net_gamma_pos()
            else:
                gamma_req = float(kwargs['greekval'])
            print('gamma req: ', gamma_req)
            g1 = (op1.gamma * dm) / (op1.lots * lm)
            g2 = (op2.gamma * dm) / (op2.lots * lm)
            lots_req = round((abs(gamma_req) * dm) / (abs(g1 + g2) * lm))

        elif kwargs['greek'] == 'theta':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                theta_req = -pf.net_theta_pos()
            else:
                theta_req = float(kwargs['greekval'])
            print('theta req: ', theta_req)
            t1 = (op1.theta * 365)/(op1.lots * lm * dm)
            t2 = (op2.theta * 365) / (op2.lots * lm * dm)
            lots_req = round((abs(theta_req) * 365) / (abs(t1 + t2) * lm * dm))

        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.underlying.update_lots(lots_req)
        op2.underlying.update_lots(lots_req)

    ops = [op1, op2]
    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)
    return ops


def create_skew(volid, vdf, pdf, date, shorted, delta, **kwargs):
    """Utility function that creates a skew position given dataframes and arguments. 

    Args:
        volid (string): vol_id, e.g. CT N7.N7
        vdf (dataframe): dataframe of strikewise vols
        pdf (dataframe): dataframe of prices and delta-wise vols
        date (pd Timestamp): initialization date 
        shorted (bool): True or False
        delta (float): delta value of this skew
        ftprice (float): Future price on DATE
        kwargs (dict): dictionary containing 1) greek and 2) greekvalue to specify lots. 

    Deleted Parameters:
        ft (object): underlying future object

    Returns:
        Tuple: Two options constituting a skew position (reverse fence). 
    """

    pdt = volid.split()[0]
    lm, dm = multipliers[pdt][1], multipliers[pdt][0]
    clot, plot = None, None
    if 'lots' in kwargs and kwargs['lots'] is not None:
        if isinstance(kwargs['lots'], (float, int)):
            clot, plot = kwargs['lots'], kwargs['lots']
        else:
            clot, plot = kwargs['lots'][0], kwargs['lots'][1]

    print('clot: ', clot)
    print('plot: ', plot)
    # creating the options
    op1 = create_vanilla_option(
        vdf, pdf, volid, 'call', shorted, date, delta=delta, lots=clot)
    op2 = create_vanilla_option(
        vdf, pdf, volid, 'put', not shorted, date, delta=delta, lots=plot)

    if 'greek' in kwargs:
        if kwargs['greek'] == 'vega':
            vega_req = float(kwargs['greekval'])
            v1 = (op1.vega * 100) / (op1.lots * dm * lm)
            lots_req = round(abs(vega_req * 100) / (abs(v1) * lm * dm))
            print('lots req: ', lots_req)
            op1.update_lots(lots_req)
            op2.update_lots(lots_req)
            op1.underlying.update_lots(lots_req)
            op2.underlying.update_lots(lots_req)

        elif kwargs['greek'] == 'gamma':
            gamma_req = float(kwargs['greekval'])
            g1 = (op1.gamma * dm) / (op1.lots * lm)
            # g2 = (op2.gamma * dm) / (op2.lots * lm)
            lots_req = round((abs(gamma_req) * dm) / (abs(g1) * lm))
            op1.update_lots(lots_req)
            op2.update_lots(lots_req)
            op1.underlying.update_lots(lots_req)
            op2.underlying.update_lots(lots_req)

        elif kwargs['greek'] == 'theta':
            theta_req = float(kwargs['greekval'])
            print('theta req: ', theta_req)
            t1 = (op1.theta * 365)/(op1.lots * lm * dm)
            # t2 = (op2.theta * 365) / (op2.lots * lm * dm)
            lots_req = round((abs(theta_req) * 365) / (abs(t1) * lm * dm))
            print('lots req: ', lots_req)
            op1.update_lots(lots_req)
            print('op1.theta: ', op1.greeks()[2])
            op2.update_lots(lots_req)
            print('op2.theta: ', op2.greeks()[2])
            op1.underlying.update_lots(lots_req)
            op2.underlying.update_lots(lots_req)

    ops = [op1, op2]
    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)

    for op in ops:
        print('util.create_skew - strike, char, short: ',
              op.K, op.char, op.shorted)

    return ops


def create_straddle(volid, vdf, pdf, date, shorted, strike, pf=None, **kwargs):
    """Utility function that creates straddle given dataframes and arguments.

    Args:
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        strike (float): Strike of this straddle
        kwargs (dict): dictionary of parameters. contains info regarding greek value of straddle. 
        pf (portfolio, optional): Portfolio object. Used to determine lot-sizes 
                                    of hedges on the fly. 

    Returns:
        Tuple: Two options constituting a straddle

    Deleted Parameters:
        ft (Future object): Future object underlying this straddle
    """

    assert not vdf.empty
    assert not pdf.empty

    pdt = volid.split()[0]
    char1, char2 = 'call', 'put'
    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    lots = kwargs['lots'] if 'lots' in kwargs else None

    tau, vol = None, None

    if 'vol' in kwargs:
        vol = kwargs['vol']
    if 'tau' in kwargs:
        tau = kwargs['tau']

    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date=date, strike=strike, lots=lots, vol=vol, tau=tau)

    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, shorted, date=date, strike=strike, lots=lots, vol=vol, tau=tau)

    lots_req = op1.lots

    if 'greek' in kwargs:
        # determine number of lots based on greek and greekvalue
        if kwargs['greek'] == 'vega':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                vega_req = -pf.net_vega_pos()
            else:
                vega_req = float(kwargs['greekval'])
            # print('util.create_straddle - vega req: ', vega_req)
            v1 = (op1.vega * 100) / (op1.lots * dm * lm)
            v2 = (op2.vega * 100) / (op2.lots * dm * lm)
            lots_req = round((abs(vega_req) * 100) / (abs(v1 + v2) * lm * dm))

        elif kwargs['greek'] == 'gamma':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                gamma_req = -pf.net_gamma_pos()
            else:
                gamma_req = float(kwargs['greekval'])
            # print('util.create_straddle - gamma req: ', gamma_req)
            g1 = (op1.gamma * dm) / (op1.lots * lm)
            g2 = (op2.gamma * dm) / (op2.lots * lm)
            lots_req = round((abs(gamma_req) * dm) / (abs(g1 + g2) * lm))

        elif kwargs['greek'] == 'theta':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                theta_req = -pf.net_theta_pos()
            else:
                theta_req = float(kwargs['greekval'])
            # print('util.create_straddle - theta req: ', theta_req)
            t1 = (op1.theta * 365) / (op1.lots * lm * dm)
            t2 = (op2.theta * 365) / (op2.lots * lm * dm)
            lots_req = round((abs(theta_req) * 365) / (abs(t1 + t2) * lm * dm))

        # print('util.create_straddle - lots_req: ', lots_req)
        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.get_underlying().update_lots(lots_req)
        op2.get_underlying().update_lots(lots_req)

    ops = [op1, op2]

    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)

    assert len(ops) == 2
    return ops


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def close_out_deltas(pf, dtc):
    """Checks to see if the portfolio is emtpty but with residual deltas. 
    Closes out all remaining future positions, resulting in an empty portfolio.

    Args:
        pf (portfolio object): The portfolio being handles
        dtc (TYPE): deltas to close; tuple of (pdt, month, price)
        Returns
            tuple: updated portfolio, and cost of closing out deltas.
                 money is spent to close short pos, and gained by selling long pos.
    """
    # print('simulation.closing out deltas')
    cost = 0
    toberemoved = {}
    print('simulation.close_out_deltas - dtc: ', dtc)
    for pdt, mth, price in dtc:
        # print(pdt, mth, price)
        all_fts = pf.get_pos_futures()
        futures = [x for x in all_fts if (x.get_product() == pdt and
                                          x.get_month() == mth)]
        for ft in futures:
            flag = 'hedge' if ft in pf.hedge_futures else 'OTC'
            if flag not in toberemoved:
                toberemoved[flag] = []

            # need to spend to buy back
            val = price if ft.shorted else -price
            cost += val
            # cost += val * ft.lots * multipliers[pdt][-1]
            toberemoved[flag].append(ft)

    # print('close_out_deltas - toberemoved: ',
    #       [str(sec) for sec in toberemoved])
    for flag in toberemoved:
        pf.remove_security(toberemoved[flag], flag)
    print('cost of closing out deltas: ', cost)

    # print('pf after closing out deltas: ', pf)
    return pf, cost


def create_composites(lst):
    """Helper method that assigns to every option in list every other option as a 'partner'.

    Args:
        lst (List): List of options. 

    """
    for op in lst:
        tmp = lst.copy()
        tmp.remove(op)
        op.set_partners(set(tmp))
    return lst


def combine_portfolios(lst, hedges=None, name=None, refresh=False):
    """Helper method that merges and returns a portfolio pf where pf.families = lst. 

    Args:
        lst (TYPE): Description
    """

    pf = Portfolio(None) if hedges is None else Portfolio(hedges, name)
    for p in lst:
        # print('p: ', p)
        # update the lists first.
        pf.OTC_options.extend(p.OTC_options)
        pf.hedge_options.extend(p.hedge_options)
        # print('--------- merging OTCs ---------')
        pf.OTC = merge_dicts(p.OTC, pf.OTC)
        # print('--------- merging hedges ----------')
        pf.hedges = merge_dicts(p.hedges, pf.hedges)
        # print('---------- next loop ----------')
    pf.set_families(lst)

    if refresh:
        pf.refresh()

    return pf


def transfer_dict(d1):
    """ Helper method that recreates a dictionary while
    maintaining the memory location of all objects in the dictionary. 

    Args:
        d1 (TYPE): Description
    """
    values = []
    keys = d1.keys()
    for key in keys:
        data = d1[key]
        if isinstance(data, list):
            newlst = []
            for entry in data:
                if isinstance(entry, (int, float)):
                    val = entry
                else:
                    val = set()
                    val.update(entry)
                    # val = entry.copy()
                newlst.append(val)
            values = newlst
        elif isinstance(data, dict):
            newdic = transfer_dict(data)
            values = newdic

    ret = dict.fromkeys(keys, values)
    # print('transfer_dict.ret: ', ret)
    return ret


def merge_lists(r1, r2):
    """Helper method that merges two lists, of the form contained in 
    the OTC[pdt][month] for the OTC/hedge dictionaries in a portfolio object.
    Merges and returns the two lists such that modifying the result does NOT
    modify the constituent lists

    Args:
        l1 (TYPE): Description
        l2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    # print('merge lists triggered')
    l1 = copy.copy(r1)
    l2 = copy.copy(r2)
    if len(l1) == 0:
        # print('basecase: l1 is empty: ', l2)
        return transfer_lists(l2)
    elif len(l2) == 0:
        # print('basecase: l2 is empty: ', l1)
        return transfer_lists(l1)

    ret = []
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        dat1, dat2 = l1[i], l2[i]
        if isinstance(dat1, set):
            # print('merge_lists: set case')
            c = set()
            for i in dat1:
                c.add(i)
            for j in dat2:
                c.add(j)
        elif isinstance(dat1, (int, float)):
            c = dat1 + dat2
        ret.append(c)
    return ret


def merge_dicts(d1, d2):
    """Helper method that merges the OTC or hedges dictionaries of
    portfolio objects such that modifying the result does NOT modify 
    any of the constituent dictionaries. 

    Args:
        r1 (TYPE): First dictionary to be merged
        r2 (TYPE): Second dictionary to be merged.

    Returns:
        TYPE: Dictionary.
    # """
    ret = {}
    # base cases: either d1 or d2 are empty.
    if len(d1) == 0:
        # print('basecase: l1 is empty: ', d2)
        ret = transfer_dict(d2.copy())
    elif len(d2) == 0:
        # print('basecase: l2 is empty: ', d1)
        ret = transfer_dict(d1.copy())
    else:
        # handles d1-unique and d1-d2 overlap keys.
        for key in d1:
            # base case: key is not in d2 -> d1[key] becomes default value for
            # this key in ret.
            if key not in d2:
                # print(key + ' unique to d1')
                if isinstance(d1[key], dict):
                    ret[key] = transfer_dict(d1[key].copy())
                else:
                    # print('hit list case')
                    ret[key] = transfer_lists(d1[key].copy())
            # case: this key exists in d2. need to merge the outputs of d1[key]
            # and d2[key]
            else:
                # case 0: d1[key] and d2[key] are lists.
                if isinstance(d1[key], list) and isinstance(d2[key], list):
                    ret[key] = merge_lists(d1[key], d2[key])
                # case 1: d1[key] and d2[key] are dictionaries themselves
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    ret[key] = merge_dicts(d1[key], d2[key])

        for key in d2:
            if key not in d1:
                # print(key + ' unique to d2')
                if isinstance(d2[key], dict):
                    ret[key] = transfer_dict(d2[key].copy())
                else:
                    # print('hit list case')
                    ret[key] = transfer_lists(d2[key].copy())
            # other case would already have been handled in d1 iteration
            else:
                try:
                    assert key in ret
                except AssertionError as e:
                    raise AssertionError(
                        'key, current keys: ', key, ret.keys())

    return ret


def transfer_lists(l):
    """Helper method that passes references of a list to another, new list
    such that modifying the output does not modify the input 

    Args:
        l (TYPE): the list to be transferred. 

    Returns:
        TYPE: new list
    """
    lst = []
    for x in l:
        if isinstance(x, (float, int)):
            val = x
        elif isinstance(x, set):
            val = x.copy()
        lst.append(val)

    return lst


def volids_from_ci(date_range, product, ci):
    from itertools import cycle, dropwhile
    dates_to_volid = {}

    for date in date_range:
        mths = cycle(contract_mths[product])
        yr = date.year % 2010
        mth = month_to_sym[date.month]
        print('mth: ', mth)
        cy = dropwhile(lambda i: i < mth, mths)
        tar = next(cy)
        taryr = yr
        for i in range(ci):
            tar = next(cy)
        # case: wrapped around.
        if tar < mth:
            taryr += 1
        tarfin = tar + str(taryr)
        dates_to_volid[date] = product + '  ' + tarfin + '.' + tarfin
    return dates_to_volid


def assign_hedge_objects(pf, vdf=None, pdf=None, book=False):
    """Helper method that generates and relates a portfolio object 
    with the the Hedger object calibrated to pf.hedge_params. 

    Args:
        pf (TYPE): Portfolio object.
        vdf (dataframe, optional): dataframe of volatilities. 
        pdf (dataframe, optional): dataframe of prices 

    Returns:
        TYPE: Description

    """
    # case: simple portfolio.
    from .hedge import Hedge
    # print('assign_hedge_objects - init pf: ', pf)
    hedger = Hedge(pf, pf.hedge_params, vdf=vdf, pdf=pdf, book=book)
    # print('assign_hedge_objects - after creating hedge object: ', pf)
    pf.hedger = hedger
    # print('assign_hedge_objects - after assigning hedger: ', pf)
    pf.hedger.update_hedgepoints()
    # print('assign_hedge_objects - pf after updating hedgepoints: ', pf)

    # case: composite portfolio
    if pf.families:
        for fa in pf.families:
            hedger = Hedge(fa, fa.hedge_params, vdf=vdf, pdf=pdf, book=book)
            fa.hedger = hedger
            fa.hedger.update_hedgepoints()

    if vdf is not None and pdf is not None:
        pf.assign_hedger_dataframes(vdf, pdf)

    # print('assign_hedge_objects - pf after hedger dataframes: ', pf)
    return pf


def mark_to_vols(pfx, vdf, dup=False, pdt=None):
    """Helper method that computes the value of a portfolio marked to the 
    specified vols

    Args:
        pfx (Portfolio): the portfolio being marked. 
        vdf (TYPE): volatility surface the portfolio is being marked to. 
        dup (bool, optional): flag that indicates whether or not this method 
        should modify the portfolio passed in (if marking is desired) or 
        return a duplicate. 

    Returns:
        TYPE: Portfolio object with the updated vols. 
    """
    assert not vdf.empty
    pf = copy.deepcopy(pfx) if dup else pfx

    if pdt is None:
        ops = pf.get_all_options()
    else:
        ops = [op for op in pf.get_all_options() if op.get_product() == pdt]
    # print('vdf: ', vdf)

    for op in ops:
        ticksize = multipliers[op.get_product()][-2]
        vid = op.get_vol_id()
        cpi = 'C' if op.char == 'call' else 'P'
        strike = round(round(op.K / ticksize) * ticksize, 2)
        try:
            vol = vdf[(vdf.vol_id == vid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike)].vol.values[0]
        except IndexError as e:
            print("scripts.util.mark_to_vols: cannot find vol with inputs ",
                  vid, cpi, op.K, strike)
            print('scripts.mark_to_vols: debug1 = ', vdf[(vdf.vol_id == vid)])
            print('scripts.mark_to_vols: debug2 = ', vdf[(vdf.vol_id == vid) &
                                                         (vdf.call_put_id == cpi)])
            print('scripts.mark_to_vols: debug3 = ', vdf[(vdf.vol_id == vid) &
                                                         (vdf.strike == strike)])
            vol = op.vol
        op.update_greeks(vol=vol)

    pf.refresh()
    return pf


def compute_market_minus(pf, vdf):
    """Helper method that takes the difference of the portfolio's 
    current value and the value as per vols specified in vdf. 

    Args:
        pf (TYPE): The portfolio (assumed to be basis book vols)
        vdf (TYPE): The settlement vols of that particular day. 
    """
    # update the vols

    newpf = copy.deepcopy(pf)
    for op in newpf.get_all_options():
        ticksize = multipliers[op.get_product()][-2]
        vid = op.get_vol_id()
        cpi = 'C' if op.char == 'call' else 'P'
        strike = round(round(op.K / ticksize) * ticksize, 2)
        try:
            vol = vdf[(vdf.vol_id == vid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike)].vol.values[0]
            op.update_greeks(vol=vol)

        except IndexError:
            print('scripts.calc.market_minus - data not found. ',
                  vid, cpi, op.K, strike)
            print('date: ', vdf.value_date.unique())
    val = pf.compute_value() - newpf.compute_value()
    mm = abs(val)
    return mm, val


def handle_date_by_pdt(pdt, date, edf):
    """Helper method that decrements the date by 1 if a given product has an overnight market. 

    Args:
        pdt (TYPE): the product being handled
        date (TYPE): the date in question
        edf (dataframe): Dataframe of exchange timings

    """
    pass


def handle_timezones(lst, edf):
    """Helper function that reconciles timezones between the products specified. 

    Args:
        lst (TYPE): list of products we're interested in. 
        edf (dataframe): Dataframe of exchange timings
    """
    pass
