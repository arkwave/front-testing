# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-05-19 20:56:16
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-21 16:49:00


from .portfolio import Portfolio
from .classes import Future, Option
from .prep_data import find_cdist, handle_dailies
import pandas as pd
import numpy as np
import os
from .calc import compute_strike_from_delta
import sys

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
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
    'QC': ['H', 'K', 'N', 'U', 'Z'],
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


def create_underlying(pdt, ftmth, pdf, date, ftprice=None, shorted=False, lots=None):
    """Utility method that creates the underlying future object given a product, month, price data and date. 

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
    uid = pdt + '  ' + ftmth
    if ftprice is None:
        try:
            ftprice = pdf[(pdf.underlying_id == uid) &
                          (pdf.value_date == date)].settle_value.values[0]
        except IndexError:
            print('util.create_underlying: cannot find price. printing outputs: ')
            print('uid: ', uid)
            print('date: ', date)
            return None, 0

    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])
    ft = Future(ftmth, ftprice, pdt, shorted=shorted, ordering=order)
    if lots is not None:
        ft.update_lots(lots)

    return ft, ftprice


def create_vanilla_option(vdf, pdf, volid, char, shorted, date=None, payoff='amer', lots=None, delta=None, strike=None, vol=None, bullet=True, **kwargs):
    """Utility method that creates an option from the info passed in. Each option is instantiated with its own future underlying object. 

    Args:
        vdf (dataframe): dataframe of volatilities
        pdf (dataframe): dataframe of prices and vols by delta
        volid (string): vol_id of the option
        char (string): call or put
        shorted (bool): True or False
        date (pd.Timestamp, optional): date to use when selecting vol
        payoff (string): american or european payoff
        lots (int, optional): number of lots 
        kwargs (dict, optional): dictionary containing extra delimiting factors, e.g. greek/greekvalue. 
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
    min_start_date = min(pdf[pdf.pdt == pdt].value_date)

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
    except IndexError:
        print('util.create_vanilla_option - cannot find tau.')
        print('inputs: ', date, volid)

    # Case 1 : Vol is None, but strike is specified.
    if vol is None and strike is not None:
        # get vol
        try:
            vol = vdf[(vdf.value_date == date) &
                      (vdf.vol_id == volid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike)].settle_vol.values[0]
        except IndexError:
            print('util.create_vanilla_option - vol1 not found, inputs below: ')
            print('date: ', date)
            print('vol_id: ', volid)
            print('call_put_id: ', cpi)
            print('strike: ', strike)

    # Case 2: Option construction is basis delta. vol and strike are None.
    elif vol is None and strike is None:
        try:
            col = str(int(delta)) + 'd'
            vol = pdf[(pdf.vol_id == volid) &
                      (pdf.value_date == date) &
                      (pdf.call_put_id == 'C')][col].values[0]
        except IndexError:
            print('util.create_vanilla_option - cannot find vol1')
            print('vol_id: ', volid)
            print('date: ', date)
            print('cpi: ', 'C')

    # if delta specified, compute strike appropriate to that delta
    if delta is not None:
        delta = delta/100
        strike = compute_strike_from_delta(
            None, delta1=delta, vol=vol, s=ft.get_price(), char=char, pdt=ft.get_product(), tau=tau)
        print('util.create_vanilla_option - strike: ', strike)

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
            # t2 = (op2.theta * 365) / (op2.lots * lm * dm)
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
    except IndexError:
        print('util.create_vanilla_option - cannot find tau.')
        print('inputs: ', date, volid)

    # Case 1 : Vol is None, but strike is specified.
    if vol is None and strike is not None:
        # get vol
        try:
            vol = vdf[(vdf.value_date == date) &
                      (vdf.vol_id == volid) &
                      (vdf.call_put_id == cpi) &
                      (vdf.strike == strike)].settle_vol.values[0]
        except IndexError:
            print('util.create_vanilla_option - vol1 not found, inputs below: ')
            print('date: ', date)
            print('vol_id: ', volid)
            print('call_put_id: ', cpi)
            print('strike: ', strike)

    # get barrier vol.
    barlevel = ko if ko is not None else ki
    if bvol is None:
        try:
            bvol = vdf[(vdf.value_date == date) &
                       (vdf.vol_id == volid) &
                       (vdf.call_put_id == cpi) &
                       (vdf.strike == barlevel)].settle_vol.values[0]
        except IndexError:
            print('util.create_barrier_option - bvol not found. inputs below: ')
            print('date: ', date)
            print('vol_id: ', volid)
            print('call_put_id: ', cpi)
            print('strike: ', barlevel)

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
    # create_vanilla_option(vdf, pdf, volid, char, shorted, date, payoff='amer', lots=None, kwargs=None, delta=None, strike=None, vol=None):
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
        kwargs (dict): dictionary of the form {'chars': ['call', 'put'], 'strike': [strike1, strike2], 'greek':(gamma theta or vega), 'greekval': the value used to determine lot size, 'lots': lottage if greek not specified.}

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
        kwargs (dict): dictionary of the form {'chars': ['call', 'put'], 'strike': [strike1, strike2], 'greek':(gamma theta or vega), 'greekval': the value used to determine lot size, 'lots': lottage if greek not specified.}
        pf (portfolio, optional): Portfolio object. 

    Deleted Parameters:
        ft (Future object): Future object underlying this straddle

    Returns:
        TYPE: Description

    """
    lot1, lot2 = None, None
    pdt = volid.split()[0]
    char1, char2 = kwargs['chars']
    delta1, delta2 = None, None

    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    strike1, strike2 = kwargs['strike'] if 'strike' in kwargs else None, None

    lot1, lot2 = kwargs['lots'] if 'lots' in kwargs else None, None

    if 'delta' in kwargs:
        delta1, delta2 = kwargs['delta']

    print('deltas: ', delta1, delta2)

    print('util.create_strangle - lot1, lot2: ', lot1, lot2)
    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date, strike=strike1, lots=lot1, delta=delta1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, shorted, date, strike=strike2, lots=lot2, delta=delta2)

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
    col = str(int(delta)) + 'd'
    pdt = volid.split()[0]

    # create_vanilla_option(vdf, pdf, ft, strike, lots, volid, char, payoff, shorted, mth, date=None)
    # creating the options
    op1 = create_vanilla_option(
        vdf, pdf, volid, 'call', shorted, date, delta=delta)
    op2 = create_vanilla_option(
        vdf, pdf, volid, 'put', not shorted, date, delta=delta)

    if kwargs['greek'] == 'vega':
        vega_req = float(kwargs['greekval'])
        lm, dm = multipliers[pdt][1], multipliers[pdt][0]
        v1 = (op1.vega * 100) / (op1.lots * dm * lm)
        lots_req = round(abs(vega_req * 100) / (abs(v1) * lm * dm))
        print('lots req: ', lots_req)
        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.underlying.update_lots(lots_req)
        op2.underlying.update_lots(lots_req)

    ops = [op1, op2]
    if ('composites' in kwargs and kwargs['composites']) or ('composites' not in kwargs):
        ops = create_composites(ops)

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
        pf (portfolio, optional): Portfolio object. Used to determine lot-sizes of hedges on the fly. 

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

    # create_vanilla_option(vdf, pdf, volid, char, shorted, date=None, payoff='amer', lots=None, delta=None, strike=None, vol=None, bullet=True, **kwargs)
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

    return ops


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# TODO: close out OTC futures as well
def close_out_deltas(pf, dtc):
    """Checks to see if the portfolio is emtpty but with residual deltas. Closes out all remaining future positions, resulting in an empty portfolio.

    Args:
        pf (portfolio object): The portfolio being handles
        dtc (TYPE): deltas to close; tuple of (pdt, month, price)
        Returns
            tuple: updated portfolio, and cost of closing out deltas. money is spent to close short pos, and gained by selling long pos.
    """
    # print('simulation.closing out deltas')
    cost = 0
    toberemoved = []
    print('simulation.close_out_deltas - dtc: ', dtc)
    for pdt, mth, price in dtc:
        # print(pdt, mth, price)
        futures = pf.hedges[pdt][mth][1]
        # print([str(ft) for ft in futures])
        # toberemoved = []
        for ft in futures:
            # need to spend to buy back
            val = price if ft.shorted else -price
            cost += val
            toberemoved.append(ft)

    # print('close_out_deltas - toberemoved: ',
    #       [str(sec) for sec in toberemoved])
    pf.remove_security(toberemoved, 'hedge')
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
        # update the lists first.
        pf.OTC_options.extend(p.OTC_options)
        pf.hedge_options.extend(p.hedge_options)
        pf.OTC = merge_dicts(p.OTC, pf.OTC)
        pf.hedges = merge_dicts(p.hedges, pf.hedges)

    pf.set_families(lst)

    if refresh:
        pf.refresh()

    return pf


def merge_dicts(d1, d2):
    ret = {}
    if len(d1) == 0:
        return d2
    elif len(d2) == 0:
        return d1
    else:
        for key in d1:
            if key not in ret:
                ret[key] = {}
            # base case: key is in d1 but not d2.
            if key not in d2:
                ret[key] = d1[key]
            elif key in d2:
                dat1, dat2 = d1[key], d2[key]
                # case 1: nested dictionary.
                if isinstance(dat1, dict):
                    nd = merge_dicts(dat1, dat2)
                    ret[key] = nd
                # case 2: list.
                elif isinstance(dat1, list):
                    nl = merge_lists(dat1, dat2)
                    ret[key] = nl

        for key in d2:
            if key not in ret:
                ret[key] = {}
            # base case: key is in d1 but not d2.
            if key not in d1:
                ret[key] = d2[key]
            elif key in d1:
                dat1, dat2 = d1[key], d2[key]
                # case 1: nested dictionary.
                if isinstance(dat2, dict):
                    nd = merge_dicts(dat1, dat2)
                    ret[key] = nd
                # case 2: list.
                elif isinstance(dat2, list):
                    nl = merge_lists(dat1, dat2)
                    ret[key] = nl
    return ret


def merge_lists(l1, l2):
    """Helper method that merges two lists. 

    Args:
        l1 (TYPE): Description
        l2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    if len(l1) == 0:
        return l2
    elif len(l2) == 0:
        return l1

    ret = []
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        dat1, dat2 = l1[i], l2[i]
        if isinstance(dat1, set):
            c = dat1.copy()
            c.update(dat2)
        elif isinstance(dat1, (int, float)):
            c = dat1 + dat2
        ret.append(c)
    return ret


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
