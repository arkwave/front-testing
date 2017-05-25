# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-05-19 20:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-05-25 17:18:05


from .portfolio import Portfolio
from .classes import Future, Option
from .prep_data import find_cdist, match_to_signals, get_min_start_date, clean_data, ciprice, civols, vol_by_delta
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import time
from .calc import compute_strike_from_delta


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
    """Helper function that generates common portfolio types. Delegates construction to specific 
     constructors, based on the input passed in. 

    Args:
        pdt (TYPE): Product
        opmth (TYPE): Option month, e.g. K7
        ftmth (TYPE): Future month, e.g. K7
        optype (TYPE): structure name. valid inputs are call, put, callspread, putspread, fence, straddle, strangle, call_butterfly and put_butterfly
        vdf (TYPE): dataframe of vols
        pdf (TYPE): dataframe of prices
        **kwargs: dictionary of the form {'strikes':[], 'char':[], 'shorted':[], 'lots': [], 'greek': str, 'greekvals': [], 'atm': bool}

    Returns:
        Portfolio object: The portfolio being created.  
    """
    print('kwargs: ', kwargs)
    pf = Portfolio()

    # create the underlying future
    ticksize = multipliers[pdt][-2]
    date = max(vdf.value_date.min(), pdf.value_date.min())
    ft, ftprice = create_underlying(pdt, ftmth, pdf, date, shorted=False)

    # create the relevant options; get all relevant information
    volid = pdt + '  ' + opmth + '.' + ftmth
    shorted = kwargs['shorted']

    if 'atm' in kwargs and kwargs['atm']:
        strike = round(round(ftprice / ticksize) * ticksize, 2)

    if optype == 'straddle':
        op1, op2 = create_straddle(
            volid, vdf, pdf, ft, date, shorted, strike, kwargs)
        ops = [op1, op2]

    elif optype == 'skew':
        delta = kwargs['delta']
        op1, op2 = create_skew(volid, vdf, pdf, ft, date,
                               shorted, delta, kwargs)
        ops = [op1, op2]

    elif optype == 'vanilla':
        strike = kwargs['strike']
        char = kwargs['char']
        op1 = create_vanilla_option(vdf, pdf, ft, strike, volid,
                                    char, 'amer', shorted, opmth,
                                    date=date, kwargs=kwargs)
        ops = [op1]

    elif optype == 'spread':
        char = kwargs['char']
        op1, op2 = create_spread(
            char, volid, vdf, pdf, ft, date, shorted, strike, kwargs)
        pass

    elif optype == 'fence':
        delta = kwargs['delta']
        op1, op2 = create_skew(volid, vdf, pdf, ft, date,
                               not shorted, delta, kwargs)
        ops = [op1, op2]

    elif optype == 'strangle':
        strike1, strike2 = kwargs['strike']
        op1, op2 = create_strangle(volid, vdf, pdf, ft, date,
                                   shorted, strike1, strike2, kwargs)
        ops = [op1, op2]

    # do tomorrow.
    elif optype == 'butterfly':
        char = kwargs['char']
        strike1, strike2, strike3, strike4 = kwargs['strike']
        op1, op2, op3, op4 = create_butterfly(char, volid, pdf, ft, date, shorted,
                                              strike1, strike2, strike3, strike4, kwargs)
        ops = [op1, op2, op3, op4]

    pf.add_security(ops, 'OTC')

    if 'hedges' in kwargs:
        print('creating hedges')
        dic = kwargs['hedges']
        if dic['type'] == 'straddle':
            # identifying the essentials
            pdt, ftmth, opmth = dic['pdt'], dic['ftmth'], dic['opmth']
            volid = pdt + '  ' + opmth + '.' + ftmth
            shorted = dic['shorted']
            # create the underlying future object
            h_ft, h_price = create_underlying(
                pdt, ftmth, pdf, date, shorted=False)
            h_strike = round(round(h_price / ticksize) * ticksize, 2) \
                if dic['strike'] == 'atm' else dic['strike']
            # create_straddle(volid, vdf, pdf, ft, date, shorted, strike,
            # kwargs):
            h1, h2 = create_straddle(volid, vdf, pdf, h_ft, date,
                                     shorted, h_strike, dic, pf=pf)

        pf.add_security([h1, h2], 'hedge')

    return pf


def create_underlying(pdt, ftmth, pdf, date, shorted=False, lots=None):
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
    ft = Future(ftmth, ftprice, pdt, shorted=shorted, ordering=order)
    if lots is not None:
        ft.update_lots(lots)

    return ft, ftprice


def create_vanilla_option(vdf, pdf, volid, char, shorted, date, payoff='amer', lots=None, kwargs=None, delta=None, strike=None):
    """Utility method that creates an option from the info passed in. Each option is instantiated with its own future underlying object. 

    Args:
        vdf (dataframe): dataframe of volatilities
        pdf (dataframe): dataframe of prices and vols by delta
        volid (string): vol_id of the option
        char (string): call or put
        payoff (string): american or european payoff
        shorted (bool): True or False
        mth (string): month of the option, e.g. N7
        date (pd.Timestamp, optional): date to use when selecting vol
        lots (int, optional): number of lots 
        kwargs (dict, optional): dictionary containing extra delimiting factors, e.g. greek/greekvalue. 
        delta (float, optional): Delta of this option as an alternative to strike. 
        strike (float): strike of the option

    Returns:
        object: Option created according to the parameters passed in. 


    """
    # sanity checks
    if delta is None and strike is None:
        raise ValueError(
            'neither delta nor strike passed in; aborting construction.')

    lots_req = lots if lots is not None else 1000

    # naming conventions
    ftmth = volid.split('.')[1]
    pdt = volid.split()[0]
    opmth = volid.split()[1].split('.')[0]
    cpi = 'C' if char == 'call' else 'P'

    # create the underlying future
    ft_shorted = shorted if char == 'call' else not shorted
    ft, ftprice = create_underlying(pdt, ftmth, pdf, date,
                                    shorted=ft_shorted, lots=lots)
    # get tau
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
        print('util.create_vanilla_option - vol1 not found, inputs below: ')
        print('date: ', date)
        print('vol_id: ', volid)
        print('call_put_id: ', cpi)
        print('strike: ', strike)

    # if delta specified, compute strike appropriate to that delta
    if delta is not None:
        strike = compute_strike_from_delta(
            None, delta1=delta, vol=vol, s=ft.get_price(), char=char, pdt=ft.get_product())

    # specifying option with information gathered thus far.
    newop = Option(strike, tau, char, vol, ft, payoff, shorted,
                   opmth, lots=lots_req, ordering=ft.get_ordering())
    pdt = ft.get_product()
    if kwargs is not None and 'greek' in kwargs:
        lots_req = 0
        if kwargs['greek'] == 'theta':
            theta_req = float(kwargs['greekval'])
            print('theta req: ', theta_req)
            lm, dm = multipliers[pdt][1], multipliers[pdt][0]
            t1 = (newop.theta * 365)/(newop.lots * lm * dm)
            # t2 = (op2.theta * 365) / (op2.lots * lm * dm)
            lots_req = round(((theta_req) * 365) / (t1 * lm * dm))

        if kwargs['greek'] == 'vega':
            pass

        if kwargs['greek'] == 'gamma':
            pass

        newop.update_lots(lots_req)
        newop.underlying.update_lots(lots_req)
    return newop


def create_butterfly(char, volid, pdf, ft, date, shorted, strike1, strike2, strike3, strike4, kwargs):
    """Utility method that creates a butterfly position. 

    Args:
        char (TYPE): Description
        volid (TYPE): Description
        pdf (TYPE): Description
        ft (TYPE): Description
        date (TYPE): Description
        shorted (TYPE): Description
        strike1 (TYPE): Description
        strike2 (TYPE): Description
        strike3 (TYPE): Description
        strike4 (TYPE): Description
        kwargs (TYPE): Description
    """
    pass


def create_spread(char, volid, vdf, pdf, ft, date, shorted, kwargs, payoff='amer'):
    """Utility method that creates a callspread 

    Args:
        char (str): call or put, indicating if this is a callspread or ap utspread. 
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        ft (Future object): Future object underlying this straddle
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        kwargs (dict): dictionary of the form {'chars': ['call', 'put'], 'strike': [strike1, strike2], 'greek':(gamma theta or vega), 'greekval': the value used to determine lot size, 'lots': lottage if greek not specified.}
    """
    # identify if this spread is created with explicit strikes or with delta
    # values
    delta1, delta2, strike1, strike2 = None, None, None, None
    if 'delta' in kwargs:
        delta1, delta2 = kwargs['delta']
    elif 'strike' in kwargs:
        strike1, strike2 = kwargs['strike']
    pdt = volid.split()[0]
    char1, char2 = kwargs['chars']

    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    # long call spread : buy itm call, sell otm call.
    # short call spread: sell itm call, buy otm call
    # long put spread: buy otm put, sell itm put/
    # short put spread: sell otm put, buy itm put.
    # make sure inputs are organized in the appropriate order.

    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date, delta=delta1, strike=strike1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, not shorted, date, delta=delta2, strike=strike2)

    if 'greek' in kwargs:
        val = kwargs['greekval']
        if kwargs['greek'] == 'gamma':
            pass
        elif kwargs['greek'] == 'vega':
            pass
        elif kwargs['theta'] == 'theta':
            pass

    return op1, op2


def create_strangle(volid, vdf, pdf, ft, date, shorted, kwargs, pf=None):
    """Utility method that creates a strangle (long or short).

    Args:
        volid (string): vol_id of the straddle
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Dataframe of prices
        ft (Future object): Future object underlying this straddle
        date (pd Timestamp): start date of simulation
        shorted (bool): self-explanatory. 
        kwargs (dict): dictionary of the form {'chars': ['call', 'put'], 'strike': [strike1, strike2], 'greek':(gamma theta or vega), 'greekval': the value used to determine lot size, 'lots': lottage if greek not specified.}

    """
    pdt = volid.split()[0]
    char1, char2 = kwargs['chars']
    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    if 'strike' in kwargs:
        strike1, strike2 = kwargs['strike']

    lot1, lot2 = kwargs['lots'] if 'lots' in kwargs else None, None

    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date, strike=strike1, lots=lot1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, shorted, date, strike=strike2, lots=lot2)

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
            lots_req = round((abs(vega_req) * 100) / ((v1 + v2) * lm * dm))

        elif kwargs['greek'] == 'gamma':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                gamma_req = -pf.net_gamma_pos()
            else:
                gamma_req = float(kwargs['greekval'])
            print('gamma req: ', gamma_req)
            g1 = (op1.gamma * dm) / (op1.lots * lm)
            g2 = (op2.gamma * dm) / (op2.lots * lm)
            lots_req = round(((gamma_req) * dm) / ((g1 + g2) * lm))

        elif kwargs['greek'] == 'theta':
            if pf is not None and kwargs['greekval'] == 'portfolio':
                theta_req = -pf.net_theta_pos()
            else:
                theta_req = float(kwargs['greekval'])
            print('theta req: ', theta_req)
            t1 = (op1.theta * 365)/(op1.lots * lm * dm)
            t2 = (op2.theta * 365) / (op2.lots * lm * dm)
            lots_req = round(((theta_req) * 365) / ((t1 + t2) * lm * dm))

        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.underlying.update_lots(lots_req)
        op2.underlying.update_lots(lots_req)

    return op1, op2


def create_skew(volid, vdf, pdf, ft, date, shorted, delta, kwargs):
    """Utility function that creates a skew position given dataframes and arguments. 

    Args:
        volid (TYPE): vol_id, e.g. CT N7.N7
        vdf (TYPE): dataframe of strikewise vols
        pdf (TYPE): dataframe of prices and delta-wise vols
        ft (TYPE): underlying future object
        date (TYPE): date 
        shorted (TYPE): True or False
        delta (TYPE): delta value of this skew
        kwargs (TYPE): dictionary containing 1) greek and 2) greekvalue to specify lots. 
    """
    col = str(int(delta)) + 'd'
    delta = delta/100
    pdt = volid.split()[0]
    tau = vdf[(vdf.value_date == date) &
              (vdf.vol_id == volid)].tau.values[0]
    # isolate vols
    try:
        vol1 = pdf[(pdf.vol_id == volid) &
                   (pdf.value_date == date) &
                   (pdf.call_put_id == 'C')][col].values[0]
    except IndexError:
        print('util.create_skew - cannot find vol1')
        print('vol_id: ', volid)
        print('date: ', date)
        print('cpi: ', 'C')

    try:
        vol2 = pdf[(pdf.vol_id == volid) &
                   (pdf.value_date == date) &
                   (pdf.call_put_id == 'P')][col].values[0]

    except IndexError:
        print('util.create_skew - cannot find vol2')
        print('vol_id: ', volid)
        print('date: ', date)
        print('cpi: ', 'P')

    strike1 = compute_strike_from_delta(None, delta1=delta, vol=vol1, s=ft.get_price(),
                                        tau=tau, char='call', pdt=pdt)
    strike2 = compute_strike_from_delta(None,  delta1=delta, vol=vol2, s=ft.get_price(),
                                        tau=tau, char='put', pdt=pdt)

    # create_vanilla_option(vdf, pdf, ft, strike, lots, volid, char, payoff, shorted, mth, date=None)
    # creating the options
    op1 = create_vanilla_option(
        vdf, pdf, volid, 'call', shorted, date, strike=strike1)
    op2 = create_vanilla_option(
        vdf, pdf, volid, 'put', not shorted, date, strike=strike2)

    if kwargs['greek'] == 'vega':
        vega_req = float(kwargs['greekval'])
        if shorted:
            vega_req = -vega_req
        # curr_vega = op1.vega + op2.vega
        lm, dm = multipliers[pdt][1], multipliers[pdt][0]
        v1 = (op1.vega * 100) / (op1.lots * dm * lm)
        lots_req = round((vega_req * 100) / (v1 * lm * dm))
        print('lots req: ', lots_req)

        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.underlying.update_lots(lots_req)
        op2.underlying.update_lots(lots_req)
    return op1, op2


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
    pdt = volid.split()[0]
    char1, char2 = 'call', 'put'
    lm, dm = multipliers[pdt][1], multipliers[pdt][0]

    op1 = create_vanilla_option(
        vdf, pdf, volid, char1, shorted, date, strike=strike)

    op2 = create_vanilla_option(
        vdf, pdf, volid, char2, shorted, date, strike=strike)

    # determine number of lots based on greek and greekvalue
    if kwargs['greek'] == 'vega':
        if pf is not None and kwargs['greekval'] == 'portfolio':
            vega_req = -pf.net_theta_pos()
        else:
            vega_req = float(kwargs['greekval'])
        print('vega req: ', vega_req)
        v1 = (op1.vega * 100) / (op1.lots * dm * lm)
        v2 = (op2.vega * 100) / (op2.lots * dm * lm)
        lots_req = round((abs(vega_req) * 100) / ((v1 + v2) * lm * dm))

    elif kwargs['greek'] == 'gamma':
        if pf is not None and kwargs['greekval'] == 'portfolio':
            gamma_req = -pf.net_gamma_pos()
        else:
            gamma_req = float(kwargs['greekval'])
        print('gamma req: ', gamma_req)
        g1 = (op1.gamma * dm) / (op1.lots * lm)
        g2 = (op2.gamma * dm) / (op2.lots * lm)
        lots_req = round(((gamma_req) * dm) / ((g1 + g2) * lm))

    elif kwargs['greek'] == 'theta':
        if pf is not None and kwargs['greekval'] == 'portfolio':
            theta_req = -pf.net_theta_pos()
        else:
            theta_req = float(kwargs['greekval'])
        print('theta req: ', theta_req)
        t1 = (op1.theta * 365)/(op1.lots * lm * dm)
        t2 = (op2.theta * 365) / (op2.lots * lm * dm)
        lots_req = round(((theta_req) * 365) / ((t1 + t2) * lm * dm))

        op1.update_lots(lots_req)
        op2.update_lots(lots_req)
        op1.get_underlying().update_lots(lots_req)
        op2.get_underlying().update_lots(lots_req)

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

    return final_vol, final_price, edf, pdf, start_date


def pull_alt_data(pdt):
    """Utility function that draws/cleans data from the alternate data table. 

    Args:
        pdt (string): The product being drawn from the database

    Returns:
        tuple: vol dataframe, price dataframe, raw dataframe. 


    """
    print('starting clock..')
    t = time.clock()
    engine = create_engine(
        'postgresql://sumit:Olam1234@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    query = "select security_id, settlement_date, future_settlement_value, option_expiry_date,implied_vol \
            FROM table_option_settlement_data where security_id like '" + pdt.upper() + " %%' and extract(YEAR from settlement_date) > 2009"
    print('query: ', query)
    df = pd.read_sql_query(query, connection)

    df.to_csv('datasets/data_dump/' + pdt.lower() +
              '_raw_data.csv', index=False)

    print('finished pulling data')
    print('elapsed: ', time.clock() - t)

    add = 1 if len(pdt) == 2 else 0

    # cleans up data
    df['vol_id'] = df.security_id.str[:9+add]
    df['call_put_id'] = df.security_id.str[9+add:11+add].str.strip()
    # getting 'S' from 'S Q17.Q17'
    df['pdt'] = df.vol_id.str[0:len(pdt)]
    # df['pdt'] = df.vol_id.str.split().str[0].str.strip()

    # getting opmth
    df['opmth'] = df.vol_id.str.split(
        '.').str[0].str.split().str[1].str.strip()
    df.opmth = df.opmth.str[0] + \
        (df.opmth.str[1:].astype(int) % 10).astype(str)

    df['strike'] = df.security_id.str[10+add:].astype(float)

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

    vdf.to_csv('datasets/data_dump/' + pdt.lower() +
               '_vol_dump.csv', index=False)
    pdf.to_csv('datasets/data_dump/' + pdt.lower() +
               '_price_dump.csv', index=False)

    return vdf, pdf, df