"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 3/4/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""
# TODO: Update documentation.

###########################################################
############### Imports/Global Variables ##################
###########################################################

from .portfolio import Portfolio
from .classes import Option, Future
from .calc import get_barrier_vol
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats import norm
from math import log, sqrt, ceil
import time
from ast import literal_eval
from collections import OrderedDict
import copy
import datetime as dt
import pprint

seed = 7
np.random.seed(seed)

# setting pandas warning levels
pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.
# filepath = 'portfolio_specs.txt'

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
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


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
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}


###############################################################
################## Data Read-in Functions #####################
###############################################################

def match_to_signals(vdf, pdf, signals):
    """Lines up the value_date convention between vdf, pdf and signals. Signals are reported at 
       the start of the day, whereas vdf/pdf report EOD values. Additionally, settlement values 
       for fridays are currently reported on the following Sunday. This function adjusts for that 
       discrepancy and filters out all days in vdf/pdf that are not in signals.

    Args:
        vdf (pandas df): dataframe of vols
        pdf (pandas df): dataframe of prices
        signals (pandas df): dataframe of signals

    Returns:
        tuple: vol and price dataframes with prices updated/removed accordingly.
    """
    # vdf.value_date -= pd.Timedelta('2 day')
    # pdf.value_date -= pd.Timedelta('2 day')

    # v_dates = pd.to_datetime(vdf.value_date.unique())
    # p_dates = pd.to_datetime(pdf.value_date.unique())

    # print('v_check 1: ', pd.Timestamp('2017-02-20') in v_dates)
    # print('v_check 2: ', pd.Timestamp('2017-01-16') in v_dates)

    vdf['monday'] = vdf.value_date.dt.weekday == 0
    # pdf['monday'] = pdf.value_date.dt.weekday == 0

    vdf.loc[vdf.monday == True, 'value_date'] -= pd.Timedelta('3 day')
    # pdf.loc[pdf.monday == True, 'value_date'] -= pd.Timedelta('3 day')
    vdf.loc[vdf.monday == False, 'value_date'] -= pd.Timedelta('1 day')
    # pdf.loc[pdf.monday == False, 'value_date'] -= pd.Timedelta('1 day')

    # vdf.value_date -= pd.Timedelta('1 day')

    # filtering relevant dates
    # vdf = vdf[(vdf.value_date > pd.Timestamp('2017-01-02')) &
    #           (vdf.value_date < pd.Timestamp('2017-04-02'))]
    # pdf = pdf[(pdf.value_date > pd.Timestamp('2017-01-02')) &
    #           (pdf.value_date < pd.Timestamp('2017-04-02'))]

    d1 = [x for x in vdf.value_date.unique()
          if x not in signals.value_date.unique()]
    d1 = pd.to_datetime(d1)

    d2 = [x for x in pdf.value_date.unique()
          if x not in signals.value_date.unique()]
    d2 = pd.to_datetime(d2)

    d3 = [x for x in signals.value_date.unique()
          if x not in vdf.value_date.unique()]
    d3 = pd.to_datetime(d3)

    d4 = [x for x in signals.value_date.unique()
          if x not in pdf.value_date.unique()]
    d4 = pd.to_datetime(d4)

    # mask = signals.value_date.isin(d3)
    vmask = vdf.value_date.isin(d1)
    pmask = pdf.value_date.isin(d2)

    vdf = vdf[~vmask]
    pdf = pdf[~pmask]

    return vdf, pdf


def generate_hedges(filepath):
    """Generates an Ordered Dictionary detailing the hedging logic, read in from the filepath specified.

    Args:
        filepath (string): path to the csv file specifying hedging logic.

    Returns:
        OrderedDictionary: Dictionary of hedges with the format:
            {greek: [[condition_1], [condition_2]]}
        where condition_i is a list specifying the details of the conditions imposed on that greek.
    """
    vdf = pd.read_csv(filepath)

    lst = []

    roll_portfolio, pf_ttm_tol, pf_roll_product = None, None, None
    roll_hedges, h_ttm_tol, h_roll_product = None, None, None

    all_hedges = {}

    for family in vdf.family.unique():
        if family not in all_hedges:
            all_hedges[family] = OrderedDict()
        df = vdf[vdf.family == family]
        df.reset_index(drop=True, inplace=True)
        hedges = all_hedges[family]
        for i in df.index:
            row = df.iloc[i]
            # static hedging
            greek = row.greek
            # initial check: ascertain whether or not general rolling conditions
            # have been included.
            if greek == 'gen':
                if row.flag == 'portfolio':
                    roll_portfolio = True if row.cond == 'TRUE' else False
                    pf_ttm_tol = row.tau if not np.isnan(row.tau) else 30
                    pf_roll_product = row.spec if row.spectype == 'product' else None

                elif row.flag == 'hedges':
                    roll_hedges = True if row.cond else False
                    h_ttm_tol = row.tau if not np.isnan(row.tau) else 30
                    h_roll_product = row.spec if row.spectype == 'product' else None
                continue

            if row.flag == 'static':
                cond = str(row.cond) if row.cond == 'zero' else int(row.cond)
                lst = [row.flag, cond, int(row.freq)]

            # bound hedging
            elif row.flag == 'bound':
                spec = str(row.spec) if row.spec == 'atm' else float(
                    row.spec)
                spectype = str(row.spectype)
                if greek in ['gamma', 'theta', 'vega']:
                    rep = row.repr
                    subcond = literal_eval(
                        row.subcond) if rep == 'exp' else None
                    if not np.isnan(row.tau):
                        print('prep_data.generate_hedges - tau not nan. ')
                        tau = float(row.tau)
                        lst = [row.flag, list(literal_eval(row.cond)), int(row.freq), tau,
                               row.tau_spec, row.kind, spectype, spec]
                        if subcond is not None:
                            lst.extend([subcond, row.repr])
                        else:
                            lst.append(row.repr)

                    else:
                        lst = [row.flag, literal_eval(row.cond), int(row.freq),
                               row.kind, spectype, spec, row.repr]

            # percentage hedging
            elif row.flag == 'pct':
                # greek = row.greek
                lst = [row.flag, float(row.cond), int(row.freq), row.subcond]
            elif row.flag == 'roll':
                # greek = row.greek
                lst = [row.flag, float(row.cond), int(
                    row.freq), literal_eval(row.subcond)]
            # append to the dictionary
            if greek in hedges:
                hedges[greek].append(lst)
            else:
                hedges[greek] = [lst]
        all_hedges[family] = hedges

    return all_hedges, roll_portfolio, pf_ttm_tol, \
        pf_roll_product,  roll_hedges, h_ttm_tol, h_roll_product


def prep_portfolio(voldata, pricedata, filepath=None, spec=None):
    """Constructs the portfolio from the requisite CSV file that specifies the details of
    each security in the portfolio.

    Args:
        voldata (TYPE): volatility data
        pricedata (TYPE): price data
        filepath (TYPE): path to the csv containing portfolio specifications

    Returns:
        TYPE: portfolio object

    Raises:
        ValueError: Description

    """

    # initializing variables
    sim_start = None
    oplist = {'hedge': [], 'OTC': []}
    ftlist = {'hedge': [], 'OTC': []}
    # sim_start = min(min(voldata.value_date), min(pricedata.value_date))
    # reading in the dataframe of portfolio specifications
    specs = pd.read_csv(filepath) if spec is None else spec
    specs = specs.fillna('None')

    if specs.empty:
        return Portfolio(), None

    pf_ids = specs[specs.Type == 'Option'].vol_id.unique()
    print('pf_ids: ', pf_ids)

    sim_start = get_min_start_date(voldata, pricedata, pf_ids)
    print('prep_portfolio start_date: ', sim_start)

    # except ValueError:
    # print('[scripts/prep_data.prep_portfolio] There are vol_ids in this
    # portfolio with no corresponding data in the datasets.')
    sim_start = pd.to_datetime(sim_start)

    t = time.time()
    pf = Portfolio(None)

    curr_mth = sim_start.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = sim_start.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)

    print('SIM START: ', sim_start)
    # constructing each object individually
    for i in range(len(specs)):
        data = specs.iloc[i]
        if data.Type == 'Future':
            # future case
            full = data.vol_id.split()
            # uid = data.vol_id
            product = full[0]
            mth = full[1]
            lst = contract_mths[product]
            ordering = find_cdist(curr_sym, mth, lst)
            # price = pricedata[(pricedata.order == ordering) &
            #                   (pricedata.value_date == sim_start)]['price'].values[0]
            print('volid: ', data.vol_id)
            price = pricedata[(pricedata['underlying_id'] == data.vol_id) &
                              (pricedata['value_date'] == sim_start)]['price'].values[0]
            flag = data.hedgeorOTC
            lots = 1000 if data.lots == 'None' else int(data.lots)
            shorted = True if data.shorted else False
            ft = Future(mth, price, product, shorted=shorted,
                        lots=lots, ordering=ordering)
            ftlist[flag].append(ft)

        elif data.Type == 'Option':

            # basic option info
            volid = str(data.vol_id)
            opmth = volid.split()[1].split('.')[0]
            char = str(data.call_put_id)
            volflag = 'C' if char == 'call' else 'P'

            # handle underlying construction
            f_mth = volid.split()[1].split('.')[1]
            f_name = volid.split()[0]
            mths = contract_mths[f_name]
            ordering = find_cdist(curr_sym, f_mth, mths)
            u_name = f_name + '  ' + volid.split('.')[1]

            try:
                f_price = pricedata[(pricedata['value_date'] == sim_start) &
                                    (pricedata['underlying_id'] == u_name)]['price'].values[0]
            except IndexError:
                print('vol_id: ', volid)
                print('f_name: ', f_name)
                print('value_date: ', sim_start)
                print('underlying_id: ', u_name)

            # lots
            lots = 1000 if data.lots == 'None' else int(data.lots)
            underlying = Future(f_mth, f_price, f_name,
                                ordering=ordering, lots=lots)
            ticksize = multipliers[f_name][-2]

            strike = round(round(f_price / ticksize) * ticksize,
                           2) if data.strike == 'atm' else float(data.strike)

            # get tau from data+
            try:
                tau = voldata[(voldata['value_date'] == sim_start) &
                              (voldata['vol_id'] == volid) &
                              (voldata['call_put_id'] == volflag)]['tau'].values[0]
            except IndexError:
                print('vol_id: ', volid)
                print('call_put_id: ', volflag)
                print('value_date: ', sim_start)
                print('strike: ', strike)
                raise ValueError(
                    'tau cannot be located! Inputs: ', volid, volflag, sim_start, strike)
            # get vol from data
            try:
                vol = voldata[(voldata['vol_id'] == volid) &
                              (voldata['call_put_id'] == volflag) &
                              (voldata['value_date'] == sim_start) &
                              (voldata['strike'] == strike)]['vol'].values[0]
            except IndexError:
                print('vol_id: ', volid)
                print('call_put_id: ', volflag)
                print('value_date: ', sim_start)
                print('strike: ', strike)
                print('vol cannot be located! interpolating...')
                df = voldata[(voldata['vol_id'] == volid) &
                             (voldata['call_put_id'] == volflag) &
                             (voldata['value_date'] == sim_start)]
                df.sort_values(by='strike', inplace=True)
                f1 = interp1d(df.strike, df.vol,
                              fill_value='extrapolate')
                vol = f1(strike)
                # raise ValueError('vol cannot be located!',
                #                  volid, volflag, sim_start, strike)
            # american vs european payoff
            payoff = str(data.optiontype)
            # american or european barrier.

            barriertype = None if data.barriertype == 'None' else str(
                data.barriertype)
            # direction of barrier.
            direc = None if data.direction == 'None' else str(data.direction)
            # knock-in. is not None iff this is a knock-in option.
            ki = None if data.knockin == 'None' else float(data.knockin)
            # knock-out. is not None iff this is a knock-out option.
            ko = None if data.knockout == 'None' else float(data.knockout)
            # bullet vs daily pay-out. defaults to False.
            bullet = True if data.bullet else False
            # hedge or OTC
            flag = str(data.hedgeorOTC)
            # short or long position on this option.
            shorted = True if data.shorted else False

            bvol = None

            # handling european barrier case: get/assign barrier vol
            if barriertype is not None:
                print('prep_portfolio - getting barrier vol')
                barlevel = ki if ki is not None else ko
                bvoldata = voldata[
                    (voldata.value_date == pd.to_datetime(sim_start))]

                bvol = get_barrier_vol(
                    bvoldata, f_name, tau, volflag, barlevel, ordering)

            opt = Option(strike, tau, char, vol, underlying,
                         payoff, shorted=shorted, month=opmth, direc=direc,
                         barrier=barriertype, lots=lots, bullet=bullet,
                         ki=ki, ko=ko, ordering=ordering, bvol=bvol)

            oplist[flag].append(opt)

    # handling bullet options
    bullets = handle_dailies(oplist, sim_start)
    for flag in bullets:
        ops = oplist[flag]
        pf.add_security(ops, flag)

    for flag in ftlist:
        fts = ftlist[flag]
        pf.add_security(fts, flag)

    elapsed = time.time() - t
    print('[PREP_PORTFOLIO] elapsed: ', elapsed)
    return pf, sim_start


def handle_dailies(dic, sim_start):
    """Summary

    Args:
        dic (TYPE): dictionary of the form {'OTC': [list of options], 'hedge':[list of options]}
        sim_start (TYPE): start_date of the simulation

    Returns:
        TYPE: Description
    """
    for flag in dic:
        lst = dic[flag]
        tmp = lst.copy()
        for op in tmp:
            bullets = []
            # daily option
            if not op.bullet:
                # getting parameters of the daily option.
                params = op.get_properties()
                lst.remove(op)
                ttm_range = round(op.tau * 365)
                expdate = sim_start + pd.Timedelta(str(ttm_range) + ' days')
                daterange = pd.bdate_range(sim_start, expdate)
                print('daterange: ', daterange)
                taus = [((expdate - b_day).days) /
                        365 for b_day in daterange if b_day != expdate]
                strike, char, vol, underlying, payoff, shorted, month, ordering, lots, settlement \
                    = params['strike'], params['char'], params['vol'], params['underlying'], \
                    params['payoff'],  params['shorted'], params['month'], \
                    params['ordering'], params['lots'],\
                    params['settlement']
                # barrier params
                direc, barrier, ki, ko, rebate, bvol = \
                    params['direc'], params['barrier'], params[
                        'ki'], params['ko'], params['rebate'], params['bvol']

                # creating the bullets corresponding to this daily option.
                for tau in taus:
                    ui = copy.deepcopy(underlying)
                    op_i = Option(strike, tau, char, vol, ui, payoff, shorted, month, direc=direc,
                                  barrier=barrier, lots=lots, bullet=False, ki=ki, ko=ko, rebate=rebate,
                                  ordering=ordering, settlement=settlement, bvol=bvol)
                    bullets.append(op_i)

            lst.extend(bullets)

    return dic


###############################################################
################### Data Cleaning Functions ###################
###############################################################


def clean_data(df, flag, date=None, edf=None, writeflag=None):
    """Function that cleans the dataframes passed into it according to the flag passed in.
    1) flag == 'exp':
        > datatype conversion to pd.Timestamp
        > filters for data > 2010.
        > converts formatting; i.e. C H17 --> C H7
    2) flag == 'vol':
        > convert date strings to pd.Timestamp
        > calculate time to maturity from vol_id (i.e. C Z7.Z7 --> TTM in years)
        > appends expiry date
        > generates additional fields from existing ones.
        > assigns preliminary ordering (i.e. c1, c2 months from current month). 
          This step involves another function civols (line 479)
        > computes all-purpose label comprising of vol_id, order, and call_put_id
            - example: C Z7.Z7 4 C --> Corn Z7.Z7 call option with ordering 4
            - example: C Z7.Z7 4 P --> Corn Z7.Z7 put option with ordering 4.
    3) flag == 'price':
        > expiry date and ordering like in vol.
        > date strings to pd.Timestamp
        > calculates returns; log(S_curr/S_prev)
        > calculates orderings with rollover, using function ciprice
    Args:
        df (pandas dataframe)   : the dataframe to be cleaned.
        flag (pandas dataframe) : determines which dataframe is being processed.
        edf (pandas dataframe)  : dataframe containing the expiries of options.


    Returns:
        TYPE: the cleaned dataframe, with the appropriate data transformations made.
    """
    assert not df.empty
    # cleaning expiry data
    if flag == 'exp':
        # cleaning expiry data, handling datatypes
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        df = df[(df['year'] >= 10)]

    # cleaning volatility data
    elif flag == 'vol':
        print('cleaning voldata')
        # handling data types
        df['value_date'] = pd.to_datetime(df['value_date'])
        df = df.dropna()
        assert not df.empty
        # calculating time to expiry from vol_id
        df = ttm(df, df['vol_id'], edf)
        df = df[df.tau > 0].dropna()
        assert not df.empty
        # generating additional identifying fields.
        df['underlying_id'] = df['vol_id'].str.split().str[0] + '  ' + \
            df['vol_id'].str.split('.').str[1]
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['ftmth'] = df['underlying_id'].str.split().str[1]
        df['op_id'] = df['op_id'] = df.vol_id.str.split().str[
            1].str.split('.').str[0]

        # setting data types
        # df.order = pd.to_numeric(df.order)
        print('df.columns: ', df.columns)
        df.tau = pd.to_numeric(df.tau)
        df.strike = pd.to_numeric(df.strike)
        df.vol = pd.to_numeric(df.vol)
        df.value_date = pd.to_datetime(df.value_date)

        df['time'] = dt.time.max
        df['time'] = df['time'].astype(pd.Timestamp)
        df['datatype'] = 'settlement'

    # cleaning price data
    elif flag == 'price':
        print('cleaning pricedata')
        # dealing with datatypes and generating new fields from existing ones.
        df['value_date'] = pd.to_datetime(df['value_date'])
        if 'pdt' not in df.columns:
            df['pdt'] = df['underlying_id'].str.split().str[0]
        if 'ftmth' not in df.columns:
            df['ftmth'] = df['underlying_id'].str.split().str[1]
        # transformative functions.
        # df = get_expiry(df, edf)
        # df = assign_ci(df, date)
        # df['order'] = ''
        # df = scale_prices(df)
        df = df.fillna(0)
        df.value_date = pd.to_datetime(df.value_date)
        df.price = pd.to_numeric(df.price)
        # df.returns = pd.to_numeric(df.returns)
        if 'time' not in df.columns:
            df['time'] = dt.time.max
        if 'datatype' not in df.columns:
            df['datatype'] = 'settlement'

        df['time'] = df['time'].astype(pd.Timestamp)
        # df.expdate = pd.to_datetime(df.expdate)
        # df = df[df.value_date <= df.expdate]
        # setting data types
        # df.order = pd.to_numeric(df.order)
        # df.expdate = pd.to_datetime(df.expdate)

    df.reset_index(drop=True, inplace=True)
    # df = df.dropna()
    assert not df.empty

    return df


def ciprice(pricedata, rollover='opex'):
    """Constructs the CI price series.

    Args:
        pricedata (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : Dataframe with the following columns:
            Product | date | underlying | order | price | returns | expiry date
    """
    t = time.time()
    if rollover == 'opex':
        ro_dates = get_rollover_dates(pricedata)
        products = pricedata['pdt'].unique()
        # iterate over produts
        by_product = None
        for product in products:
            df = pricedata[pricedata.pdt == product]
            # lst = contract_mths[product]
            assert not df.empty
            most_recent = []
            by_date = None
            try:
                relevant_dates = ro_dates[product]
            except KeyError:
                df2 = df[
                        ['pdt', 'value_date', 'underlying_id', 'order', 'price', 'returns', 'expdate']]
                df2.columns = [
                    'pdt', 'value_date', 'underlying_id', 'order', 'price', 'returns', 'expdate']
                by_product = df2
                continue
            # iterate over rollover dates for this product.
            for date in relevant_dates:
                df = df[df.order > 0]
                order_nums = sorted(df.order.unique())
                breakpoint = max(most_recent) if most_recent else min(
                    df['value_date'])
                # print('breakpoint, date: ', breakpoint, date)
                by_order_num = None
                # iterate over all order_nums for this product. for each cont, grab
                # entries until first breakpoint, and stack wide.
                for ordering in order_nums:
                    # print('breakpoint, end, cont: ', breakpoint, date, cont)
                    df2 = df[df.order == ordering]
                    tdf = df2[(df2['value_date'] < date) & (df2['value_date'] >= breakpoint)][
                        ['pdt', 'value_date', 'underlying_id', 'order', 'price', 'returns', 'expdate']]
                    # print(tdf.empty)
                    tdf.columns = [
                        'pdt', 'value_date', 'underlying_id', 'order', 'price', 'returns', 'expdate']
                    tdf.reset_index(drop=True, inplace=True)
                    by_order_num = tdf if by_order_num is None else pd.concat(
                        [by_order_num, tdf])

                # by_date contains entries from all order_nums until current
                # rollover date. take and stack this long.
                by_date = by_order_num if by_date is None else pd.concat(
                    [by_date, by_order_num])
                most_recent.append(date)
                df.order -= 1

            by_product = by_date if by_product is None else pd.concat(
                [by_product, by_order_num])
        final = by_product

    else:
        final = -1
    elapsed = time.time() - t
    # print('[CI-PRICE] elapsed: ', elapsed)
    # final.to_csv('ci_price_final.csv', index=False)
    return final


def vol_by_delta(voldata, pricedata):
    """takes in a dataframe of vols and prices (same format as those returned by read_data),
     and generates delta-wise vol organized hierarchically by date, underlying and vol_id

    Args:
        voldata (TYPE): dataframe of vols
        pricedata (TYPE): dataframe of prices

    Returns:
        pandas dataframe: delta-wise vol of each option.
    """
    # print('voldata: ', voldata)
    # print('pricedata: ', pricedata)
    relevant_price = pricedata[
        ['pdt', 'underlying_id', 'value_date', 'price']]
    relevant_vol = voldata[['pdt', 'value_date', 'vol_id', 'strike',
                            'call_put_id', 'tau', 'vol', 'underlying_id']]

    # handle discrepancies in underlying_id format
    relevant_price.underlying_id = relevant_price.underlying_id.str.split().str[0]\
        + '  ' + relevant_price.underlying_id.str.split().str[1]

    relevant_vol.underlying_id = relevant_vol.underlying_id.str.split().str[0] + '  '\
        + relevant_vol.underlying_id.str.split().str[1]

    print('merging')
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['pdt', 'value_date', 'underlying_id'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['vol'] > 0)]

    print('computing deltas')

    merged['delta'] = merged.apply(compute_delta, axis=1)
    # merged.to_csv('merged.csv')
    merged['pdt'] = merged['underlying_id'].str.split().str[0]

    merged.delta = merged.delta.abs()

    print('getting labels')
    # getting labels for deltas
    delta_vals = np.arange(0.05, 0.96, 0.01)
    delta_labels = [str(int(100*x)) + 'd' for x in delta_vals]

    print('preallocating')
    # preallocating dataframes
    vdf = merged[['value_date', 'underlying_id', 'tau', 'vol_id',
                  'pdt', 'call_put_id']].drop_duplicates()

    products = merged.pdt.unique()

    vbd = pd.DataFrame(columns=delta_labels)

    print('beginning iteration:')
    # iterate first over products, thenn dates for that product, followed by
    # vol_ids in that product/date
    dlist = []
    for pdt in products:
        tmp = merged[merged.pdt == pdt]
        # tmp.to_csv('test.csv')
        dates = tmp.value_date.unique()
        vids = tmp.vol_id.unique()
        cpi = list(tmp.call_put_id.unique())
        for date in dates:
            for vid in vids:
                for ind in cpi:
                    # filter by vol_id and by day.
                    df = tmp[(tmp.value_date == date) &
                             (tmp.vol_id == vid) &
                             (tmp.call_put_id == ind)]

                    # sorting in ascending order of delta for interpolation
                    # purposes
                    df = df.sort_values(by='delta')

                    # reshaping data for interpolation.
                    drange = np.arange(0.05, 0.96, 0.01)
                    deltas = df.delta.values
                    vols = df.vol.values
                    # interpolating delta using Piecewise Cubic Hermite
                    # Interpolation (Pchip)

                    try:
                        # print('deltas: ', deltas)
                        # print('vols: ', vols)
                        # f1 = PchipInterpolator(
                        #     deltas, vols, extrapolate=True)
                        f1 = interp1d(deltas, vols, kind='linear',
                                      fill_value='extrapolate')
                    except ValueError:
                        # print('INTERPOLATION BROKE')
                        # print('vid: ', vid)
                        # print('cpi: ', ind)
                        # print('date: ', date)
                        # print('deltas: ', deltas)
                        # print('vols: ', vols)
                        continue

                    # grabbing delta-wise vols based on interpolation.
                    vols = f1(drange)

                    dic = dict(zip(delta_labels, vols))
                    # adding the relevant values from the indexing dataframe

                    dic['pdt'] = pdt
                    dic['vol_id'] = vid
                    dic['value_date'] = date
                    dic['call_put_id'] = ind
                    dic['minval'] = df.delta.values.min()
                    dic['maxval'] = df.delta.values.max()
                    dlist.append(dic)

    vbd = pd.DataFrame(dlist, columns=delta_labels.extend([
                       'pdt', 'vol_id', 'value_date', 'call_put_id', 'minval', 'maxval']))

    vbd = pd.merge(vdf, vbd, on=['pdt', 'vol_id', 'value_date', 'call_put_id'])

    # resetting indices
    return vbd


def civols(vdf, pdf, rollover='opex'):
    """Constructs the CI vol series.
    Args:
        vdf (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : dataframe with the following columns:
        pdt|order|value_date|underlying_id|vol_id|op_id|call_put_id|tau|strike|vol

    """
    t = time.time()
    if rollover == 'opex':
        ro_dates = get_rollover_dates(pdf)
        products = vdf['pdt'].unique()
        # iterate over produts
        by_product = None
        for product in products:
            df = vdf[vdf.pdt == product]
            most_recent = []
            by_date = None
            try:
                relevant_dates = ro_dates[product]
            except KeyError:
                # no rollover dates for this product
                df2 = df[['pdt', 'order', 'value_date', 'underlying_id', 'vol_id',
                          'op_id', 'call_put_id', 'tau', 'strike', 'vol']]
                df2.columns = ['pdt', 'order', 'value_date', 'underlying_id',
                               'vol_id', 'op_id', 'call_put_id', 'tau', 'strike', 'vol']
                by_product = df2
                continue

            # iterate over rollover dates for this product.
            for date in relevant_dates:
                # filter order > 0 to get rid of C_i that have been dealt with.
                df = df[df.order > 0]
                # sort orderings.
                order_nums = sorted(df.order.unique())
                breakpoint = max(most_recent) if most_recent else min(
                    df['value_date'])
                by_order_num = None
                # iterate over all order_nums for this product. for each cont, grab
                # entries until first breakpoint, and stack wide.
                for ordering in order_nums:
                    cols = ['pdt', 'order', 'value_date', 'underlying_id', 'vol_id',
                            'op_id', 'call_put_id', 'tau', 'strike', 'vol']
                    df2 = df[df.order == ordering]
                    tdf = df2[(df2['value_date'] < date) &
                              (df2['value_date'] >= breakpoint)][cols]
                    # renaming columns
                    tdf.columns = ['pdt', 'order', 'value_date', 'underlying_id',
                                   'vol_id', 'op_id', 'call_put_id', 'tau', 'strike', 'vol']
                    # tdf.reset_index(drop=True, inplace=True)
                    by_order_num = tdf if by_order_num is None else pd.concat(
                        [by_order_num, tdf])

                # by_date contains entries from all order_nums until current
                # rollover date. take and stack this long.
                by_date = by_order_num if by_date is None else pd.concat(
                    [by_date, by_order_num])
                most_recent.append(date)
                df.order -= 1

            by_product = by_date if by_product is None else pd.concat(
                [by_product, by_date])

        final = by_product
    else:
        final = -1
    elapsed = time.time() - t
    # print('[CI-VOLS] elapsed: ', elapsed)
    return final


#####################################################
################ Helper Functions ###################
#####################################################

def ttm(df, s, edf):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years

    Args:
        df (dataframe): dataframe containing option description.
        s (Series): Series of vol_ids
        edf (dataframe): dataframe of expiries.
    """
    s = s.unique()
    df['tau'] = ''
    df['expdate'] = ''
    for iden in s:
        expdate = get_expiry_date(iden, edf)
        try:
            expdate = pd.to_datetime(expdate.values[0])
        except IndexError:
            print('expdate not found for Vol ID: ', iden)
            expdate = pd.Timestamp('1990-01-01')
        currdate = pd.to_datetime(df[(df['vol_id'] == iden)]['value_date'])
        timedelta = (expdate - currdate).dt.days / 365
        df.ix[df['vol_id'] == iden, 'tau'] = timedelta
        df.ix[df['vol_id'] == iden, 'expdate'] = pd.to_datetime(expdate)
    return df


def get_expiry_date(volid, edf):
    """Computes the expiry date of the option given a vol_id """
    target = volid.split()
    op_yr = target[1][1]  # + decade
    # op_yr = op_yr.astype(str)
    op_mth = target[1][0]
    # un_yr = pd.to_numeric(target[1][-1]) + decade
    # un_yr = un_yr.astype(str)
    # un_mth = target[1][3]
    prod = target[0]
    overall = op_mth + op_yr  # + '.' + un_mth + un_yr
    expdate = edf[(edf['opmth'] == overall) & (edf['product'] == prod)][
        'expiry_date']
    expdate = pd.to_datetime(expdate)
    return expdate


def assign_ci(df, date):
    """Identifies the continuation numbers of each underlying.

    Args:
        df (Pandas Dataframe): Dataframe of price data, in the same format as that returned by read_data.

    Returns:
        Pandas dataframe     : Dataframe with the CIs populated.
    """
    today = pd.to_datetime(date)
    # today = pd.Timestamp('2017-01-01')
    curr_mth = month_to_sym[today.month]
    curr_yr = today.year
    products = df['pdt'].unique()
    df['order'] = ''
    for pdt in products:
        lst = contract_mths[pdt]
        df2 = df[df.pdt == pdt]
        ftmths = df2.ftmth.unique()
        for ftmth in ftmths:
            m1 = curr_mth + str(curr_yr % (2000 + decade))
            # print('ftmth: ', ftmth)
            # print('m1: ', m1)
            try:
                dist = find_cdist(m1, ftmth, lst)
            except ValueError:
                dist = -99
            df.ix[(df.pdt == pdt) & (df.ftmth == ftmth), 'order'] = dist
    return df


def find_cdist(x1, x2, lst):
    """Given two symbolic months (e.g. N7 and Z7), identifies the ordering of the month (c1, c2, etc.)

    Args:
        x1 (TYPE): current month
        x2 (TYPE): target month
        lst (TYPE): list of contract months for this product.

    Returns:
        int: ordering
    """
    x1mth = x1[0]
    x1yr = int(x1[1:])
    x2mth = x2[0]
    x2yr = int(x2[1:])

    print('find_cdist inputs: ', x1, x2, lst)

    # print(x1yr >)
    # case 1: month is a contract month.
    if x1mth in lst:
        # print('if case')
        reg = (lst.index(x2mth) - lst.index(x1mth)) % len(lst)
        # case 1.1: difference in years.
        # example: (Z7, Z9)
        if x2yr > x1yr and (x1mth == x2mth):
            # print('case 1')
            yrdiff = x2yr - x1yr
            dist = len(lst) * yrdiff
        # example: (K7, Z9)
        elif (x2yr > x1yr) and (x2mth > x1mth):
            # print('case 2')
            yrdiff = x2yr - x1yr
            dist = reg + (len(lst) * yrdiff)
        # examples: (Z7, H8), (N7, Z7), (Z7, U7)
        elif (x2yr == x1yr) and (x1mth > x2mth):
            return -1
        else:
            # print('case 3')
            # print('reg: ', reg)
            return reg

    # case 2: month is NOT a contract month. C1 would be nearest contract
    # month.
    else:
        num_fewer = [x for x in lst if x < x1mth]
        num_more = [x for x in lst if x > x1mth]
        # example: (V7, Z7)
        if (x1yr == x2yr) and (x2mth > x1mth):
            dist = num_more.index(x2mth) + 1
        # example: (V7, Z8)
        elif (x2yr > x1yr) and (x2mth > x1mth):
            yrdiff = x2yr - x1yr
            dist = yrdiff*len(num_more) + yrdiff*len(num_fewer) + \
                (num_more.index(x2mth) + 1)
        # example: (V7, H9)
        elif (x2yr > x1yr) and (x2mth < x1mth):
            yrdiff = x2yr - x1yr
            dist = yrdiff * len(num_more) + (yrdiff-1) * \
                len(num_fewer) + (num_fewer.index(x2mth) + 1)
        else:
            dist = None

    return dist


def scale_prices(pricedata):
    """Converts price data into returns, by applying log(curr/prev). 
       Treats each underlying security by itself so as to avoid taking
       the quotient of two different securities.

    Args:
        pricedata (pandas dataframe): Dataframe of prices, of the form returned by read_data

    Returns:
        pandas dataframe: dataframe with an additional field indicating returns.
    """
    ids = pricedata['underlying_id'].unique()
    pricedata['returns'] = ''
    for x in ids:
        # scale each price independently
        df = pricedata[(pricedata['underlying_id'] == x)]
        s = df['price']
        s1 = s.shift(-1)
        if len(s1) == 1 and np.isnan(s1.values[0]):
            ret = 0
        else:
            ret = np.log(s1/s)
        pricedata.ix[
            (pricedata['underlying_id'] == x), 'returns'] = ret

    # print(pricedata)
    pricedata = pricedata.fillna(0)
    # print(pricedata)
    return pricedata


def get_expiry(pricedata, edf, rollover=None):
    """Appends expiry dates to price data.

    Args:
        pricedata (TYPE): Dataframe of prices.
        edf (TYPE): Dataframe of expiries
        rollover (None, optional): rollover criterion. defaults to None.

    Returns:
        TYPE: Description
    """
    products = pricedata['pdt'].unique()
    pricedata['expdate'] = ''
    pricedata['expdate'] = pd.to_datetime(pricedata['expdate'])
    for prod in products:
        # 1: isolate the rollover date.
        df = pricedata[pricedata['pdt'] == prod]

        uids = df['underlying_id'].unique()
        for uid in uids:
            # need to get same-month (i.e. Z7.Z7 expiries)
            mth = uid.split()[1]
            try:
                roll_date = edf[(edf.opmth == mth) & (edf['product'] == prod)][
                    'expiry_date'].values[0]
                pricedata.ix[(pricedata['pdt'] == prod) &
                             (pricedata['underlying_id'] == uid), 'expdate'] = roll_date
            except IndexError:
                print('mth: ', mth)
                print('uid: ', uid)
                print('prod: ', prod)

    return pricedata


def get_rollover_dates(pricedata):
    """Generates dictionary of form {product: [c1 rollover, c2 rollover, ...]}. 
       If ci rollover is 0, then no rollover happens.

    Args:
        pricedata (TYPE): Dataframe of prices, same format as that returned by read_data

    Returns:
        rollover_dates: dictionary of rollover dates, organized by product.
    """
    products = pricedata['pdt'].unique()
    rollover_dates = {}
    for product in products:
        # filter by product.
        df = pricedata[pricedata.pdt == product]
        order_nums = sorted(pricedata['order'].unique())
        rollover_dates[product] = [0] * len(order_nums)
        for i in range(len(order_nums)):
            order = order_nums[i]
            df2 = df[df['order'] == order]
            test = df2[df2['value_date'] > df2['expdate']]['value_date']
            if not test.empty:
                try:
                    rollover_dates[product][i] = min(test)
                except (ValueError, TypeError):
                    print('i: ', i)
                    print('cont: ', order)
                    print('min: ', min(test))
                    print('product: ', product)
            else:
                expdate = df2['expdate'].unique()[0]
                rollover_dates[product][i] = pd.Timestamp(expdate)
    return rollover_dates


def compute_delta(x):
    """Helper function to aid with vol_by_delta, rendered in this format to make use of pd.apply

    Args:
        x (pandas dataframe): dataframe of vols.

    Returns:
        double: value of delta
    """
    s = x.price
    K = x.strike
    tau = x.tau
    char = x.call_put_id
    vol = x.vol
    r = 0
    try:
        d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
            (vol * sqrt(tau))
    except (ZeroDivisionError):
        d1 = -np.inf

    if char == 'C':
        # call option calc for delta and theta
        delta1 = norm.cdf(d1)
    if char == 'P':
        # put option calc for delta and theta
        delta1 = norm.cdf(d1) - 1

    return delta1


def get_min_start_date(vdf, pdf, lst, signals=None):
    """Gets the smallest starting date such that data exists by checking all the vol_ids
    present in the portfolio (inputted through lst). Returns the smallest date such that data
    exists for all vol_ids in the portfolio. If signals are included, then compares the initial result
    to the minimum date of the signals df, and returns the larger date.

    Args:
        vdf (pandas df): dataframe of volatilities.
        pdf (pandas df): dataframe of prices
        lst (list): list of vol_ids present in portfolio.
        signals (pandas df, optional): list of buy/sell/hold signals per day.

    Returns:
        pd.Timestamp: smallest date that satisfies all the conditions.
    """
    v_dates = []
    p_dates = []
    # test = pdf.merge(vdf, on=['pdt', 'value_date', 'underlying_id', 'order'])
    # test.to_csv('datasets/merged.csv', index=False)
    sig_date = None
    # get relevant underlying IDs rather than iterating through all of them.
    p_lst = [x.split()[0] + '  ' + x.split('.')[1] for x in lst]
    if signals is not None:
        signals.value_date = pd.to_datetime(signals.value_date)
        sig_date = min(signals.value_date)
        return sig_date

    elif len(lst) > 0:
        for vid in lst:
            df = vdf[vdf.vol_id == vid]
            v_dates.append(min(df.value_date))
        for uid in p_lst:
            # print('uid, date: ', uid, df.value_date.min())
            df = pdf[pdf.underlying_id == uid]
            p_dates.append(df.value_date.min())
        # print('get_min_start_date - vdates: ', v_dates)
        # print('get_min_start_date - pdates: ', p_dates)
        return max(max(v_dates), max(p_dates))

    else:
        raise ValueError(
            'neither signals nor lst is valid. No start date to be found. ')


def sanity_check(vdates, pdates, start_date, end_date, signals=None,):
    """Helper function that checks date ranges of vol/price/signal data and ensures
    that start/end dates passed into the simulation are consistent. 

    Args:
        vdates (TYPE): Dataframe of volatilities
        pdates (TYPE): Dataframe of prices
        start_date (TYPE): start date of the simulation
        end_date (TYPE): end date of the simulation
        signals (None, optional): dataframe of signals 

    Raises:
        ValueError: raised if date ranges of any of the dataframes do not line up,
                    or if start/end dates passed in are inconsistent/require time travel. 

    """
    if not np.array_equal(vdates, pdates):
        print('vol_dates: ', vdates)
        print('price_dates: ', pdates)
        print('difference: ', [x for x in vdates if x not in pdates])
        print('difference 2: ', [x for x in pdates if x not in vdates])
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range. Aborting run.')
    if signals is not None:
        sig_dates = signals.value_date.unique()
        if not np.array_equal(sig_dates, vdates):
            print('v - sig difference: ',
                  [x for x in vdates if x not in sig_dates])
            print('sig - v difference: ',
                  [x for x in sig_dates if x not in vdates])
            print('vdates: ', pd.to_datetime(vdates))
            raise ValueError('signal dates dont match up with vol dates')

        if not np.array_equal(sig_dates, pdates):
            print('p - sig difference: ',
                  [x for x in pdates if x not in sig_dates])
            print('sig - v differenceL ',
                  [x for x in sig_dates if x not in pdates])
            raise ValueError('signal dates dont match up with price dates')

    # if end_date specified, check that it makes sense (i.e. is greater than
    # start date)
    if all(i is not None for i in [start_date, end_date]):
        if start_date > end_date:
            raise ValueError(
                'Invalid end_date entered; current end_date is less than start_date')

    print('DATA INTEGRITY VERIFIED!')
    return


############### Intraday Data Processing Functions ####################


# def clean_intraday_data(df, sdf):
#     """Helper function that processes intraday data into a usable format. Performs the
#     following steps:
#     1) Column generation/naming: adds pdt, ftmth, uid, time and date columns, filters these columns
#     2) Timestep reconciliation.
#         > if there are multiple products, reconciles the timesteps as follows:
#             - for each day:
#                 d1, d2 = len(p1), len(p2)
#                 x = min(d1, d2)
#                 - for e in x:
#                     find closest thing in other <= e
#                     bundle together as e.
#     3) merges settlement data with intraday data, marks with a flag
#     4) isolates beginning of settlement period, marks with a flag.


#     Args:
#         df (TYPE): Dataframe of intraday prices.
#     """
#     ## Step 1 ##
#     df = handle_intraday_conventions(df)
#     ## Step 2 ##
#     df = timestep_recon(df)
#     ## Step 3 ##
#     df = insert_settlements(df, sdf)

#     return df


def handle_intraday_conventions(df):
    """Helper method that deals with product/ftmth/uid construction from BBG ticker symbols, checks/amends data types and filters out weekends/bank holidays from the data. 

    Args:
        df (TYPE): Description
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    ## Step 1 ##
    # first: Convert S H8 Comdty -> S  H8

    print('df.columns: ', df.columns)

    df['pdt'] = df.commodity.str[:2].str.strip()
    df['ftmth'] = df.commodity.str[2:-6].str.strip()
    df['underlying_id'] = df.pdt + '  ' + df.ftmth
    # datetime -> date and time columns.
    df.date_time = pd.to_datetime(df.date_time)
    df['time'] = df.date_time.dt.time
    df['value_date'] = pd.to_datetime(df.date_time.dt.date)

    # filter out weekends/bank holidays.
    cal = calendar()
    holidays = pd.to_datetime(cal.holidays(
        start=df.value_date.min(), end=df.value_date.max())).tolist()
    df = df[~df.value_date.isin(holidays)]
    df = df[df.value_date.dt.dayofweek < 5]

    # adding in flags used to isolate intraday vs settlement and intraday vs
    # settlement period
    df['datatype'] = 'intraday'

    cols = ['value_date', 'time', 'underlying_id',
            'pdt', 'ftmth', 'price', 'datatype']
    df = df[cols]

    df.columns = ['value_date', 'time', 'underlying_id',
                  'pdt', 'ftmth', 'price', 'datatype']

    df = df[df.price > 0]

    df.reset_index(drop=True, inplace=True)

    return df


def timestep_recon(df):
    """Helper method that reconciles timesteps for multi-product or multi-contract intraday simulations. Does the following:
    > for each day:
        get unique dataframes for each product/contract. 
        isolate dataset with minimum values.
        > for each timestamp in minimum:
            get closest timestamp in others that are <= timestamp. 
            bundle together under timestamp. 

    Base case: if df only contains 1 product and 1 contract, returns the dataframe with no modifications.   

    Args:
        df (pandas dataframe): dataframe of intraday price data. 
    """
    cols = df.columns
    uids = df.underlying_id.unique()
    dic_lst = []
    # base case: 1 uid.
    if len(uids) == 1:
        return df
    else:
        date_range = pd.to_datetime(df.value_date.unique())
        for date in date_range:
            t = time.clock()
            tdf = df[df.value_date == date]
            grps = tdf.groupby('underlying_id')
            # isolate the group with the least data
            ords = sorted([(x[0], len(x[1]), x[1])
                           for x in grps], key=lambda t: t[1])
            target_uid, target_len, target_data = ords.pop(0)
            dfs = list(zip(*ords))[2]
            target_data.reset_index(drop=True, inplace=True)
            print('----- date: %s ------' % (date.strftime('%Y-%m-%d')))
            print('target_length: ', target_len)
            for index in target_data.index:
                data = target_data.iloc[index]
                ts = data.time

                # helper to get all the other individuals.
                other_data = get_closest_ts_data(ts, dfs)
                # case: others have no data earlier than ts.
                if not other_data:
                    continue
                # isolate data for this day.
                # append data
                dic_lst.append(dict(zip(cols, data.values)))
                # extend other data
                dic_lst.extend(other_data)
            print('datalen: ', list(zip(*ords))[1])
            print('%s completed' % (date.strftime('%Y-%m-%d')))
            print('elapsed: ', time.clock() - t)
            print('--------------------------------------')

    fdf = pd.DataFrame.from_records(dic_lst)
    # removing all (time, price) duplicate entries.
    fdf['tup'] = fdf.price.astype(str) + ' ' + fdf.time.astype(str)
    fdf = fdf.drop_duplicates('tup')
    fdf = fdf[fdf.columns[:-1]]

    fdf = fdf[['pdt', 'ftmth', 'underlying_id', 'value_date',
               'time', 'price', 'datatype']]

    return fdf


def get_closest_ts_data(ts, others):
    """Helper function that takes in a timestamp, and a list of dataframes. 
    returns price associated with timestamp from each df in others that is closest to yet lesser than ts. 

    Args:
        ts (TYPE): target timestamp
        others (TYPE): list of dataframes. 
    """
    ret = []
    for df in others:
        cols = df.columns
        ts_list = [x for x in df.time if x <= ts]
        if not ts_list:
            continue
        valid_ts = max(ts_list)
        # filter data corresponding to closest timestep.
        data = df[df.time == valid_ts]
        # reassign timestep.
        data.time = ts
        data = dict(zip(cols, data.values[0]))
        ret.append(data)
    return ret


def insert_settlements(df, pdf):
    """Appends the settlement data to the intraday data. 

    Args:
        df (TYPE): Dataframe of intraday prices
        sdf (TYPE): Dataframe of settlement prices. 
    """
    assert not df.empty
    assert not pdf.empty
    pdf['time'] = dt.time.max
    pdf['datatype'] = 'settlement'
    pdf = pdf[['value_date', 'time', 'pdt', 'ftmth',
               'underlying_id', 'price', 'datatype']]
    df = df[['value_date', 'time', 'pdt', 'ftmth',
             'underlying_id', 'price', 'datatype']]
    x = pd.concat([df, pdf])

    # handle data types
    x.value_date = pd.to_datetime(x.value_date)
    x.time = x.time.astype(pd.Timestamp)

    # some sanity checks.
    assert not x.empty
    assert 'settlement' in x.datatype.unique()
    assert 'intraday' in x.datatype.unique()
    return x


def reorder_ohlc_data(df, pf):
    """Processes the OHLC data based on the breakeven of the 
    portfolio at on the simulation date.  

    Args:
        df (TYPE): dataframe of prices that 
        pf (TYPE): portfolio being handled 

    Returns:
        tuple: the initial dataframe and the modified dataframe. 
    """
    unique_uids = pf.get_unique_uids()
    breakevens = pf.breakeven()
    init_df = copy.deepcopy(df)

    # filter out all unnecessary uids.
    # print('df: ', df)

    print('unique uids: ', unique_uids)
    df = df[df.underlying_id.isin(unique_uids)]

    # print('reorder_ohlc_data - df: ', df)

    # assign the time for opens, leaving settlements the same.
    df.ix[df.price_id == 'px_open', 'time'] = dt.time(20, 59, 59, 0)

    for uid in unique_uids:
        pdt, mth = uid.split()
        # first: filter the dataframe to just consider this uid.
        tdf = df[df.underlying_id == uid]
        # second: get the breakeven for this uid.
        comp_val = pf.hedger.get_hedge_interval(uid)
        # sanity check: there should be exactly 4 entries.
        try:
            assert len(tdf) == 4
        except AssertionError as e:
            raise AssertionError("faulty tdf: ", tdf) from e
        px_open = tdf[tdf.price_id == 'px_open'].price.values[0]
        px_high = tdf[tdf.price_id == 'px_high'].price.values[0]
        px_low = tdf[tdf.price_id == 'px_low'].price.values[0]
        px_close = tdf[tdf.price_id == 'px_settle'].price.values[0]

        # case 1: open -> high -> low -> close.
        ord1 = (abs(px_high-px_open) + abs(px_low-px_high) +
                abs(px_close-px_low)) / comp_val
        # case 2: open -> low -> high -> close.
        ord2 = (abs(px_open-px_low) + abs(px_high-px_low) +
                abs(px_high-px_close)) / comp_val
        print('uid, ord1, ord2: ', uid, ord1, ord2)

        # if OHLC provides more breakevens than OLHC, go with OLHC and vice
        # versa
        if ord1 > ord2:
            # case: order as open-low-high-close.
            df.ix[df.price_id == 'px_low', 'time'] = dt.time(
                21, 59, 58, 0)
            df.ix[df.price_id == 'px_high', 'time'] = dt.time(
                22, 59, 58, 0)
            data_order = 'olhc'

        else:
            # case: order as open-high-low-close
            df.ix[df.price_id == 'px_high', 'time'] = dt.time(
                21, 59, 58, 0)
            df.ix[df.price_id == 'px_low', 'time'] = dt.time(
                22, 59, 58, 0)
            data_order = 'ohlc'

    df.sort_values(by=['time'])
    df.reset_index(drop=True, inplace=True)

    df = granularize(df, pf, ohlc=True)

    # print('df after reorder: ', df)

    return init_df, df, data_order


def clean_intraday_data(df, edf=None):
    """Does the following: 
    1) For each product/time combo:
        filter out price points with 0 volume
        aggregates volumes by price. 
        removes duplicate price entries. 


    Args:
        df (TYPE): Description
    """
    df = df[df.volume > 0]
    if 'pdt' not in df.columns:
        df['pdt'] = df.commodity.str[:2].str.strip()
    if 'time' not in df.columns:
        df['time'] = df.date_time.dt.time
    df['ftmth'] = df.commodity.str[2:5].str.strip()
    df['underlying_id'] = df.pdt + '  ' + df.ftmth
    df.date_time = pd.to_datetime(df.date_time)
    # filter for exchange timings.
    df = sanitize_intraday_timings(df, edf=edf)
    lst = []
    for (comm, date_time), grp in df.groupby(['underlying_id', 'date_time']):
        grp['block'] = (grp.price.shift(1) != grp.price).astype(int).cumsum()
        # print(grp)
        for (price, block), grp2 in grp.groupby(['price', 'block'], sort=False):
            dic = {'underlying_id': comm,
                   'date_time': date_time,
                   'price': price,
                   'volume': grp2.volume.sum()}
            lst.append(dic)
    ret = pd.DataFrame(lst)
    return ret


def sanitize_intraday_timings(df, filepath=None, edf=None):
    """Helper function that ignores pre-exchange printed results and only keeps entries
    within the start/end of the exchange timings. 

    Args:
        df (Dataframe): Dataframe of intraday prices. 

    Returns:
        Dataframe: With all timings outside of exchange timings removed. 
    """
    # read in the exchange timing dataframe.
    print('sanitize_intraday_timings: filepath - ', filepath)

    filepath = filepath if filepath is not None else ''
    filepath += 'datasets/exchange_timings.csv'

    print('final filepath: ', filepath)
    edf = pd.read_csv(filepath) if edf is None else edf
    edf['Exch Start Hours'] = pd.to_datetime(edf['Exch Start Hours']).dt.time
    edf['Exch End Hours'] = pd.to_datetime(edf['Exch End Hours']).dt.time

    edf.columns = ['pdt' if x == 'Product Id' else x for x in edf.columns]

    merged = pd.merge(
        df, edf[['pdt', 'Exch Start Hours', 'Exch End Hours']], on=['pdt'])

    merged = merged[(merged.time >= merged['Exch Start Hours']) &
                    (merged.time <= merged['Exch End Hours'])]

    merged.drop(['Exch Start Hours', 'Exch End Hours'], inplace=True, axis=1)
    return merged


# TODO: handle rounding of strikes when necessary.
def granularize(df, pf, interval=None, ohlc=False):
    """Helper function that takes in a dataframe filtered through reorder_ohlc_data, 
    and checks for consecutive price moves that exceed the breakeven/flat value hedging
    interval specified for that underlying id. If this condition is met, it splits up the move into hedge-interval level moves. Cases checked are as follows:

    1) case where the move is less than interval. 
        > ignores data, continues to next value. 
    2) case where move is exactly equal to interval. 
        > marks price point as relevant, updates comparative value to this value. 
    3) case where move is larger than interval. 
        > creates move_mult rows with intermediate values, where move_mult = floor(move/interval)


    Args:
        df (dataframe): dataframe of prices
        pf (portfolio object): portfolio being handled. 


    Returns:
        dataframe: dataframe with the price moves granularized according to the flat value/breakeven value. irrelevant datapoints are filtered out. 
    """

    # initial sanity check to see if only settlement data is present.
    if 'intraday' not in df.datatype.unique():
        return df

    fin_df = df.copy()

    # marking all relevant price moves as such.
    fin_df['relevant'] = ''

    # mark all settlements as relevant
    fin_df.ix[(fin_df.datatype == 'settlement'), 'relevant'] = True

    # edge case: OHLC data is handled a little differently. All default to true,
    # and irrelevant prices are explicitly set to false.
    # if ohlc:
    #     fin_df['relevant'] = True

    # get the UIDS that need to be handled.
    uids = df.underlying_id.unique()

    # get the last hedge points to ascertain the base
    # value against which we need to base interval-level moves.
    curr_prices = pf.hedger.get_hedgepoints().copy()

    for uid in uids:
        pdt = uid.split()[0]
        ticksize = multipliers[pdt][2]
        uid_df = df[df.underlying_id == uid].sort_values('time')
        uid_df.reset_index(drop=True, inplace=True)
        # get the hedging value.
        interval = pf.hedger.get_hedge_interval(
            uid) if interval is None else interval
        print('interval: ', interval)
        curr_price = curr_prices[uid]
        print('curr_price: ', curr_price)
        print('uid, last hedgepoint: ', uid, curr_price)

        # iterate over the rows of the uid_df
        lastrow = None
        for index in uid_df.index:
            row = uid_df.iloc[index]
            diff = row.price - curr_price

            relevant, move_mult = pf.hedger.is_relevant_price_move(
                uid, row.price, comparison=curr_price)

            # skip settlements since they are valid by default.
            if row.datatype == 'settlement':
                continue

            # if it's less than interval and intraday, nothing needs to be
            # done. set to false.
            if not relevant and row.datatype == 'intraday':
                # set it to false.
                if ohlc:
                    fin_df.ix[(fin_df.underlying_id == uid) &
                              (fin_df.time == row.time) &
                              (fin_df.price == row.price), 'relevant'] = False
                continue

            # case: price move is relevant.
            else:
                # if it's close to interval, then is_relevant_price_move will pick it up.
                # just reset comparative price , and mark as relevant
                if np.isclose(abs(diff), interval) or abs(diff) == interval:
                    print('--------------- handling row ' +
                          str(index) + ' --------------')
                    print('diff, interval: ', diff, interval)
                    print(
                        'found move close to interval. resetting curr_price to ' + str(row.price))

                    fin_df.ix[(fin_df.underlying_id == uid) &
                              (fin_df.time == row.time) &
                              (fin_df.price == row.price), 'relevant'] = True

                    curr_price = row.price

                # case: difference is greater than the hedge level. need to
                # create new row.
                else:
                    print('--------------- handling row ' +
                          str(index) + ' --------------')
                    if index == 0:
                        # edge case: if open is > 1 be move, take hedge there, but set
                        # it as last hedge point
                        curr_price = row.price
                        print(
                            'open is > 1 be move; curr_price updated to ' + str(row.price))
                        # mark this point as relevant.
                        fin_df.ix[(fin_df.underlying_id == uid) &
                                  (fin_df.time == row.time) &
                                  (fin_df.price == row.price), 'relevant'] = True

                    # not the first row. create new rows to simulate resting
                    # orders.
                    else:
                        print('curr_price: ', curr_price)
                        print('row price: ', row.price)
                        print('diff: ', diff)
                        print('datatype: ', row.datatype)
                        # number of breakevens/value moved = number of new rows that
                        # need to be created, and is given by move_mult
                        curr_time = uid_df.iloc[index-1].time
                        print('curr_time: ', curr_time)

                        print('move mult: ', move_mult)

                        for x in range(int(move_mult)):
                            # multiplier to ascertain if the price rose or fell from
                            # last hedgepoint
                            mult = -1 if diff < 0 else 1

                            newprice = curr_price + (interval*mult)
                            # round newprice to closest future tick that is larger
                            # than newprice.
                            # newprice = ceil(newprice/ticksize)*ticksize
                            print('intermediate price: ', newprice)

                            if lastrow is not None:
                                # case: new row added in previous loop has a time greater than
                                # the previous index row in the dataframe; use this
                                # time.
                                if lastrow['time'] > curr_time:
                                    prev_time = lastrow['time']
                                    print('using lastrow time: ', prev_time)
                                    newtime = dt.time(prev_time.hour, prev_time.minute,
                                                      prev_time.second, prev_time.microsecond + 1)
                                else:
                                    newtime = dt.time(curr_time.hour, curr_time.minute,
                                                      curr_time.second, curr_time.microsecond + 1)
                                    print('newtime, currtime: ',
                                          newtime, curr_time)

                            else:
                                newtime = dt.time(curr_time.hour, curr_time.minute,
                                                  curr_time.second, curr_time.microsecond + 1)
                                print('newtime, currtime: ',
                                      newtime, curr_time)

                            newrow = {'value_date': row.value_date, 'time': newtime, 'pdt': row.pdt,
                                      'ftmth': row.ftmth, 'price': newprice, 'datatype': 'intraday',
                                      'underlying_id': uid, 'relevant': True}

                            lastrow = newrow
                            if ohlc:
                                newrow['price_id'] = 'midpt'

                            print('newrow added: ', newrow)
                            fin_df = fin_df.append(newrow, ignore_index=True)
                            print('curr_price updated to ' + str(newprice))
                            curr_price = newprice

    # sort values by time, filter relevant entries and reset indexes.
    fin_df.sort_values(by='time', inplace=True)
    if ohlc:
        print('pdf after ohlc reorder: ', fin_df)
    fin_df = fin_df[fin_df.relevant == True]
    fin_df.reset_index(drop=True, inplace=True)

    return fin_df


##########################################################################
##########################################################################
##########################################################################
