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
from scipy.interpolate import interp1d
from scipy.stats import norm
from math import log, sqrt
import time
from ast import literal_eval
from collections import OrderedDict
import copy
import datetime as dt
import os
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
    'CA': ['H', 'K', 'U', 'Z'],
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
    'CA': [1.0604, 50, 0.25, 1, 53.02],
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

    vdf['monday'] = vdf.value_date.dt.weekday == 0

    vdf.loc[vdf.monday == True, 'value_date'] -= pd.Timedelta('3 day')
    vdf.loc[vdf.monday == False, 'value_date'] -= pd.Timedelta('1 day')

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
        spec (dataframe, optional): pandas dataframe containing the portfolio specifications 

    Returns:
        TYPE: portfolio object

    Raises:
        ValueError: raised if object creation parameters cannot be found. 

    """

    # initializing variables
    sim_start = None
    oplist = {'hedge': [], 'OTC': []}
    ftlist = {'hedge': [], 'OTC': []}

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
        dict: OTC/hedge -> list of daily options. 
    """
    from pandas.tseries.offsets import BDay
    sim_start += BDay(1)
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
                print('sim_start: ', sim_start)
                print('expdate: ', expdate)
                dx = sim_start
                end = expdate
                incl = 1
                taus = []
                while dx <= end:
                    exptime = ((dx - sim_start).days + incl)/365
                    taus.append(exptime)
                    step = 3 if dx.dayofweek == 4 else 1 
                    dx += pd.Timedelta(str(step) + ' days')

                # print('taus: ', taus)

                # daterange = pd.bdate_range(sim_start, expdate)
                # print(daterange)
                # print('4th July in DateRange: ', pd.Timestamp('2018-07-04') in daterange)

                # # print('daterange: ', daterange)
                # taus = [((expdate - (b_day)).days) /
                #         365 for b_day in daterange if b_day < expdate]

                # print(taus)
                # print(len(taus))

                strike, char, vol, underlying, payoff, shorted, month, ordering, lots, settlement \
                    = params['strike'], params['char'], params['vol'], params['underlying'], \
                    params['payoff'],  params['shorted'], params['month'], \
                    params['ordering'], params['lots'],\
                    params['settlement'], params['bvol'], params['bvol2']
                # barrier params
                direc, barrier, ki, ko, rebate, bvol, bvol2 = \
                    params['direc'], params['barrier'], params[
                        'ki'], params['ko'], params['rebate'], params['bvol'], params['bvol2']

                # creating the bullets corresponding to this daily option.
                for tau in taus:
                    ui = copy.deepcopy(underlying)
                    op_i = Option(strike, tau, char, vol, ui, payoff, shorted, month, direc=direc,
                                  barrier=barrier, lots=lots, bullet=False, ki=ki, ko=ko, rebate=rebate,
                                  ordering=ordering, settlement=settlement, bvol=bvol, bvol2=bvol2)
                    bullets.append(op_i)

            lst.extend(bullets)

    return dic


###############################################################
################### Data Cleaning Functions ###################
###############################################################


def clean_data(df, flag, edf=None, writeflag=None):
    """Function that cleans the dataframes passed into it according to the flag passed in.
    1) flag == 'exp':
        > datatype conversion to pd.Timestamp
        > filters for data > 2010.
        > converts formatting; i.e. C H17 --> C H7
    2) flag == 'vol':
        > convert date strings to pd.Timestamp
        > generates additional fields from existing ones.

    3) flag == 'price':
        > expiry date and ordering like in vol.
        > date strings to pd.Timestamp

    Args:
        df (pandas dataframe): the dataframe to be cleaned.
        flag (pandas dataframe): determines which dataframe is being processed.
        edf (pandas dataframe): dataframe containing the expiries of options.

        writeflag (None, optional): Description

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
        df.expdate = pd.to_datetime(df.expdate)
        df = df.dropna()
        assert not df.empty
        # calculating time to expiry from vol_id
        df['tau'] = (df.expdate - df.value_date).dt.days/365

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
        df = df.fillna(0)
        df.value_date = pd.to_datetime(df.value_date)
        df.price = pd.to_numeric(df.price)
        if 'time' not in df.columns:
            df['time'] = dt.time.max
        if 'datatype' not in df.columns:
            df['datatype'] = 'settlement'
    df.reset_index(drop=True, inplace=True)
    # df = df.dropna()
    assert not df.empty

    return df


def vol_by_delta(voldata, pricedata):
    """takes in a dataframe of vols and prices (same format as those returned by read_data),
     and generates delta-wise vol organized hierarchically by date, underlying and vol_id

    Args:
        voldata (TYPE): dataframe of vols
        pricedata (TYPE): dataframe of prices

    Returns:
        pandas dataframe: delta-wise vol of each option.
    """
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

    # print('find_cdist inputs: ', x1, x2, lst)

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


def handle_intraday_conventions(df):
    """Helper method that deals with product/ftmth/uid construction from BBG ticker symbols, 
    checks/amends data types and filters out weekends/bank holidays from the data. 

    Args:
        df (TYPE): Description
    """
    # from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    ## Step 1 ##
    # first: Convert S H8 Comdty -> S  H8

    df['pdt'] = df.underlying_id.str[:2].str.strip()
    df['ftmth'] = df.underlying_id.str[2:].str.strip()

    df['underlying_id'] = df.pdt + '  ' + df.ftmth
    # datetime -> date and time columns.
    df.date_time = pd.to_datetime(df.date_time)
    df['time'] = df.date_time.dt.time
    df['value_date'] = pd.to_datetime(df.date_time.dt.date)

    # adding in flags used to isolate intraday vs settlement and intraday vs
    # settlement period
    df['datatype'] = 'intraday'

    cols = ['value_date', 'time', 'underlying_id',
            'pdt', 'ftmth', 'price', 'datatype', 'date_time', 'volume']
    df = df[cols]

    df.columns = ['value_date', 'time', 'underlying_id',
                  'pdt', 'ftmth', 'price', 'datatype', 'date_time', 'volume']

    df = df[df.price > 0]

    df.reset_index(drop=True, inplace=True)

    return df


def timestep_recon(df):
    """Helper method that reconciles timesteps for multi-product or multi-contract intraday simulations. 
    Does the following:
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
    # breakevens = pf.breakeven()
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

    df.sort_values(by=['index'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = granularize(df, pf, ohlc=True)

    # print('df after reorder: ', df)

    return init_df, df, data_order


def clean_intraday_data(df, start_date, end_date, edf=None, filepath=None):
    """
    Does the following:
        1) For each product/time combo:
            filter out price points with 0 volume
            aggregates volumes by price. 
            removes duplicate price entries. 


    Args:
        df (dataframe): Dataframe of raw intraday prices.
        start_date (string): simulation start date, passed into sanitize_intraday_timings
        end_date (string): simulation end date, passed into sanitize_intraday_timings
        edf (dataframe, optional): dataframe of exchange timings
        filepath (string, optional): filepath to dataframe of exchange timings 

    Returns:
        TYPE: Description
    """
    assert not df.empty

    df = df[df.volume > 0]
    if 'pdt' not in df.columns:
        df['pdt'] = df.commodity.str[:2].str.strip()
    if 'time' not in df.columns:
        df['time'] = df.date_time.dt.time
    df['ftmth'] = df.commodity.str[2:5].str.strip()
    df['underlying_id'] = df.pdt + '  ' + df.ftmth

    assert not df.empty

    # filter for exchange timings.
    df = sanitize_intraday_timings(
        df, start_date, end_date, edf=edf, filepath=filepath)

    assert not df.empty
    lst = []

    # print('clean_intraday_data: beginning aggregation...', end="")
    for (comm, date_time), grp in df.groupby(['underlying_id', 'date_time']):
        grp['block'] = (grp.price.shift(1) != grp.price).astype(int).cumsum()
        for (price, block), grp2 in grp.groupby(['price', 'block'], sort=False):
            dic = {'underlying_id': comm,
                   'date_time': date_time,
                   'price': price,
                   'volume': grp2.volume.sum()}
            lst.append(dic)
    ret = pd.DataFrame(lst)

    assert not ret.empty

    return ret


def sanitize_intraday_timings(df, start_date, end_date, filepath=None, edf=None):
    """Helper function that ignores pre-exchange printed results and only keeps entries
    within the start/end of the exchange timings. 

    Args:
        df (Dataframe): Dataframe of intraday prices. 

    Returns:
        Dataframe: With all timings outside of exchange timings removed. 
    """

    timingspath = filepath + 'exchange_timings.csv'

    if os.path.exists(timingspath):
        edf = pd.read_csv(timingspath)
    else:
        from .fetch_data import fetch_exchange_timings
        edf = fetch_exchange_timings()
        edf.to_csv(timingspath, index=False)

    edf['exch_start_hours'] = pd.to_datetime(edf['exch_start_hours']).dt.time
    edf['exch_end_hours'] = pd.to_datetime(edf['exch_end_hours']).dt.time

    edf.columns = ['pdt' if x == 'product_id' else x for x in edf.columns]

    merged = pd.merge(
        df, edf[['pdt', 'exch_start_hours', 'exch_end_hours', 'pytz_desc']], on=['pdt'])

    fin = pd.DataFrame()

    for pdt in merged.pdt.unique():
        t_merged = merged[merged.pdt == pdt]
        pdt_start = t_merged['exch_start_hours'].values[0]
        pdt_end = t_merged['exch_end_hours'].values[0]
        # overnight market case. covert, filter, unconvert.
        if pdt_start > pdt_end:
            # localize.
            t_merged = handle_overnight_market_timings(
                t_merged, start_date, end_date)
        else:
            t_merged = t_merged[(t_merged.time >= t_merged['exch_start_hours']) &
                                (t_merged.time <= t_merged['exch_end_hours'])]

        assert not t_merged.empty
        t_merged.date_time = t_merged.date_time.dt.tz_localize(
            t_merged.pytz_desc.unique()[0])

        # convert to the default timezone: Dubai.
        t_merged.date_time = t_merged.date_time.dt.tz_convert('Asia/Dubai')
        fin = pd.concat([fin, t_merged])
    fin.drop(['exch_start_hours', 'exch_end_hours'], inplace=True, axis=1)
    return fin


def handle_overnight_market_timings(df, start_date, end_date):
    """Helper function that handles timezone conversion and filtering by doing the following:
    1) localize. 
    2) convert to DXB. 
    3) filter. 
    4) convert back to local timezone. 

    Args:
        df (TYPE): dataframe of price data. 
    """
    timezone = df.pytz_desc.unique()[0]
    default = 'Asia/Dubai'
    # find the exchange timings in terms of dxb time.
    pdt_start = pd.to_datetime(df['exch_start_hours'].values[
                               0].strftime('%H:%M:%S'))
    pdt_end = pd.to_datetime(df['exch_end_hours'].values[
                             0].strftime('%H:%M:%S'))

    pdt_start = pdt_start.tz_localize(timezone).tz_convert(default).time()
    pdt_end = pdt_end.tz_localize(timezone).tz_convert(default).time()

    # print('pdt_start: ', pdt_start)
    # print('pdt_end: ', pdt_end)

    # 1) localize.
    df.date_time = df.date_time.dt.tz_localize(timezone)
    # 2) convert to dxb time.
    df.date_time = df.date_time.dt.tz_convert(default)
    # 3) filter according to the dxb-standardized time.
    df = df[(df.date_time.dt.time >= pdt_start) &
            (df.date_time.dt.time <= pdt_end) &
            (pd.to_datetime(df.date_time.dt.date) >= start_date) &
            (pd.to_datetime(df.date_time.dt.date) <= end_date)]
    # 4) convert back to local timezone.
    df.date_time = df.date_time.dt.tz_convert(timezone)
    # 5) strip timezone awareness.
    df.date_time = df.date_time.dt.tz_localize(None)

    return df


# TODO: handle rounding of strikes when necessary.
# TODO: handle HedgeParser implementation to account for trailing stops etc.
def granularize(df, pf, interval=None, ohlc=False, intraday=False):
    """Helper function that takes in a dataframe 
    and checks for consecutive price moves that exceed the breakeven/flat value hedging
    interval specified for that underlying id. If this condition is met, 
    it splits up the move into hedge-interval level moves. Cases checked are as follows:

    1) case where the move is less than interval. 
        > ignores data, continues to next value. 
    2) case where move is exactly equal to interval. 
        > marks price point as relevant, updates comparative value to this value. 
    3) case where move is larger than interval. 
        > creates move_mult rows with intermediate values, where move_mult = floor(move/interval)


    Args:
        df (dataframe): dataframe of prices
        pf (portfolio object): portfolio being handled. 

        interval (None, optional): Description
        ohlc (bool, optional): Description
        intraday (bool, optional): Description

    Returns:
        dataframe: dataframe with the price moves granularized according to the 
        flat value/breakeven value. irrelevant datapoints are filtered out. 
    """

    # initial sanity check to see if only settlement data is present.
    if 'intraday' not in df.datatype.unique():
        return df

    fin_df = df.copy()

    # print('pre-granularize df: ', df)

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
                        print('interval: ', interval)
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
    if ohlc:
        fin_df.sort_values(by='time', inplace=True)
        print('pdf after ohlc reorder: ', fin_df)

    fin_df = fin_df[fin_df.relevant == True]
    # edge case: duplicate consecutive entries in results. want to filter by
    # block of prices, keeping order intact.
    if intraday:
        fin_df['block'] = (fin_df.price.shift(
            1) != fin_df.price).astype(int).cumsum()
        fin_df = fin_df.drop_duplicates('block')
        fin_df.sort_values(by='time', inplace=True)

    print('fin_df: ', fin_df)
    fin_df.reset_index(drop=True, inplace=True)

    return fin_df


def create_intermediate_rows(lst, lastrow):
    """Helper function that constructs intermediate rows as specified by
    the granularize function. 

    Args:
        lst (TYPE): Description
        lastrow (TYPE): Description
    """
    pass


def pnp_format(filepath, pdts=None):
    """Helper method that takes the relevant product positions from PnP 
    and outputs a table compatible with prep_data.prep_portfolio 

    Args:
        filepath (TYPE): Filepath to the PnP file. 
        pdts (TYPE): Relevant products. 

    Returns:
        TYPE: Description
    """
    # final columns.
    f_cols = ['Type', 'strike', 'vol_id', 'call_put_id',
              'optiontype', 'shorted', 'hedgeorOTC', 'lots', 'pdt', 'counterparty']

    # handling options.
    optdump = pd.read_excel(filepath, sheetname='OptDump')
    # filtering for MM-relevant results, getting relevant columns, dropping
    # nans.

    cols = list(optdump.columns)
    print('cols: ', cols)
    # cols = cols[12:-12]
    optdump = optdump[cols].dropna()
    optdump = optdump[optdump['Portfolio'].str.contains('MM-')]
    # assign product.
    optdump['pdt'] = optdump['Portfolio'].str.split('-').str[1].str.strip()
    # renaming columns; keep counterparty because we want to extract
    # directionals after processing.
    optdump.columns = ['label', 'Portfolio', 'buy/sell', 'vid_str', 'strike', 'cpi',
                       'counterparty', 'sum_init_pre', 'lots', 'pdt']
    # assign vol_id
    optdump['vol_id'] = optdump.pdt + '  ' + optdump.vid_str
    # assign strike
    optdump.loc[(~optdump.pdt.isin(['SM', 'CA', 'DF', 'QC', 'CC'])),
                'strike'] = optdump.strike * 100
    optdump.strike = round(optdump.strike, 3)
    # assign shorted
    optdump.loc[(optdump['buy/sell'] == 'B'), 'shorted'] = False
    optdump.loc[(optdump['buy/sell'] == 'S'), 'shorted'] = True
    # assign call_put_id
    optdump.loc[(optdump['cpi'] == 'C'), 'call_put_id'] = 'call'
    optdump.loc[(optdump['cpi'] == 'P'), 'call_put_id'] = 'put'
    # assign type, optiontype and hedgeorOTC. defaults to Option, 'amer' and
    # OTC
    optdump['Type'] = 'Option'
    optdump['optiontype'] = 'amer'
    optdump['hedgeorOTC'] = 'OTC'

    ops = optdump[f_cols]
    ops = ops[ops.lots != 0]

    print('option columns: ', ops.columns)

    directionals = ops[ops.counterparty.str.contains('-OID')]
    ops.drop('counterparty', axis=1, inplace=True)

    # handle the futures, performing the same steps.
    df = pd.read_excel(filepath, sheetname='FutDump')
    df['Net Pos'].replace('-', 0, inplace=True)
    df = df[df['Net Pos'] != 0]

    df = df.dropna()
    df = df[df.Portfolio.str.contains('MM-')]
    cols = ['Portfolio', 'Contract', 'Net Pos']
    df = df[cols]
    df['pdt'] = df.Portfolio.str.split('-').str[1].str.strip()
    df['vol_id'] = df.pdt + '  ' + df.Contract.str.strip()
    df['Type'] = 'Future'
    df['strike'] = np.nan
    df['call_put_id'] = np.nan
    df['optiontype'] = np.nan
    df['Net Pos'] = df['Net Pos'].astype(float)

    df.loc[(df['Net Pos'] < 0), 'shorted'] = True
    df.loc[(df['Net Pos'] > 0), 'shorted'] = False
    df['hedgeorOTC'] = 'hedge'
    df['lots'] = abs(df['Net Pos'])
    # df = df[f_cols]
    df = df[df.lots != 0]
    df.drop(['Contract', 'Net Pos', 'Portfolio'], axis=1, inplace=True)

    print('future columns: ', df.columns)

    # concatenate and instantiate dummy variables.
    final = pd.concat([ops, df])
    final['barriertype'] = np.nan
    final['direction'] = np.nan
    final['knockin'] = np.nan
    final['knockout'] = np.nan
    final['bullet'] = np.nan

    if pdts is not None:
        final = final[final.vol_id.str[:2].str.strip().isin(pdts)]

    return final, directionals


def aggregate_pnp_positions(df):
    """Helper function that iterates through the final DF obtained from pnp_format, and
    aggregates option positions by vol_id, strike and lots to arrive at net positions. 

    Args:
        df (TYPE): Description
    """
    # handle the options
    df['mult'] = (df.shorted == False).astype(int)
    df['mult'] = df['mult'].replace(0, -1)
    df['real_lots'] = df['mult'] * df.lots
    lst = []
    ops = df[df.Type == 'Option']
    for (strike, vol_id, cpi), grp in ops.groupby(['strike', 'vol_id', 'call_put_id']):
        # print(strike, vol_id, cpi)
        pdt = vol_id[:2].strip()
        net_pos = grp.real_lots.sum()
        shorted = True if net_pos < 0 else False
        dic = {'Type': 'Option', 'strike': strike, 'vol_id': vol_id, 'call_put_id': cpi,
               'optiontype': 'amer', 'shorted': shorted, 'hedgeorOTC': 'OTC',
               'lots': abs(net_pos), 'pdt': pdt, 'barriertype': np.nan, 'direction': np.nan,
               'knockin': np.nan, 'knockout': np.nan, 'bullet': True}
        lst.append(dic)
    fts = df[df.Type == 'Future']
    for uid, grp in fts.groupby('vol_id'):
        dic = {}
        net_pos = grp.real_lots.sum()
        shorted = True if net_pos < 0 else False
        dic = {'Type': 'Future', 'strike': np.nan, 'vol_id': uid, 'call_put_id': np.nan,
               'optiontype': np.nan, 'shorted': shorted, 'hedgeorOTC': 'hedge',
               'lots': abs(net_pos), 'pdt': pdt, 'barriertype': np.nan, 'direction': np.nan,
               'knockin': np.nan, 'knockout': np.nan, 'bullet': np.nan}
        lst.append(dic)

    final = pd.DataFrame(lst)
    final = final[final.lots != 0]

    print('final.columns: ', final.columns)

    # final.drop(['mult', 'real_lots'], axis=1, inplace=True)

    return final


def _filter_outliers(df, threshold):
    """Helper function called in filter_outliers
    that marks all outlier values as True. 

    Args:
        df (TYPE): dataframe of cleaned intraday prices grouped by product, UID and date. 
        threshold (float): value above which a price diff considered an anomaly. 
    """

    # get the threshold
    tick = multipliers[df.pdt.unique()[0]][-3]
    threshold *= tick
    df.ix[abs(df.price.diff()) > threshold, 'anomalous'] = True
    return df


def filter_outliers(df, fixed=10, dic=None, drop=False):
    """Helper function that filters out anomalous price points (i.e.)
    prices moving 60 ticks in one second, etc. 

    Args:
        df (TYPE): dataframe of cleaned intraday prices. 
        fixed (float, optional): indicates if the thresholds are fixed multiples. 
        dic (None, optional): if fixed is None, use this dictionary of product -> threshold (in ticks) to filter outliers.

    Returns:
        TYPE: Description
    """
    df = df.groupby(['pdt', 'underlying_id', 'value_date']
                    ).apply(_filter_outliers, threshold=fixed)

    if drop:
        df = df[pd.isnull(df.anomalous)]

    return df


##########################################################################
##########################################################################
##########################################################################
