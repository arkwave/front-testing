"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 3/4/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""

###########################################################
############### Imports/Global Variables ##################
###########################################################


from .portfolio import Portfolio
from .classes import Option, Future
import pandas as pd
import datetime as dt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from math import log, sqrt
import time
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

###############################################################
################## Data Read-in Functions #####################
###############################################################


def read_data(filepath):
    """Wrapper method that handles all read-in and preprocessing. This function does the following:
    1) reads in path to volatility, price and expiry tables from portfolio_specs.txt
    2) reads in dataframes from said paths
    3) cleans that data in different ways, depending on the flag passed in. Exact information can be found in clean_data function.

    Args:
        filepath (string, optional): the relative filepath to portfolio_specs.txt

    Returns:
        pandas dataframes x 3: volatility data, price data, expiry data.
    """
    t = time.time()
    with open(filepath) as f:
        try:
            # get paths
            volpath = f.readline().strip('\n')
            pricepath = f.readline().strip('\n')
            expath = f.readline().strip('\n')
            # import dataframes
            volDF = pd.read_csv(volpath).dropna()
            priceDF = pd.read_csv(pricepath).dropna()
            edf = pd.read_csv(expath).dropna()
            # clean dataframes
            edf = clean_data(edf, 'exp')
            volDF = clean_data(volDF, 'vol', edf=edf)
            priceDF = clean_data(priceDF, 'price', edf=edf)
            # final preprocessing steps
            final_price = ciprice(priceDF)
            final_vol = civols(volDF, priceDF)

        except FileNotFoundError:
            print(volpath)
            print(pricepath)
            print(expath)
            import os
            print(os.getcwd())
    elapsed = time.time() - t
    # print('[READ_DATA] elapsed: ', elapsed)
    final_vol.to_csv('datasets/final_vols.csv', index=False)
    final_price.to_csv('datasets/final_price.csv', index=False)
    edf.to_csv('datasets/final_expdata.csv', index=False)
    return final_vol, final_price, edf


# def prep_portfolio(voldata, pricedata, filepath):
#     """
# Reads in portfolio specifications from portfolio_specs.txt and
# constructs a portfolio object. The paths to the dataframes are specified
# in the first 3 lines of portfolio_specs.txt, while the remaining
# securities to be added into this portfolio are stored in the remaining
# lines. By design, all empty lines or lines beginning with %% are
# ignored.

#     Args:
#         voldata (pandas dataframe)  : dataframe containing the volatility surface (i.e. strike-wise volatilities)
#         pricedata (pandas dataframe): dataframe containing the daily price of underlying.
#         filepath (string)           : path to portfolio_specs.txt

#     Returns:
#         pf (Portfolio)              : a portfolio object.
#     """
#     oplist = {'hedge': [], 'OTC': []}
#     ftlist = {'hedge': [], 'OTC': []}
#     sim_start = min(min(voldata.value_date), min(pricedata.value_date))
#     sim_start = pd.to_datetime(sim_start)
#     t = time.time()
#     pf = Portfolio()
#     curr_mth = dt.date.today().month
#     curr_mth_sym = month_to_sym[curr_mth]
#     curr_yr = dt.date.today().year % (2000 + decade)
#     curr_sym = curr_mth_sym + str(curr_yr)
#     # sim_start = pd.Timestamp('2017-01-01')
#     # curr_sym = month_to_sym[sim_start.month] + \
#     #     str(sim_start.year % (2000 + decade))
#     # print('x1: ', curr_sym)
#     with open(filepath) as f:
#         for line in f:
#             # ignore lines with %% or blank lines.
#             if "%%" in line or line in ['\n', '\r\n']:
#                 continue
#             else:
#                 inputs = line.split(',')
#                 # input specifies an option
#                 if inputs[0] == 'Option':
#                     strike = float(inputs[1])
#                     volid = str(inputs[2])
#                     opmth = volid.split()[1].split('.')[0]
#                     char = str(inputs[3])
#                     volflag = 'C' if char == 'call' else 'P'

#                     # get tau from data
#                     tau = voldata[(voldata['value_date'] == sim_start) &
#                                   (voldata['vol_id'] == volid) &
#                                   (voldata['call_put_id'] == volflag)]['tau'].values[0]
#                     # print('days to exp: ', round(tau * 365))
#                     # get vol from data
#                     vol = voldata[(voldata['vol_id'] == volid) &
#                                   (voldata['call_put_id'] == volflag) &
#                                   (voldata['value_date'] == sim_start) &
#                                   (voldata['strike'] == strike)]['settle_vol'].values[0]
#                     # american vs european payoff
#                     payoff = str(inputs[4])
#                     # american or european barrier.
#                     barriertype = None if inputs[
#                         5] == 'None' else str(inputs[5])
#                     # direction of barrier.
#                     direc = None if inputs[6] == 'None' else str(inputs[6])
#                     # knock-in. is not None iff this is a knock-in option.
#                     ki = None if inputs[7] == 'None' else int(inputs[7])
#                     # knock-out. is not None iff this is a knock-out option.
#                     ko = None if inputs[8] == 'None' else int(inputs[8])
#                     # bullet vs daily pay-out. defaults to False.
#                     bullet = True if inputs[9] == 'True' else False
#                     # hedge or OTC
#                     flag = str(inputs[11]).strip('\n')
#                     # short or long position on this option.
#                     shorted = True if inputs[10] == 'short' else False

#                     # handle underlying construction
#                     f_mth = volid.split()[1].split('.')[1]
#                     f_name = volid.split()[0]
#                     mths = contract_mths[f_name]
#                     ordering = find_cdist(curr_sym, f_mth, mths)
#                     # print('ordering inputs: ', curr_sym, f_mth)
#                     u_name = volid.split('.')[0]
#                     f_price = pricedata[(pricedata['value_date'] == sim_start) &
#                                         (pricedata['underlying_id'] == u_name)]['settle_value'].values[0]
#                     # print('PRICE AND DATE UNDERLYING: ', sim_start, f_price)
#                     # print('VOL AND DATE: ', sim_start, vol)
#                     underlying = Future(
#                         f_mth, f_price, f_name, ordering=ordering)
#                     opt = Option(strike, tau, char, vol, underlying,
#                                  payoff, shorted=shorted, month=opmth, direc=direc, barrier=barriertype,
#                                  bullet=bullet, ki=ki, ko=ko, ordering=ordering)
#                     oplist[flag].append(opt)
#                     # pf.add_security(opt, flag)

#                 # input specifies a future
#                 elif inputs[0] == 'Future':

#                     full = inputs[1].split()
#                     product = full[0]
#                     mth = full[1]
#                     ordering = find_cdist(curr_sym, mth)
#                     price = pricedata[(pricedata['underlying_id'] == inputs[1]) &
#                                       (pricedata['value_date'] == sim_start)]['settle_value'].values[0]
#                     flag = inputs[4].strip('\n')
#                     shorted = True if inputs[4] == 'short' else False
#                     ft = Future(mth, price, product,
#                                 shorted=shorted, ordering=ordering)
#                     ftlist[flag].append(ft)
#                     # pf.add_security([ft], flag)
#     # handling bullet options
#     bullets = handle_dailies(oplist)
#     # print('bullet list:', len(bullets['OTC']))
#     for flag in bullets:
#         ops = oplist[flag]
#         pf.add_security(ops, flag)
#         # for op in ops:
#         #     pf.add_security(op, flag)
#     for flag in ftlist:
#         fts = ftlist[flag]
#         pf.add_security(fts, flag)

#     elapsed = time.time() - t
#     # print('[PREP_PORTFOLIO] elapsed: ', elapsed)
#     return pf


def prep_portfolio(voldata, pricedata, filepath='specs.csv'):
    """Constructs the portfolio from the requisite CSV file that specifies the details of 
    each security in the portfolio.

    Args:
        voldata (TYPE): volatility data
        pricedata (TYPE): price data
        filepath (TYPE): path to the csv containing portfolio specifications

    Returns:
        TYPE: portfolio object 

    """

    # initializing variables
    oplist = {'hedge': [], 'OTC': []}
    ftlist = {'hedge': [], 'OTC': []}
    sim_start = min(min(voldata.value_date), min(pricedata.value_date))
    sim_start = pd.to_datetime(sim_start)
    t = time.time()
    pf = Portfolio()
    curr_mth = dt.date.today().month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = dt.date.today().year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    # reading in the dataframe of portfolio values
    specs = pd.read_csv(filepath)
    specs = specs.fillna('None')
    # constructing each object individually
    for i in range(len(specs)):
        data = specs.iloc[i]
        if data.Type == 'Future':
            # future case
            full = data.vol_id.split()
            product = full[0]
            mth = full[1]
            lst = contract_mths[product]
            ordering = find_cdist(curr_sym, mth, lst)
            price = pricedata[(pricedata['underlying_id'] == data.vol_id) &
                              (pricedata['value_date'] == sim_start)]['settle_value'].values[0]
            flag = data.hedgeorOTC
            lots = 1000 if data.lots == 'None' else int(data.lots)
            shorted = True if data.shorted else False
            ft = Future(mth, price, product, shorted=shorted,
                        lots=lots, ordering=ordering)
            ftlist[flag].append(ft)

        elif data.Type == 'Option':
            strike = float(data.strike)
            volid = str(data.vol_id)
            opmth = volid.split()[1].split('.')[0]
            char = str(data.call_put_id)
            volflag = 'C' if char == 'call' else 'P'

            # get tau from data
            tau = voldata[(voldata['value_date'] == sim_start) &
                          (voldata['vol_id'] == volid) &
                          (voldata['call_put_id'] == volflag)]['tau'].values[0]
            # get vol from data
            try:
                vol = voldata[(voldata['vol_id'] == volid) &
                              (voldata['call_put_id'] == volflag) &
                              (voldata['value_date'] == sim_start) &
                              (voldata['strike'] == strike)]['settle_vol'].values[0]
            except IndexError:
                print('vol_id: ', volid)
                print('call_put_id: ', volflag)
                print('value_date: ', sim_start)
                print('strike: ', strike)
                raise ValueError('voldata cannot be located!')
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
            # lots
            lots = 1000 if data.lots == 'None' else int(data.lots)

            # handle underlying construction
            f_mth = volid.split()[1].split('.')[1]
            f_name = volid.split()[0]
            mths = contract_mths[f_name]
            ordering = find_cdist(curr_sym, f_mth, mths)
            # print('ordering inputs: ', curr_sym, f_mth)
            u_name = f_name + '  ' + volid.split('.')[1]
            try:
                f_price = pricedata[(pricedata['value_date'] == sim_start) &
                                    (pricedata['underlying_id'] == u_name)]['settle_value'].values[0]
            except IndexError:
                print('vol_id: ', volid)
                print('f_name: ', f_name)
                print('value_date: ', sim_start)
                print('underlying_id: ', u_name)

            underlying = Future(f_mth, f_price, f_name, ordering=ordering)
            opt = Option(strike, tau, char, vol, underlying,
                         payoff, shorted=shorted, month=opmth, direc=direc,
                         barrier=barriertype, lots=lots, bullet=bullet,
                         ki=ki, ko=ko, ordering=ordering)
            oplist[flag].append(opt)

    # handling bullet options
    bullets = handle_dailies(oplist)
    for flag in bullets:
        ops = oplist[flag]
        pf.add_security(ops, flag)

    for flag in ftlist:
        fts = ftlist[flag]
        pf.add_security(fts, flag)

    # elapsed = time.time() - t
    # print('[PREP_PORTFOLIO] elapsed: ', elapsed)
    return pf


def handle_dailies(dic):
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
                strike, char, vol, underlying, payoff, shorted, month, ordering, lots \
                    = params['strike'], params['char'], params['vol'], params['underlying'], \
                    params['payoff'],  params['shorted'], params['month'], \
                    params['ordering'], params['lots']
                # barrier params
                direc, barrier, ki, ko, rebate = \
                    params['direc'], params['barrier'], params[
                        'ki'], params['ko'], params['rebate']

                # creating the bullets corresponding to this daily option.
                for i in range(1, ttm_range+1):
                    tau = i/365
                    assert tau > 0
                    op_i = Option(strike, tau, char, vol, underlying,
                                  payoff, shorted, month, direc=direc, barrier=barrier, lots=lots,
                                  bullet=False, ki=ki, ko=ko, rebate=rebate, ordering=ordering)
                    bullets.append(op_i)
            lst.extend(bullets)

    return dic


###############################################################
################### Data Cleaning Functions ###################
###############################################################


def clean_data(df, flag, edf=None):
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
        > assigns preliminary ordering (i.e. c1, c2 months from current month). This step involves another function civols (line 479)
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
    # cleaning expiry data
    if flag == 'exp':
        # cleaning expiry data, handling datatypes
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        df = df[(df['year'] > 10)]
        s = df['opmth'].copy()
        # taking years mod 10, i.e. S17 --> S7.
        df.ix[:, 'opmth'] = s.str[0] + \
            (pd.to_numeric(s.str[1:]) % 10).astype(str)

    # cleaning volatility data
    elif flag == 'vol':
        # handling data types
        df['value_date'] = pd.to_datetime(df['value_date'])
        df = df.dropna()
        # calculating time to expiry from vol_id
        df = ttm(df, df['vol_id'], edf)
        df = df[df.tau > 0].dropna()
        # generating additional identifying fields.
        df['underlying_id'] = df[
            'vol_id'].str.split().str[0] + '  ' + df['vol_id'].str.split('.').str[1]
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['ftmth'] = df['underlying_id'].str.split().str[1]
        df['op_id'] = df['op_id'] = df.vol_id.str.split().str[
            1].str.split('.').str[0]

        # transformative functions
        df = assign_ci(df)
        try:
            df['label'] = df['vol_id'] + ' ' + \
                df['order'].astype(str) + ' ' + df.call_put_id
        except TypeError:
            print(df.vol_id[0])
            print('vol_id type: ', type(df.vol_id[0]))
            print('order type: ', type(df.order[0]))

    # cleaning price data
    elif flag == 'price':
        # dealing with datatypes and generating new fields from existing ones.
        df['value_date'] = pd.to_datetime(df['value_date'])
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['ftmth'] = df['underlying_id'].str.split().str[1]
        # transformative functions.
        df = get_expiry(df, edf)
        df = assign_ci(df)
        df = scale_prices(df)
        df = df.fillna(0)
        df.expdate = pd.to_datetime(df.expdate)
        df = df[df.value_date <= df.expdate]

    df.reset_index(drop=True, inplace=True)
    df = df.dropna()
    # df.to_csv('datasets/cleaned_' + flag + '.csv', index=False)

    return df


def ciprice(pricedata, rollover='opex'):
    """Constructs the CI price series.

    Args:
        pricedata (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : Dataframe with the following columns:
            Product | date | underlying | order | settle_value | returns | expiry date
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
            most_recent = []
            by_date = None
            try:
                relevant_dates = ro_dates[product]
            except KeyError:
                df2 = df[
                        ['pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns', 'expdate']]
                df2.columns = [
                    'pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns', 'expdate']
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
                        ['pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns', 'expdate']]
                    # print(tdf.empty)
                    tdf.columns = [
                        'pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns', 'expdate']
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
    """Takes in a dataframe of vols and prices (same format as those returned by read_data),
     and generates delta-wise vol organized hierarchically by date, underlying and vol_id. Uses piecewise-cubic Hermite Interpolation to interpolate delta-vol curve.

    Args:
        voldata (TYPE): dataframe of vols
        pricedata (TYPE): dataframe of prices

    Returns:
        pandas dataframe: dataframe with following columns:
            Product| Date | underlying | TTM | vol_id | order | 5D | ... | 95D
    """
    t = time.time()
    relevant_price = pricedata[
        ['underlying_id', 'value_date', 'settle_value', 'order']]
    relevant_vol = voldata[['value_date', 'vol_id', 'strike',
                            'call_put_id', 'tau', 'settle_vol', 'underlying_id']]

    print('merging')
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['value_date', 'underlying_id'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['settle_vol'] > 0)]

    print('computing deltas')
    merged['delta'] = merged.apply(compute_delta, axis=1)
    # merged.to_csv('merged.csv')
    merged['pdt'] = merged['underlying_id'].str.split().str[0]

    print('getting labels')
    # getting labels for deltas
    delta_vals = np.arange(0.05, 1, 0.05)
    delta_labels = [str(int(100*x)) + 'd' for x in delta_vals]
    # all_cols = ['underlying_id', 'tau', 'vol_id'].extend(delta_labels)

    print('preallocating')
    # preallocating dataframes
    call_df = merged[merged.call_put_id == 'C'][
        ['pdt', 'value_date', 'underlying_id', 'tau', 'vol_id', 'order']].drop_duplicates()
    put_df = merged[merged.call_put_id == 'P'][
        ['pdt', 'value_date', 'underlying_id', 'tau', 'vol_id', 'order']].drop_duplicates()

    # adding option month as a column
    c_pdt = call_df.vol_id.str.split().str[0]
    c_opmth = call_df.vol_id.str.split().str[1].str.split('.').str[0]
    c_fin = c_pdt + ' ' + c_opmth
    call_df['op_id'] = c_fin
    p_pdt = put_df.vol_id.str.split().str[0]
    p_opmth = put_df.vol_id.str.split().str[1].str.split('.').str[0]
    p_fin = p_pdt + ' ' + p_opmth
    put_df['op_id'] = p_fin

    # appending rest of delta labels as columns.
    call_df = pd.concat([call_df, pd.DataFrame(columns=delta_labels)], axis=1)
    put_df = pd.concat([put_df, pd.DataFrame(columns=delta_labels)], axis=1)
    products = merged.pdt.unique()

    print('beginning iteration:')

    for pdt in products:
        tmp = merged[merged.pdt == pdt]
        # tmp.to_csv('test.csv')
        dates = tmp.value_date.unique()
        vids = tmp.vol_id.unique()
        for date in dates:
            for vid in vids:
                # filter by vol_id and by day.
                df = tmp[(tmp.value_date == date) & (tmp.vol_id == vid)]
                calls = df[df.call_put_id == 'C']
                puts = df[df.call_put_id == 'P']
                # setting absolute value.
                puts.delta = np.abs(puts.delta)
                # sorting in ascending order of delta for interpolation
                # purposes
                calls = calls.sort_values(by='delta')
                puts = puts.sort_values(by='delta')
                # reshaping data for interpolation.
                drange = np.arange(0.05, 1, 0.05)
                cdeltas = calls.delta.values
                cvols = calls.settle_vol.values
                pdeltas = puts.delta.values
                pvols = puts.settle_vol.values
                # interpolating delta using Piecewise Cubic Hermite
                # Interpolation (Pchip)
                try:
                    f1 = PchipInterpolator(cdeltas, cvols, axis=1)
                    f2 = PchipInterpolator(pdeltas, pvols, axis=1)
                except IndexError:
                    continue
                # grabbing delta-wise vols based on interpolation.
                call_deltas = f1(drange)
                put_deltas = f2(drange)

                try:
                    call_df.loc[(call_df.vol_id == vid) & (call_df.value_date == date),
                                delta_labels] = call_deltas
                except ValueError:
                    print('target: ', call_df.loc[(call_df.vol_id == vid) & (
                        call_df.value_date == date), delta_labels])
                    print('values: ', call_deltas)

                try:
                    put_df.loc[(put_df.vol_id == vid) & (put_df.value_date == date),
                               delta_labels] = put_deltas
                except ValueError:
                    print('target: ', call_df.loc[(call_df.vol_id == vid) & (
                        call_df.value_date == date), delta_labels])
                    print('values: ', call_deltas)

    print('Done. writing to csv...')
    # call_df.to_csv('call_deltas.csv', index=False)
    # put_df.to_csv('put_deltas.csv', index=False)

    # resetting indices
    call_df.reset_index(drop=True, inplace=True)
    put_df.reset_index(drop=True, inplace=True)
    elapsed = time.time() - t
    print('[vol_by_delta] elapsed: ', elapsed)
    return call_df, put_df


def civols(vdf, pdf, rollover='opex'):
    """Constructs the CI vol series.
    Args:
        vdf (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : dataframe with the following columns:
        pdt|order|value_date|underlying_id|vol_id|op_id|call_put_id|tau|strike|settle_vol

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
                          'op_id', 'call_put_id', 'tau', 'strike', 'settle_vol']]
                df2.columns = ['pdt', 'order', 'value_date', 'underlying_id',
                               'vol_id', 'op_id', 'call_put_id', 'tau', 'strike', 'settle_vol']
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

                    df2 = df[df.order == ordering]
                    tdf = df2[(df2['value_date'] < date) & (df2['value_date'] >= breakpoint)][
                        ['pdt', 'order', 'value_date', 'underlying_id', 'vol_id', 'op_id', 'call_put_id', 'tau', 'strike', 'settle_vol']]
                    # renaming columns
                    tdf.columns = ['pdt', 'order', 'value_date', 'underlying_id',
                                   'vol_id', 'op_id', 'call_put_id', 'tau', 'strike', 'settle_vol']
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
        # print('Expdate: ', expdate)
        try:
            expdate = expdate.values[0]
        except IndexError:
            print('Vol ID: ', iden)
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


def assign_ci(df):
    """Identifies the continuation numbers of each underlying.

    Args:
        df (Pandas Dataframe): Dataframe of price data, in the same format as that returned by read_data.

    Returns:
        Pandas dataframe     : Dataframe with the CIs populated.
    """
    today = dt.date.today()
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
            dist = find_cdist(m1, ftmth, lst)
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
        else:
            # print('case 3')
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


# def generate_mappings(pricedata):
#     mappings = {}
#     products = pricedata['pdt'].unique()
#     curr_mth = dt.date.today().month
#     for product in products:
#         mths = contract_mths[product]
#     pass


def scale_prices(pricedata):
    """Converts price data into returns, by applying log(curr/prev). Treats each underlying security by itself so as to avoid taking the quotient of two different securities.

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
        s = df['settle_value']
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
    """Generates dictionary of form {product: [c1 rollover, c2 rollover, ...]}. If ci rollover is 0, then no rollover happens.

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
    s = x.settle_value
    K = x.strike
    tau = x.tau
    char = x.call_put_id
    vol = x.settle_vol
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
