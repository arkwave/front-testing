"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 30/3/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""

# # Imports
from . import portfolio
from . import classes
import pandas as pd
import datetime as dt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from math import log, sqrt 
import time
seed = 7
np.random.seed(seed)
'''
TODO   read in multipliers from csv
'''

# setting pandas warning levels
pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.
filepath = 'portfolio_specs.txt'

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


def read_data(filepath):
    """
    Summary: Reads in the relevant data files specified in portfolio_specs.txt, which is specified by filepath.
    """
    t = time.time()
    with open(filepath) as f:
        try:
            # get paths
            volpath = f.readline().strip('\n')
            pricepath = f.readline().strip('\n')
            expath = f.readline().strip('\n')
            # import dataframes
            volDF = pd.read_csv(volpath)
            priceDF = pd.read_csv(pricepath)
            edf = pd.read_csv(expath)
            # clean dataframes
            edf = clean_data(edf, 'exp')
            volDF = clean_data(volDF, 'vol', edf=edf)
            priceDF = clean_data(priceDF, 'price', edf=edf)

        except FileNotFoundError:
            print(volpath)
            print(pricepath)
            print(expath)
            import os
            print(os.getcwd())
    elapsed = time.time() - t
    print('[READ_DATA] elapsed: ', elapsed)
    return volDF, priceDF, edf


def prep_portfolio(voldata, pricedata, sim_start):
    """
    Reads in portfolio specifications from portfolio_specs.txt and constructs a portfolio object. The paths to the dataframes are specified in the first 3 lines of portfolio_specs.txt, while the remaining securities to be added into this portfolio are stored in the remaining lines. By design, all empty lines or lines beginning with %% are ignored.

    Args:
        voldata (pandas dataframe)  : dataframe containing the volatility surface (i.e. strike-wise volatilities)
        pricedata (pandas dataframe): dataframe containing the daily price of underlying.
        sim_start (pandas dataframe): start date of the simulation. defaults to the earliest date in the dataframes.

    Returns:
        pf (Portfolio)              : a portfolio object.
    """
    t = time.time()
    pf = portfolio.Portfolio()
    curr_mth = dt.date.today().month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = dt.date.today().year % 2000 + decade
    curr_sym = curr_mth_sym + str(curr_yr)
    with open(filepath) as f:
        for line in f:
            if "%%" in line or line in ['\n', '\r\n']:
                continue
            else:
                inputs = line.split(',')
                # input specifies an option
                if inputs[0] == 'Option':
                    strike = float(inputs[1])
                    volid = str(inputs[2])
                    opmth = volid.split()[1].split('.')[0]
                    char = inputs[3]
                    volflag = 'C' if char == 'call' else 'P'

                    # get tau from data
                    tau = voldata[(voldata['value_date'] == sim_start) &
                                  (voldata['vol_id'] == volid) &
                                  (voldata['call_put_id'] == volflag)]['tau'].values[0]
                    # get vol from data
                    vol = voldata[(voldata['vol_id'] == volid) &
                                  (voldata['call_put_id'] == volflag) &
                                  (voldata['value_date'] == sim_start) &
                                  (voldata['strike'] == strike)]['settle_vol'].values[0]

                    payoff = str(inputs[4])

                    barriertype = None if inputs[
                        5] == 'None' else str(inputs[5])
                    direc = None if inputs[6] == 'None' else str(inputs[6])
                    ki = None if inputs[7] == 'None' else int(inputs[7])
                    ko = None if inputs[8] == 'None' else int(inputs[8])
                    bullet = True if inputs[9] == 'True' else False
                    flag = str(inputs[11]).strip('\n')
                    shorted = True if inputs[10] == 'short' else False

                    # handle underlying construction
                    f_mth = volid.split()[1].split('.')[1]
                    f_name = volid.split()[0]
                    mths = contract_mths[f_name]                
                    ordering = find_cdist(curr_sym, f_mth, mths)
                    u_name = volid.split('.')[0]
                    f_price = pricedata[(pricedata['value_date'] == sim_start) &
                                        (pricedata['underlying_id'] == u_name)]['settle_value'].values[0]
                    underlying = classes.Future(f_mth, f_price, f_name, ordering=ordering)
                    opt = classes.Option(strike, tau, char, vol, underlying,
                                         payoff, shorted=shorted, month=opmth, direc=direc, barrier=barriertype,
                                         bullet=bullet, ki=ki, ko=ko, ordering=ordering)
                    pf.add_security(opt, flag)

                # input specifies a future
                elif inputs[0] == 'Future':

                    full = inputs[1].split()
                    product = full[0]
                    mth = full[1]
                    ordering = find_cdist(curr_sym, mth)
                    price = pricedata[(pricedata['underlying_id'] == inputs[1]) &
                                      (pricedata['value_date'] == sim_start)]['settle_value'].values[0]
                    flag = inputs[4].strip('\n')
                    shorted = True if inputs[4] == 'short' else False
                    ft = classes.Future(mth, price, product, shorted=shorted, ordering=ordering)

                    pf.add_security(ft, flag)
                    
    elapsed = time.time() - t
    print('[PREP_PORTFOLIO] elapsed: ', elapsed)
    return pf



def clean_data(df, flag, edf=None):
    """Function that cleans the dataframes passed into it by:
    1) dropping NaN entries
    2) converting dates to datetime objects
    3) In the case of the vol dataframe, reads in the vol_id and computes the time to expiry.
    Args:
        df (pandas dataframe)   : the dataframe to be cleaned.
        flag (pandas dataframe) : determines which dataframe is being processed.
        edf (pandas dataframe)  : dataframe containing the expiries of options.

    Returns:
        TYPE: Description
    """
    # cleaning expiry data
    if flag == 'exp':
        # cleaning expiry data
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        df = df[(df['year'] > 10)]
        s = df['opmth'].copy()
        # df['opmth'] = ''
        df.ix[:, 'opmth'] = s.str[0] + \
            (pd.to_numeric(s.str[1:]) % 10).astype(str)

    # cleaning volatility data
    elif flag == 'vol':
        df['value_date'] = pd.to_datetime(df['value_date'])
        df = df.dropna()
        # calculating time to expiry
        df = ttm(df, df['vol_id'], edf)

        # generating additional identifying fields.
        df['underlying_id'] = df[
            'vol_id'].str.split().str[0] + '  ' + df['vol_id'].str.split('.').str[1]

        df['pdt'] = df['underlying_id'].str.split().str[0]

        # df['contract_mth'] = df['underlying_id'].str.split().str[1].str[0]
        # df['contract_yr'] = pd.to_numeric(
        # df['underlying_id'].str.split().str[1].str[1])

        df['ftmth'] = df['underlying_id'].str.split().str[1]
        # transformative functions
        df = assign_ci(df)
        # df.to_csv('debug_assignment_ci.csv')
        try:
            df['label'] = df['vol_id'] + ' ' + \
                df['cont'].astype(str) + ' ' + df.call_put_id
        except TypeError:
            print(df.vol_id[0])
            print('vol_id type: ', type(df.vol_id[0]))
            print('cont type: ', type(df.cont[0]))

    # cleaning price data
    elif flag == 'price':
        df['value_date'] = pd.to_datetime(df['value_date'])
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['ftmth'] = df['underlying_id'].str.split().str[1]
        # df['contract_mth'] = df['underlying_id'].str.split().str[1].str[0]
        # df['contract_yr'] = pd.to_numeric(
        #     df['underlying_id'].str.split().str[1].str[1])
        # transformative functions.
        df = get_expiry(df, edf)
        df = assign_ci(df)
        df = scale_prices(df)

    df.reset_index(drop=True, inplace=True)
    df = df.dropna()
    df.to_csv('datasets/cleaned_' + flag + '.csv', index=False)

    return df


def ttm(df, s, edf):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
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
    curr_mth = month_to_sym[today.month]
    curr_yr = today.year
    products = df['pdt'].unique()
    df['cont'] = ''
    for pdt in products:
        lst = contract_mths[pdt]
        df2 = df[df.pdt == pdt]
        ftmths = df2.ftmth.unique()
        for ftmth in ftmths:
            m1 = curr_mth + str(curr_yr % (2000 + decade))
            dist = find_cdist(m1, ftmth, lst)
            df.ix[(df.pdt == pdt) & (df.ftmth == ftmth), 'cont'] = dist
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


def generate_mappings(pricedata):
    mappings = {}
    products = pricedata['pdt'].unique()
    curr_mth = dt.date.today().month
    for product in products:
        mths = contract_mths[product]
    pass


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
        ret = np.log(s1/s)
        pricedata.ix[
            (pricedata['underlying_id'] == x), 'returns'] = ret
    pricedata = pricedata.dropna()
    return pricedata


def get_expiry(pricedata, edf, rollover=None):
    """Summary

    Args:
        pricedata (TYPE): Description
        edf (TYPE): Description
        rollover (None, optional): Description

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
        conts = sorted(pricedata['cont'].unique())
        rollover_dates[product] = [0] * len(conts)
        for i in range(len(conts)):
            cont = conts[i]
            df2 = df[df['cont'] == cont]
            test = df2[df2['value_date'] > df2['expdate']]['value_date']
            if not test.empty:
                try:
                    rollover_dates[product][i] = min(test)
                except (ValueError, TypeError):
                    print('i: ', i)
                    print('cont: ', cont)
                    print('min: ', min(test))
                    print('product: ', product)
            else:
                expdate = df2['expdate'].unique()[0]
                rollover_dates[product][i] = pd.Timestamp(expdate)
    return rollover_dates


def ciprice(pricedata, rollover='opex'):
    """Constructs the CI price series.

    Args:
        pricedata (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : prices arranged according to c_i indexing.
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
            conts = sorted(df['cont'].unique())
            most_recent = []
            by_date = None
            relevant_dates = ro_dates[product]
            # iterate over rollover dates for this product.
            for date in relevant_dates:
                # print('most_recent: ', most_recent)
                # print('date: ', date)
                breakpoint = max(most_recent) if most_recent else min(
                    df['value_date'])
                by_cont = None
                # iterate over all conts for this product. for each cont, grab
                # entries until first breakpoint, and stack wide.
                for cont in conts:
                    df2 = df[df.cont == cont]
                    tdf = df2[(df2['value_date'] < date) & (df2['value_date'] >= breakpoint)][
                        ['value_date', 'returns']]
                    tdf.columns = ['value_date', product + '_c' + str(cont)]
                    tdf.reset_index(drop=True, inplace=True)
                    by_cont = tdf if by_cont is None else pd.concat(
                        [by_cont, tdf], axis=1)
                # by_date contains entries from all conts until current
                # rollover date. take and stack this long.
                by_date = by_cont if by_date is None else pd.concat(
                    [by_date, by_cont])
                most_recent.append(date)
            by_product = by_date if by_product is None else pd.concat(
                [by_product, by_cont], axis=1)
        by_product.dropna(inplace=True)
        by_product = by_product.loc[:, ~by_product.columns.duplicated()]
        # by_product.to_csv('by_product_debug.csv', index=False)
        final = by_product
    else:
        final = -1
    elapsed = time.time() - t
    print('[CI-PRICE] elapsed: ', elapsed)
    return final


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


def vol_by_delta(voldata, pricedata):
    """takes in a dataframe of vols and prices (same format as those returned by read_data),
     and generates delta-wise vol organized hierarchically by date, underlying and vol_id
    
    Args:
        voldata (TYPE): dataframe of vols
        pricedata (TYPE): dataframe of prices
    
    Returns:
        pandas dataframe: delta-wise vol of each option.
    """
    relevant_price = pricedata[['underlying_id', 'value_date', 'settle_value', 'cont']]
    relevant_vol = voldata[['value_date', 'vol_id', 'strike','call_put_id', 'tau', 'settle_vol', 'underlying_id']]

    print('merging')
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['value_date', 'underlying_id'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['settle_vol'] > 0)]

    print('computing deltas')
    merged['delta'] = merged.apply(compute_delta, axis=1)
    merged.to_csv('merged.csv')
    merged['pdt'] = merged['underlying_id'].str.split().str[0]

    print('getting labels')
    # getting labels for deltas
    delta_vals = np.arange(0.05, 1, 0.05)
    delta_labels = [str(int(100*x)) + 'd' for x in delta_vals]
    # all_cols = ['underlying_id', 'tau', 'vol_id'].extend(delta_labels)

    print('preallocating')
    # preallocating dataframes
    call_df = merged[merged.call_put_id == 'C'][
        ['value_date', 'underlying_id', 'tau', 'vol_id', 'cont', 'pdt']].drop_duplicates()
    put_df = merged[merged.call_put_id == 'P'][
        ['value_date', 'underlying_id', 'tau', 'vol_id', 'cont', 'pdt']].drop_duplicates()

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
        tmp.to_csv('test.csv')
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
                # sorting in ascending order of delta for interpolation purposes
                calls = calls.sort_values(by='delta')
                puts = puts.sort_values(by='delta')
                # reshaping data for interpolation.
                drange = np.arange(0.05, 1, 0.05)
                cdeltas = calls.delta.values
                cvols = calls.settle_vol.values
                pdeltas = puts.delta.values
                pvols = puts.settle_vol.values
                # interpolating delta using Piecewise Cubic Hermite Interpolation (Pchip)
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
                    print('target: ', call_df.loc[(call_df.vol_id == vid) & (call_df.value_date == date), delta_labels])
                    print('values: ', call_deltas)

                try:
                    put_df.loc[(put_df.vol_id == vid) & (put_df.value_date == date), 
                           delta_labels] = put_deltas
                except ValueError:
                    print('target: ', call_df.loc[(call_df.vol_id == vid) & (call_df.value_date == date), delta_labels])
                    print('values: ', call_deltas)               

    print('Done. writing to csv...')
    # call_df.to_csv('call_deltas.csv', index=False)
    # put_df.to_csv('put_deltas.csv', index=False)    

    # resetting indices
    call_df.reset_index(drop=True, inplace=True)
    put_df.reset_index(drop=True, inplace=True)
    return call_df, put_df


def civols(vdf, pdf, rollover='opex'):
    """Constructs the CI price series.

    Args:
        vdf (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : prices arranged according to c_i indexing.
    """
    t = time.time()
    if rollover == 'opex':
        ro_dates = get_rollover_dates(pdf)
        products = vdf['pdt'].unique()
        # iterate over produts
        by_product = None
        for product in products:
            df = vdf[vdf.pdt == product]
            # lst = contract_mths[product]
            conts = sorted(df['cont'].unique())
            most_recent = []
            by_date = None
            relevant_dates = ro_dates[product]
            print('rollover dates: ', relevant_dates)
            # iterate over rollover dates for this product.
            for date in relevant_dates:
                # print('most_recent: ', most_recent)
                # print('date: ', date)
                breakpoint = max(most_recent) if most_recent else min(
                    df['value_date'])
                by_cont = None
                # iterate over all conts for this product. for each cont, grab
                # entries until first breakpoint, and stack wide.
                for cont in conts:
                    print('cont: ', cont)
                    df2 = df[df.cont == cont]
                    tdf = df2[(df2['value_date'] < date) & (df2['value_date'] >= breakpoint)][['value_date', 'tau' , 'vol_id','underlying_id','call_put_id', 'strike', 'settle_vol']]
                    print('vol_ids: ', tdf.vol_id.unique())
                    # deriving additional columns
                    tdf['cont'] = cont 
                    tdf['pdt'] = product
                    tdf['op_id'] = tdf.vol_id.str.split().str[1].str.split('.').str[0]
                    # renaming columns
                    tdf.columns = ['value_date', 'tau','vol_id', 'underlying_id', 'call_put_id', 'strike', 'settle_vol', 'cont', 'pdt', 'op_id']
                    # tdf.reset_index(drop=True, inplace=True)
                    by_cont = tdf if by_cont is None else pd.concat(
                        [by_cont, tdf])
                # by_date contains entries from all conts until current
                # rollover date. take and stack this long.
                by_date = by_cont if by_date is None else pd.concat(
                    [by_date, by_cont])
                most_recent.append(date)
            by_product = by_date if by_product is None else pd.concat(
                [by_product, by_cont])
        by_product.dropna(inplace=True)
        # by_product = by_product.loc[:, ~by_product.columns.duplicated()]
        by_product.to_csv('by_product_debug.csv', index=False)
        final = by_product
    else:
        final = -1
    elapsed = time.time() - t
    print('[CI-TEST] elapsed: ', elapsed)
    return final


if __name__ == '__main__':
    # compute simulation start day; earliest day in dataframe.
    voldata, pricedata, edf = read_data(filepath)

    # just a sanity check, these two should be the same.
    sim_start = min(min(voldata['value_date']), min(pricedata['value_date']))
    assert (sim_start == min(voldata['value_date']))
    assert (sim_start == min(pricedata['value_date']))

    final_price = ciprice(pricedata)
    final_vols = civols(voldata, pricedata)
    call_vols, put_vols = vol_by_delta(final_vols, pricedata)

    # writing to csvs
    
