"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 28/3/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""

# # Imports
# from . import portfolio
# from . import classes
import pandas as pd
import calendar
import datetime as dt
import ast
import sys
import traceback
import numpy as np
import scipy

'''
TODO: 1) price/vol series transformation
      2) read in multipliers from csv
'''


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
    pf = portfolio.Portfolio()
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
                    u_name = volid.split('.')[0]
                    f_price = pricedata[(pricedata['value_date'] == sim_start) &
                                        (pricedata['underlying_id'] == u_name)]['settle_value'].values[0]

                    underlying = classes.Future(f_mth, f_price, f_name)
                    opt = classes.Option(strike, tau, char, vol, underlying,
                                         payoff, shorted=shorted, month=opmth, direc=direc, barrier=barriertype,
                                         bullet=bullet, ki=ki, ko=ko)
                    pf.add_security(opt, flag)

                # input specifies a future
                elif inputs[0] == 'Future':
                    full = inputs[1].split()
                    product = full[0]
                    mth = full[1]
                    price = pricedata[(pricedata['underlying_id'] == inputs[1]) &
                                      (pricedata['value_date'] == sim_start)]['settle_value'].values[0]
                    flag = inputs[4].strip('\n')
                    shorted = True if inputs[4] == 'short' else False

                    ft = classes.Future(mth, price, product, shorted=shorted)
                    pf.add_security(ft, flag)

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
        s = df['opmth']
        df['opmth'] = s.str[0] + (pd.to_numeric(s.str[1:]) % 10).astype(str)

    # cleaning volatility data
    elif flag == 'vol':
        df['value_date'] = pd.to_datetime(df['value_date'])
        df = df.dropna()
        # calculating time to expiry
        df = ttm(df, df['vol_id'], edf)
        df['underlying_id'] = df[
            'vol_id'].str.split().str[0] + '  ' + df['vol_id'].str.split('.').str[1]
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['op_mth'] = df['vol_id'].str.split('.').str[0].str.split().str[1]
        df['contract_mth'] = df['underlying_id'].str.split().str[1].str[0]
        df['contract_yr'] = pd.to_numeric(
            df['underlying_id'].str.split().str[1].str[1])
        df = assign_ci(df)

    # cleaning price data
    elif flag == 'price':
        df['value_date'] = pd.to_datetime(df['value_date'])
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['contract_mth'] = df['underlying_id'].str.split().str[1].str[0]
        df['contract_yr'] = pd.to_numeric(
            df['underlying_id'].str.split().str[1].str[1])
        df = get_expiry(df, edf)
        df = assign_ci(df)
        df = scale_prices(df)

    df = df.dropna()
    # df.to_csv('datasets/cleaned_' + flag + '.csv', index=False)

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
    curr_day = today.day
    curr_yr = today.year
    products = df['pdt'].unique()
    df['cont'] = ''
    for pdt in products:
        all_mths = contract_mths[pdt]
        # finding rightward distance.
        for mth in all_mths:
            if mth not in df['contract_mth'].values:
                continue
            dist = (all_mths.index(mth) - all_mths.index(curr_mth)) % 5
            df.ix[(df['contract_mth'] == mth) & (df['contract_yr'] == curr_yr % (2000 + decade))
                  & (df['pdt'] == 'C'), 'cont'] = dist
    return df


# def scale_vols(voldata, pricedata, flag='atm'):
#     """Scales vol data. Currently, laterally shifts the vol curves by taking atm_vol_curr - atm_vol_prev.

#     Args:
#         voldata (TYPE): Dataframe of volatilities of the form returned by read_data
# pricedata (TYPE): Dataframe of prices of the form returned by read_data

#     Returns:
#         voldata: dataframe with an additional field indicating scaled volatilities.
#     """
#     # find atm price
#     ids = voldata.vol_id.unique()
#     for iden in ids:
#         df = voldata[(voldata['vol_id'] == iden)]
#         dates = sorted(df.value_date)
#         uid = df['underlying_id'].unique()
#         for date in dates:
#             # get atm price
#             atm_price = pricedata[(pricedata.value_date == date) & (
#                 pricedata.underlying_id == uid)]['settle_value'].values[0]
#             # interpolate vol data

#             cvols = df[(df['value_date'] == date) & (df.underlying_id == uid) & (df.call_put_id == 'C')][
#                 'settle_vol']
#             pvols = df[(df['value_date'] == date) & (df.underlying_id == uid) & (df.call_put_id == 'P')][
#                 'settle_vol']

#             strikes = df[(df['value_date'] == date) & (df.underlying_id == uid)][
#                 'strike']

#     # isolate atm vol

#     # lateral scaling


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
    return rollover_dates


# FIXME: filtering will break, cont/product filtered together.
def construct_ci_price(pricedata, rollover='opex'):
    """Constructs the CI price.

    Args:
        pricedata (TYPE): price data frame of same format as read_data
        rollover (str, optional): the rollover strategy to be used. defaults to opex, i.e. option expiry.

    Returns:
        pandas dataframe : prices arranged according to c_i indexing.
    """
    tmp = None
    final = None
    if rollover == 'opex':
        ro_dates = get_rollover_dates(pricedata)

        products = pricedata['pdt'].unique()
        for product in products:
            retDF = None 
            all_mths = contract_mths[product]
            ci_num = len(all_mths)
            conts = sorted(pricedata['cont'].unique())
            most_recent = 0
            # for all underlyings corresponding to this product, in CI order.
            for cont in conts:
                df = pricedata[(pricedata.pdt == product) &
                               (pricedata['cont'] == cont)]
                rdate = ro_dates[product][cont]
                # rollover date exists.
                if rdate != 0:
                    # select items of relevance
                    breakpoint = most_recent if most_recent != 0 else min(df[
                                                                          'value_date'])
                    df = df[(df['value_date'] < rdate) & (df['value_date'] >= breakpoint)][
                        ['cont', 'settle_value', 'returns', 'value_date']]
                    most_recent = rdate
                # no rollover date.
                else:
                    df = df[['cont', 'settle_value', 'returns', 'value_date']]
                # updating tmp
                tmp = df if tmp is None else pd.concat([tmp, df])
            # now that c_1 has been established, shift to find the rest of the c_i for this product.
            tmp.reset_index(drop=True, inplace=True)
            for i in range(ci_num):
                ciseries = tmp[tmp['cont']>=i]['returns']
                ciseries.reset_index(drop=True, inplace=True)
                # ciseries = ciseries.shift(-(ciseries.isnull().sum()))
                ciseries.name = product + '_c' + str(i)
                retDF = ciseries if retDF is None else pd.concat([retDF, ciseries], axis=1)
            # names = [product + '_c' + str(i) for i in range(ci_num)]
            # retDF.names = names 
            final = retDF if final is None else pd.concat([final, retDF])
            final.to_csv('debug_final.csv')

    else:
        return -1
    return final 


def construct_ci_vols(voldata, edf, rollover=None):
    """Scales volatility surfaces and associates them with a product and an ordering number (ci).

    Args:
        pricedata (TYPE): Description
        edf (TYPE): Description
        rollover (None, optional): Description

    Returns:
        TYPE: Description
    """
    products = voldata.pdt.unique()
    for product in products:
        # filter first by cont, then by opmth
        df = voldata[voldata['pdt'] == product]
        conts = df.cont.unique()
        for cont in conts:
            df2 = voldata[voldata.pdt==product & voldata.cont=cont]
            opmths = df2.op_mth.unique()
            for mth in opmth:
                df3 = voldata[voldata.pdt==product & voldata.cont==cont & voldata.op_mth == mth]
                # now scale vols.
                # get atm vol today
                # get atm vol yesterday
                # laterally shift entire vol curve by diff = curr - yesterday.
                # append to a new dataframe similar to construct_ci_price and return. naming convention: Product ci opmth


if __name__ == '__main__':
    # compute simulation start day; earliest day in dataframe.
    voldata, pricedata, edf = read_data(filepath)

    # just a sanity check, these two should be the same.
    sim_start = min(min(voldata['value_date']), min(pricedata['value_date']))
    assert (sim_start == min(voldata['value_date']))
    assert (sim_start == min(pricedata['value_date']))
    # pf = prep_portfolio(voldata, pricedata, sim_start)
