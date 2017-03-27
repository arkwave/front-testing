"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 23/3/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""

# Imports
# from .portfolio import Portfolio
# from .classes import Option, Future
import pandas as pd
import calendar
import datetime as dt
import ast
import sys
import traceback

'''
TODO: 1) price/vol series transformation
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
            volpath = f.readline().strip('\n')
            pricepath = f.readline().strip('\n')
            expath = f.readline().strip('\n')
            volDF = pd.read_csv(volpath)
            priceDF = pd.read_csv(pricepath)
            edf = pd.read_csv(expath)
            volDF = clean_data(volDF, 'vol', edf)
            priceDF = clean_data(priceDF, 'price', edf)
            edf = edf.dropna()
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
        pf (Portfolio)              : reads in
    """
    pf = Portfolio()
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

                    underlying = Future(f_mth, f_price, f_name)
                    opt = Option(strike, tau, char, vol, underlying,
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

                    ft = Future(mth, price, product, shorted=shorted)
                    pf.add_security(ft, flag)

    return pf


def clean_data(df, flag, edf):
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
    df = df.dropna()
    # adjusting for datetime stuff
    df['value_date'] = pd.to_datetime(df['value_date'])
    if flag == 'vol':
        # cleaning volatility data
        df = df.dropna()
        # calculating time to expiry
        df = ttm(df, df['vol_id'], edf)
    elif flag == 'price':
        # clean price data
        df['pdt'] = df['underlying_id'].str.split().str[0]
        df['contract_mth'] = df['underlying_id'].str.split().str[1].str[0]
        df['contract_yr'] = pd.to_numeric(
            df['underlying_id'].str.split().str[1].str[1])
    df.to_csv('datasets/cleaned_' + flag + '.csv', index=False)
    return df


def ttm(df, s, edf):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
    s = s.unique()
    df['tau'] = ''
    df['expdate'] = ''
    for iden in s:
        expdate = get_expiry_date(iden, edf).values[0]
        print(expdate)
        currdate = pd.to_datetime(df[(df['vol_id'] == iden)]['value_date'])
        timedelta = (expdate - currdate).dt.days / 365
        df.ix[df['vol_id'] == iden, 'tau'] = timedelta
        df.ix[df['vol_id'] == iden, 'expdate'] = pd.to_datetime(expdate)
    return df


def get_expiry_date(volid, edf):
    """Computes the expiry date of the option given a vol_id """
    target = volid.split()
    op_yr = pd.to_numeric(target[1][1]) + decade
    op_yr = op_yr.astype(str)
    un_yr = pd.to_numeric(target[1][-1]) + decade
    un_yr = un_yr.astype(str)
    op_mth = target[1][0]
    un_mth = target[1][3]
    prod = target[0]
    overall = op_mth + op_yr + '.' + un_mth + un_yr
    expdate = edf[(edf['vol_id'] == overall) & (edf['product'] == prod)][
        'expiry_date']
    expdate = pd.to_datetime(expdate)
    return expdate


def assign_ci(df):
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

if __name__ == '__main__':
    # compute simulation start day; earliest day in dataframe.
    voldata, pricedata, edf = read_data(filepath)
    # just a sanity check, these two should be the same.
    # sim_start = min(min(voldata['value_date']), min(pricedata['value_date']))
    # pf = prep_portfolio(voldata, pricedata, sim_start)
