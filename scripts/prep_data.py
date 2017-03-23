"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 23/3/2017
Python version : 3.5
Description    : Script contains methods to read-in and format data. These methods are used in simulation.py.

"""

# Imports
from .portfolio import Portfolio
from .classes import Option, Future
import pandas as pd
import calendar
import datetime as dt
import ast
import sys
import traceback

'''
TODO: 1) price/vol series transformation
'''


# useful variables.
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'L', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'L': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10
filepath = 'portfolio_specs.txt'


def read_data(filepath):
    """
    Summary: Reads in the relevant data files specified in portfolio_specs.txt
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
    Reads in portfolio specifications from portfolio_specs.txt
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
                    flag = str(inputs[10])
                    OTC = True if inputs[11].strip('\n') == 'OTC' else False

                    # handle underlying construction
                    f_mth = volid.split()[1].split('.')[1]
                    f_name = volid.split()[0]
                    u_name = volid.split('.')[0]
                    f_price = pricedata[(pricedata['value_date'] == sim_start) &
                                        (pricedata['underlying_id'] == u_name)]['settle_value'].values[0]

                    underlying = Future(f_mth, f_price, f_name)
                    opt = Option(strike, tau, char, vol, underlying,
                                 payoff, direc=direc, barrier=barriertype,
                                 bullet=bullet, ki=ki, ko=ko)
                    pf.add_security(opt, flag)

                # input specifies a future
                elif inputs[0] == 'Future':
                    full = inputs[1].split()
                    product = full[0]
                    mth = full[1]
                    price = pricedata[(pricedata['underlying_id'] == inputs[1]) &
                                      (pricedata['value_date'] == sim_start)]['settle_value'].values[0]
                    flag = inputs[3]
                    OTC = True if inputs[4] == 'OTC' else False

                    ft = Future(mth, price, product, OTC)
                    pf.add_security(ft, flag)

    return pf


def clean_data(df, flag, edf):
    df = df.dropna()
    # adjusting for datetime stuff
    df['value_date'] = pd.to_datetime(df['value_date'])
    if flag == 'vol':
        # cleaning volatility data
        df = df.dropna()
        # calculating time to expiry
        df = ttm(df, df['vol_id'], edf)
    df.to_csv('datasets/cleaned_' + flag + '.csv', index=False)
    return df


def ttm(df, s, edf=None):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
    s = s.unique()
    df['tau'] = ''
    for iden in s:
        expdate = get_expiry_date(iden, edf).values[0]
        currdate = pd.to_datetime(df[(df['vol_id'] == iden)]['value_date'])
        timedelta = (expdate - currdate).dt.days / 365
        df = df.assign(tau=timedelta)
    return df


def get_expiry_date(volid, edf):
    # handle differences in format; expiry dates are in format Z17.Z17 for
    # 2017 contracts, while vol/price data are in Z7.Z7 format.

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


if __name__ == '__main__':
    # compute simulation start day; earliest day in dataframe.
    voldata, pricedata, edf = read_data('filepath')
    # just a sanity check, these two should be the same.
    sim_start = min(min(voldata['value_date']), min(pricedata['value_date']))
    pf = prep_portfolio(voldata, pricedata, sim_start)
