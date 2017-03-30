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
from scripts.prep_data import read_data
import pandas as pd
import calendar
import datetime as dt
import ast
import sys
import traceback
import numpy as np
import scipy
import math
import time 

'''
TODO:  2) read in multipliers from csv
'''


# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.
filepath = 'portfolio_specs.txt'

vdf, pdf, edf = read_data(filepath)

# composite label that has product, opmth, cont.
vdf['label'] = vdf['vol_id'] + ' ' + vdf['cont'].astype(str) + ' ' + vdf.call_put_id


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


# TODO: add in some kind of day counter that syncs up with the prices -> can use to track rollover dates as well.
def civols(vdf, pdf,rollover='opex'):
    """Scales volatility surfaces and associates them with a product and an ordering number (ci).

    Args:
        vdf (TYPE): vol dataframe of same form as the one returned by read_data
        pdf (TYPE): price dataframe of same form as the one returned by read_data
        rollover (None, optional): rollover logic; defaults to 'opex' (option expiry.)

    Returns:
        TYPE: Description
    """
    # label = composite index that displays 1) Product 2) opmth 3) cond number.
    t = time.time()
    labels = vdf.label.unique()
    retDF = vdf.copy()
    retDF['vol change'] = ''
    ret = None
    for label in labels:
        df = vdf[vdf.label == label]
        dates = sorted(df['value_date'].unique())
        # df.reset_index(drop=True, inplace=True)
        for i in range(len(dates)):
            # first date in this label-df
            try:
                date = dates[i]
                if i == 0:
                    dvol = 0
                else:
                    prevdate = dates[i-1] 
                    prev_atm_price = pdf[(pdf['value_date'] == prevdate)]['settle_value'].values[0]
                    curr_atm_price = pdf[(pdf['value_date'] == date)]['settle_value'].values[0]
                    # calls
                    curr_vol_surface = df[(df['value_date'] == date)][['strike','settle_vol']]
                    # print(curr_vol_surface)
                    if curr_vol_surface.empty:
                        print('CURR SURF EMPTY')
                    prev_vol_surface = df[(df['value_date'] == prevdate)][['strike','settle_vol']]
                    # print(prev_vol_surface)
                    if prev_vol_surface.empty:
                        print('PREV VOL SURF EMPTY')
                    # round strikes up/down to nearest 10.                
                    curr_atm_vol = curr_vol_surface.loc[(curr_vol_surface['strike'] == (round(curr_atm_price/10) * 10)), 'settle_vol']
                    if curr_atm_vol.empty:
                        print('ATM EMPTY. BREAKING.')
                    curr_atm_vol = curr_atm_vol.values[0]
                    if np.isnan(curr_atm_vol):
                        print('ATM VOL IS NAN')
                    prev_atm_vol = prev_vol_surface.loc[(prev_vol_surface['strike'] == (round(prev_atm_price/10) * 10)), 'settle_vol']
                    if prev_atm_vol.empty:
                        print('PREV SURF EMPTY')
                    prev_atm_vol = prev_atm_vol.values[0]
                    if np.isnan(prev_atm_vol):
                        print('PREV VOL IS NAN')
                    dvol = curr_vol_surface['settle_vol'] - prev_atm_vol
                    # print('Diff: ', diff)
                retDF.ix[(retDF.label == label) & (retDF['value_date'] == date), 'vol change'] = dvol 
            except (IndexError):
                print('Label: ', label)
                print('Index: ', index)
                print('product: ', product)
                print('cont: ', cont)
                print('idens: ', mth)
        # assign each vol surface to an appropriately named column in a new dataframe.
        product = label[0]
        call_put_id = label[-1]
        # FIXME: year-long expiries, i.e. Z6.Z7
        opmth = label.split('.')[0].split()[1][0]
        ftmth = label.split('.')[1].split()[0][0]
        cont =  int(label.split('.')[1].split()[1])
        mthlist = contract_mths[product]
        dist = find_cdist(opmth, ftmth, mthlist)
        # column is of the format: product_c(opdist)(cont)_callorput
        vals = retDF[retDF.label==label][['strike', 'vol change']]
        vals.reset_index(drop=True, inplace=True)
        vals.columns = ['strike' , product + '_c' + str(cont) + '_' + str(dist) + '_' + call_put_id]
        ret = vals if ret is None else pd.concat([ret, vals], axis = 1)


    elapsed = time.time() - t
    print('[CIVOLS] Time Elapsed: ', elapsed)
    return ret



def find_cdist(x1, x2, lst):
    """Given two symbolic months (e.g. N7 and Z7), identifies the ordering of the month (c1, c2, etc.)
    
    Args:
        x1 (TYPE): current month
        x2 (TYPE): target month
        lst (TYPE): list of contract months for this product.
    
    Returns:
        int: ordering
    """
    # case 1: month is a contract month.
    if x1 in lst:
        dist = (lst.index(x1) - lst.index(x2)) % len(lst)
    # case 2: month is NOT a contract month. C1 would be nearest contract month. 
    else:
        mthvals = [sym_to_month[x] for x in lst]
        mthvals.append(sym_to_month[x1])
        mthvals = sorted(mthvals)
        dist = mthvals.index(x2) - mthvals.index(x1)
    return dist

# label = 'C  N7.Z7 4 C'
# df = vdf[vdf.label == label]
# dates = sorted(df.value_date.unique())
# d1 = dates[0]
# d2 = dates[1]
# prev_atm_price = pdf[(pdf['value_date'] == d1)]['settle_value'].values[0]
# curr_atm_price = pdf[(pdf['value_date'] == d2)]['settle_value'].values[0]
# curr_vol_surface = df[(df['value_date'] == d2)][['strike','settle_vol']]
# prev_vol_surface = df[(df['value_date'] == d1)][['strike','settle_vol']]
# prev_atm_vol = prev_vol_surface.loc[(prev_vol_surface['strike'] == (round(prev_atm_price/10) * 10)), 'settle_vol']
# prev_atm_vol = prev_atm_vol.values[0]
# dvol = curr_vol_surface['settle_vol'] - prev_atm_vol