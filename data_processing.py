"""
File Name      : prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 30/3/2017
Python version : 3.5
Description    : Script containing problematic code to be debugged.

"""

# # Imports
from scripts.prep_data import read_data
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt
import time
from scipy.interpolate import PchipInterpolator

'''
TODO:  2) read in multipliers from csv
'''


# initializing variables
# setting pandas warning level.
pd.options.mode.chained_assignment = None
# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10
# specifies the filepath for the read-in file.

# vdf, pdf, edf = read_data(filepath)
# composite label that has product, opmth, cont.
seed = 7
np.random.seed(seed)


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


########################################################################
############################# Functions ################################
########################################################################

def compute_delta(x):
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
    relevant_price = pricedata[
        ['pdt', 'underlying_id', 'value_date', 'settle_value', 'order']]
    relevant_vol = voldata[['pdt', 'value_date', 'vol_id', 'strike',
                            'call_put_id', 'tau', 'settle_vol', 'underlying_id']]

    print('merging')
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['pdt', 'value_date', 'underlying_id'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['settle_vol'] > 0)]

    print('computing deltas')
    merged['delta'] = merged.apply(compute_delta, axis=1)
    # merged.to_csv('merged.csv')

    print('getting labels')
    # getting labels for deltas
    delta_vals = np.arange(0.05, 1, 0.05)
    delta_labels = [str(int(100*x)) + 'd' for x in delta_vals]
    # all_cols = ['underlying_id', 'tau', 'vol_id'].extend(delta_labels)

    print('preallocating')
    # preallocating dataframes
    call_df = merged[merged.call_put_id == 'C'][
        ['pdt', 'value_date', 'underlying_id', 'order', 'tau', 'vol_id']].drop_duplicates()
    put_df = merged[merged.call_put_id == 'P'][
        ['pdt', 'value_date', 'underlying_id', 'order', 'tau', 'vol_id']].drop_duplicates()

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
    # call_df.to_csv('call_deltas_test.csv', index=False)
    # put_df.to_csv('put_deltas_test.csv', index=False)

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
            most_recent = []
            by_date = None
            relevant_dates = ro_dates[product]
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
                    # print('breakpoint, date, ordering, underlying, empty: ',
                    #       breakpoint, date, ordering, tdf.underlying_id.unique(), tdf.empty)
                    # # deriving additional columns
                    # tdf['op_id'] = tdf.vol_id.str.split().str[
                    #     1].str.split('.').str[0]
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
                # print('by_date:', by_date[
                #       by_date.order == 1].underlying_id.unique())
                most_recent.append(date)
                df.order -= 1

            by_product = by_date if by_product is None else pd.concat(
                [by_product, by_date])
            # print('by_product: ', by_product[
            #       by_product.order == 1].underlying_id.unique())
        # by_product.dropna(inplace=True)
        final = by_product
    else:
        final = -1
    elapsed = time.time() - t
    print('[CI-VOLS] elapsed: ', elapsed)
    # final.to_csv('ci_vol_final.csv', index=False)
    return final


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
            ordering = order_nums[i]
            df2 = df[df['order'] == ordering]
            test = df2[df2['value_date'] > df2['expdate']]['value_date']
            if not test.empty:
                try:
                    rollover_dates[product][i] = min(test)
                except (ValueError, TypeError):
                    print('i: ', i)
                    print('cont: ', ordering)
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
            most_recent = []
            by_date = None
            relevant_dates = ro_dates[product]
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
                        ['pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns']]
                    # print(tdf.empty)
                    tdf.columns = [
                        'pdt', 'value_date', 'underlying_id', 'order', 'settle_value', 'returns']
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
    print('[CI-PRICE] elapsed: ', elapsed)
    # final.to_csv('ci_price_final.csv', index=False)
    return final


##########################################################################
##########################################################################

if __name__ == '__main__':
    # compute simulation start day; earliest day in dataframe.
    filepath = 'portfolio_specs.txt'
    voldata, pricedata, edf = read_data(filepath)

    # just a sanity check, these two should be the same.
    sim_start = min(min(voldata['value_date']), min(pricedata['value_date']))
    assert (sim_start == min(voldata['value_date']))
    assert (sim_start == min(pricedata['value_date']))

    final_price = ciprice(pricedata)
    final_vols = civols(voldata, pricedata)
    call_vols, put_vols = vol_by_delta(final_vols, pricedata)

    # final_vols.to_csv()
    # final_price.to_csv('ci_price_final.csv', index=False)
    call_vols.to_csv('call_vols_by_delta.csv', index=False)
    put_vols.to_csv('put_vols_by_delta.csv', index=False)
