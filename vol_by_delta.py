from scripts.prep_data import read_data
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt
from scipy.interpolate import PchipInterpolator

seed = 7
np.random.seed(seed)

pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10

# specifies the filepath for the read-in file.

# composite label that has product, opmth, cont.
# vdf['label'] = vdf['vol_id'] + ' ' + \
#     vdf['cont'].astype(str) + ' ' + vdf.call_put_id


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
    relevant_price = pricedata[['underlying_id', 'value_date', 'settle_value', 'cont']]
    relevant_vol = voldata[['value_date', 'vol_id', 'strike','call_put_id', 'tau', 'settle_vol', 'underlying_id']]

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
    # iterate first over products, thenn dates for that product, followed by vol_ids in that product/date
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

    # changing call_df.tau and put_df.tau to days to expiry.
    call_df.tau = call_df.tau * 365
    put_df.tau = put_df.tau * 365
    print('Done. writing to csv...')
    # call_df.to_csv('call_deltas.csv', index=False)
    # put_df.to_csv('put_deltas.csv', index=False)    

    # resetting indices
    call_df.reset_index(drop=True, inplace=True)
    put_df.reset_index(drop=True, inplace=True)
    return call_df, put_df



if __name__ == '__main__':
    filepath = 'portfolio_specs.txt'
    vdf, pdf, edf = read_data(filepath)
    vbd_c, vbd_p = vol_by_delta(vdf, pdf)
    vbd_c.to_csv('vols_by_delta_c.csv')
    vbd_p.to_csv('vols_by_delta_p.csv')

