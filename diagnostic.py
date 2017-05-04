from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
brokerage = 1
from math import log, sqrt, exp
from scipy.stats import norm
import pandas as pd
from ast import literal_eval

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'LCC': [1.2153, 10, 1, 25, 12.153],
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

filepath = 'hedging.csv'
# with open(filepath) as f:
#     try:
#         d_cond = f.readline().strip('\n')
#         g_cond = f.readline().strip('\n')
#         v_cond = f.readline().strip()
#         print(d_cond)
#         print(g_cond)
#         print(v_cond)
#     except FileNotFoundError:
#         print(filepath)


def generate_hedges(filepath=filepath):
    df = pd.read_csv(filepath)
    hedges = {}
    for i in df.index:
        row = df.iloc[i]
        # static hedging
        if row.flag == 'static':
            greek = row.greek
            hedges[greek] = [row.flag, row.cond, int(row.freq)]
        # bound hedging
        elif row.flag == 'bound':
            greek = row.greek
            hedges[greek] = [row.flag, literal_eval(row.cond), int(row.freq)]
        # percentage hedging
        elif row.flag == 'pct':
            greek = row.greek
            hedges[greek] = [row.flag, float(row.cond),
                             int(row.freq), row.subcond]

    return hedges

hedges = generate_hedges()


from scripts.prep_data import read_data
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt
from scipy.interpolate import PchipInterpolator
import time

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
#     vdf['order'].astype(str) + ' ' + vdf.call_put_id


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
    relevant_price = pricedata[
        ['underlying_id', 'value_date', 'settle_value', 'order']]
    relevant_vol = voldata[['value_date', 'vol_id', 'strike', 'order',
                            'call_put_id', 'tau', 'settle_vol', 'underlying_id']]

    print('merging')
    merged = pd.merge(relevant_vol, relevant_price,
                      on=['value_date', 'underlying_id', 'order'])
    # filtering out negative tau values.
    merged = merged[(merged['tau'] > 0) & (merged['settle_vol'] > 0)]

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
                  'order', 'pdt', 'call_put_id']].drop_duplicates()

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
                    vols = df.settle_vol.values
                    # interpolating delta using Piecewise Cubic Hermite
                    # Interpolation (Pchip)

                    try:
                        f1 = PchipInterpolator(deltas, vols, axis=1)
                    except IndexError:
                        continue
                    # grabbing delta-wise vols based on interpolation.
                    vols = f1(drange)

                    dic = dict(zip(delta_labels, vols))
                    # adding the relevant values from the indexing dataframe
                    dic['pdt'] = pdt
                    dic['vol_id'] = vid
                    dic['value_date'] = date
                    dic['call_put_id'] = ind
                    dlist.append(dic)

    vbd = pd.DataFrame(dlist, columns=delta_labels.extend([
                       'pdt', 'vol_id', 'value_date', 'call_put_id']))

    vbd = pd.merge(vdf, vbd, on=['pdt', 'vol_id', 'value_date', 'call_put_id'])

    # resetting indices
    return vbd

if __name__ == '__main__':
    # filepath = 'portfolio_specs.txt'
    # vdf, pdf, edf = read_data(filepath)
    vdf = pd.read_csv('datasets/small_data/final_vols.csv')
    pdf = pd.read_csv('datasets/small_data/final_price.csv')
    vdf.value_date = pd.to_datetime(vdf.value_date)
    pdf.value_date = pd.to_datetime(pdf.value_date)
    t = time.clock()

    vbd, merged = vol_by_delta(vdf, pdf)
    # vbd = pd.concat([vbd_c, vbd_p], axis=0)
    # vbd_c.to_csv('vols_by_delta_c.csv')
    # vbd_p.to_csv('vols_by_delta_p.csv')
    # vbd.to_csv('test_vols_by_delta.csv', index=False)
    elapsed = time.clock() - t
    print('time elapsed: ', elapsed)

    df1 = pd.read_csv('datasets/small_data/final_price.csv')
    df1.value_date = pd.to_datetime(df1.value_date)

    finmerged = pd.merge(
        df1, vbd, on=['pdt', 'underlying_id', 'vol_id', 'order'])
    finmerged.to_csv('datasets/small_data/final_merged.csv', index=False)
