import numpy as np
import pandas as pd
# import time
from sqlalchemy import create_engine
from scripts.portfolio import Portfolio
from scripts.util import create_underlying, create_vanilla_option
import scripts.prep_data as pr
import os
from scripts.fetch_data import grab_data, pull_intraday_data


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

seed = 7
np.random.seed(seed)
pd.options.mode.chained_assignment = None

# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10


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


"""Purpose: Sanity check the granularize function """

# start_date = '2017-09-18'
# end_date = '2017-10-06'
# pdts = ['LH']

# settle_vols, settle_prices, edf = grab_data(
#     pdts, start_date, end_date, volids=['LH  Z7.Z7'])

# data_path = 'C:/Users/Ananth/Desktop/Modules/HistoricSimulator/'

# # handle date and time datatypes
# settle_vols.time = settle_vols.time.astype(pd.Timestamp)
# settle_vols.value_date = pd.to_datetime(settle_vols.value_date)
# settle_prices.value_date = pd.to_datetime(settle_prices.value_date)
# settle_prices.time = settle_prices.time.astype(pd.Timestamp)

# cleaned = data_path + 'datasets/debug/lh_' + start_date + \
#     '_' + end_date + '_intraday_cleaned.csv'


# if os.path.exists(cleaned):
#     fpdf = pd.read_csv(cleaned)
#     fpdf.value_date = pd.to_datetime(fpdf.value_date)
#     fpdf.time = pd.to_datetime(fpdf.time).dt.time

# else:
#     # intraday data.
#     it_data = pull_intraday_data(
#         pdts, start_date=start_date, end_date=end_date)
#     it_data = pr.sanitize_intraday_timings(it_data, filepath=data_path)

#     fpdf = pr.insert_settlements(it_data, settle_prices)
#     fpdf.to_csv(cleaned, index=False)


# # create the position.
# callop = create_vanilla_option(settle_vols, settle_prices, 'LH  Z7.Z7', 'call',
#                                False, strike='atm', greek='theta', greekval=10000)
# pf = Portfolio(None, name='LH_pf')
# pf.add_security([callop], 'OTC')
# delta = pf.net_greeks['LH']['Z7'][0]
# shorted = True if delta > 0 else False
# lots = round(abs(delta))

# ft, _ = create_underlying('LH', 'Z7', settle_prices, date=settle_prices.value_date.min(),
#                           shorted=shorted, lots=lots)

# pf.add_security([ft], 'hedge')

# # filter one particular day.
# tst_df = fpdf[(fpdf.value_date == pd.to_datetime('2017-09-21')) &
#               (fpdf.underlying_id == 'LH  Z7')]
# interval = 1.4
from scripts.prep_data import clean_intraday_data, sanitize_intraday_timings
import datetime as dt
import os
import time

path = 'intraday_cleaning_test_full.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    df.date_time = pd.to_datetime(df.date_time)
else:
    user = 'sumit'
    password = 'Olam1234'
    engine = create_engine('postgresql://' + user + ':' + password +
                           '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()
    query = "select * from public.table_intra_day_trade where commodity like 'LCZ7 %%' and date_time >= '2017-09-18' and date_time <= '2017-10-06' "

    t = time.clock()
    df = pd.read_sql_query(query, connection)
    print('data pulling time elapsed: ', time.clock() - t)
    df.to_csv(path, index=False)


edf = pd.read_csv('datasets/exchange_timings.csv')
df['pdt'] = df.commodity.str[:2].str.strip()
df['time'] = df.date_time.dt.time
tdf = df[df.time == dt.time(8, 30, 00)]


# df = sanitize_intraday_timings(df, edf=edf)

t = time.clock()
df2 = clean_intraday_data(df, edf=edf)
print('data cleaning elapsed: ', time.clock() - t)
