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


def read_hedges(filepath):
    df = pd.read_csv(filepath)
    hedges = {}
    for i in df.index:
        row = df.iloc[i]
        # static hedging
        if row.flag == 'static':
            greek = row.greek
            hedges[greek] = [row.cond, int(row.freq)]
        # bound hedging
        elif row.flag == 'bound':
            greek = row.greek
            hedges[greek] = [literal_eval(row.cond), int(row.freq)]

        # percentage hedging
        elif row.flag == 'pct':
            greek = row.greek
            hedges[greek] = [float(row.cond), int(row.freq)]
    return hedges

hedges = read_hedges()
