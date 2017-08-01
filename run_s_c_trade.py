# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-28 15:41:07
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-01 15:00:16


from scripts.fetch_data import grab_data
from scripts.util import create_straddle
from scripts.prep_data import generate_hedges, sanity_check
from scripts.portfolio import Portfolio
from simulation import run_simulation
from scripts.hedge import Hedge


# from HistoricSimulator import scripts
# from scripts.fetch_data import grab_data
# from scripts.util import create_straddle
import pandas as pd
from timeit import default_timer as timer

######### variables ################
yr = 2016
start_date = str(yr) + '-08-15'
end_date = str(yr) + '-12-01'

pdts = ['C', 'S']
contract = 'H' + str((yr+1) % 2010) + '.' + 'H' + str((yr+1) % 2010)

corn_contract = 'C' + '  ' + contract
soy_contract = 'S' + '  ' + contract


volids = [pdt + '  ' + contract for pdt in pdts]

####################################


vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          volids=volids, write_dump=False)

sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), pd.to_datetime(start_date), pd.to_datetime(end_date))


s1, s2 = create_straddle(soy_contract, vdf, pdf,
                         pd.to_datetime(start_date), False, 'atm', greek='vega', greekval=50000)

c1, c2 = create_straddle(corn_contract, vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', greek='vega', greekval=50000)


# print('s1: ', str(s1))
# print('s2: ', str(s2))
# print('c1: ', str(c1))
# print('c2: ', str(c2))


pf = Portfolio()
pf.add_security([s1, s2, c1, c2], 'OTC')

print('portfolio: ', pf)

hedges = generate_hedges('hedging.csv')

print('hedges: ', hedges)

t = timer()

log = run_simulation(vdf, pdf, edf, pf, hedges,
                     roll_portfolio=False, roll_hedges=False)


print('elapsed: ', timer() - t)


dirname = 'results/20170731 - Soy-Corn Split Trade/'
filename = 's_c_trade_log.csv'

# analytics_csv.to_csv(str(yr) + '_' + str(name) +
#                      '_trade_analytics.csv', index=False)


log.to_csv(dirname + str(yr) + '_' + str(filename), index=False)
