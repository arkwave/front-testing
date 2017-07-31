# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-28 15:41:07
# @Last Modified by:   Ananth
# @Last Modified time: 2017-07-31 17:22:57

from scripts.fetch_data import grab_data
from scripts.util import create_straddle
import pandas as pd
from simulation import run_simulation
from scripts.prep_data import generate_hedges, sanity_check
from scripts.portfolio import Portfolio
from timeit import default_timer as timer

######### variables ################
yr = 2016
start_date = str(yr) + '-08-15'
end_date = str(yr) + '-12-31'

pdts = ['C', 'W']
contract = 'H' + str((yr+1) % 2010) + '.' + 'H' + str((yr+1) % 2010)

corn_contract = 'C' + '  ' + contract
wheat_contract = 'W' + '  ' + contract


volids = [pdt + '  ' + contract for pdt in pdts]

####################################


vdf, pdf, edf = grab_data(pdts, start_date, end_date, volids=volids)

sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), pd.to_datetime(start_date), pd.to_datetime(end_date))


w1, w2 = create_straddle(wheat_contract, vdf, pdf,
                         pd.to_datetime(start_date), False, 'atm', greek='vega', greekval=50000)

c1, c2 = create_straddle(corn_contract, vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', greek='vega', greekval=50000)

# print('CC Call: ', str(cc1))
# print('CC Put: ', str(cc2))
# print('QC Call: ', str(qc1))
# print('QC Put: ', str(qc2))

pf = Portfolio()
pf.add_security([w1, w2, c1, c2], 'OTC')

print('portfolio: ', pf)

hedges = generate_hedges('hedging.csv')

print('hedges: ', hedges)

t = timer()

grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values,\
    net_daily_values, net_cumul_values, log, analytics_csv = run_simulation(
        vdf, pdf, edf, pf, hedges, roll_portfolio=False, roll_hedges=False)


print('elapsed: ', timer() - t)


name = 'w_c_log'
analytics_csv.to_csv(str(yr) + '_' + str(name) +
                     '_trade_analytics.csv', index=False)


# log.to_csv(str(yr) + '_' + str(name) + '_trade_log.csv', index=False)
