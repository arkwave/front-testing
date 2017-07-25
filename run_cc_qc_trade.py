# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-21 15:41:52
# @Last Modified by:   arkwave
# @Last Modified time: 2017-07-24 19:20:23
from scripts.fetch_data import grab_data
from scripts.util import create_straddle
import pandas as pd
from simulation import run_simulation
from scripts.prep_data import generate_hedges, sanity_check
from scripts.portfolio import Portfolio


######### variables ################
yr = 2016
start_date = str(yr) + '-05-02'
end_date = str(yr) + '-12-31'
# rd1 = pd.to_datetime(str(yr) + '-07-01')
# rd2 = pd.to_datetime(str(yr) + '-07-01')
# rd2 = pd.to_datetime(str(yr) + '-09-04')

# rollover_dates = {'QC  U6.U6': [rd1],
#                   'CC  U6.U6': [rd1],
#                   'CC  Z6.Z6': [rd2],
#                   'QC  Z6.Z6': [rd2], }

pdts = ['QC', 'CC']

volids = ['QC  U6.U6', 'QC  Z6.Z6', 'CC  U6.U6',
          'CC  Z6.Z6', 'QC  H7.H7', 'CC  H7.H7',
          'CC  K7.K7', 'QC  K7.K7', 'CC  N7.N7', 'QC  N7.N7']
####################################


vdf, pdf, edf = grab_data(pdts, start_date, end_date, volids=volids)

sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), pd.to_datetime(start_date), pd.to_datetime(end_date))


cc1, cc2 = create_straddle('CC  U6.U6', vdf, pdf,
                           pd.to_datetime(start_date), True, 'atm', greek='vega', greekval=50000)

qc1, qc2 = create_straddle('QC  U6.U6', vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', greek='vega', greekval=50000)

# print('CC Call: ', str(cc1))
# print('CC Put: ', str(cc2))
# print('QC Call: ', str(qc1))
# print('QC Put: ', str(qc2))

pf = Portfolio()
pf.add_security([qc1, qc2, cc1, cc2], 'OTC')

print('portfolio: ', pf)

hedges = generate_hedges('hedging.csv')

print('hedges: ', hedges)

grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values,\
    net_daily_values, net_cumul_values, log = run_simulation(
        vdf, pdf, edf, pf, hedges, roll_portfolio=True, roll_product='CC', ttm_tol=60)


name = 'qc_cc_log'
log.to_csv('../../QC_CC_Splits/' + str(name) + '.csv', index=False)
