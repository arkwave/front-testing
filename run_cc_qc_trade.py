# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-21 15:41:52
# @Last Modified by:   Ananth
# @Last Modified time: 2017-07-21 21:16:25
from scripts.fetch_data import grab_data
from scripts.util import create_straddle
import pandas as pd
from simulation import run_simulation
from scripts.prep_data import generate_hedges, sanity_check
from scripts.portfolio import Portfolio


######### variables ################
start_date = '2017-05-01'
end_date = '2017-07-21'
rd1 = pd.to_datetime('2017-06-30')
rd2 = pd.to_datetime('2017-06-30')
rollover_dates = {'QC  U7.U7': [rd1], 'CC  U7.U7': [rd2]}
pdts = ['QC', 'CC']
volids = ['QC  U7.U7', 'QC  Z7.Z7', 'CC  U7.U7', 'CC  Z7.Z7']
####################################


vdf, pdf, edf = grab_data(pdts, start_date, end_date, volids=volids)

sanity_check(vdf.value_date.unique(),
             pdf.value_date.unique(), pd.to_datetime(start_date), pd.to_datetime(end_date))


cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf,
                           pd.to_datetime(start_date), False, 'atm', lots=1000)

qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', lots=1000)

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
        vdf, pdf, edf, pf, hedges, rollover_dates=rollover_dates, roll_portfolio=True)
