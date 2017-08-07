# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-21 15:41:52
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-04 17:51:52

from scripts.fetch_data import grab_data
from scripts.util import create_straddle
import pandas as pd
from simulation import run_simulation
from scripts.prep_data import generate_hedges, sanity_check
from scripts.portfolio import Portfolio


"""
Variables required:
1) Products, start_date, end_date, volids (optional) --> used to draw data. 
2) Filepath to hedging (maybe alternate with a checkbox and ranges


3) Simulation parameters:
	> brokerage
	> slippage 
	> signals
	> roll_portfolio
	> roll_hedges 
	> roll_product
	> ttm_tol
	> volids (if required to manually specify rollovers.)


4) Portfolio parameters:
	1) structure type - string. 
	2) structure specifics:
		> strike(s)
		> delta
		> vol_id 
		> init date
		> lot specification. 

"""


######### variables ################
yr = 2016
# start_date = str(yr) + '-01-03'
# end_date = str(yr) + '-03-31'
start_date = str(yr) + '-05-02'
end_date = str(yr) + '-12-31'

pdts = ['QC', 'CC']


volids = ['QC  U6.U6', 'QC  Z6.Z6', 'CC  U6.U6',
          'CC  Z6.Z6', 'QC  H7.H7', 'CC  H7.H7',
          'CC  K7.K7', 'QC  K7.K7', 'CC  N7.N7', 'QC  N7.N7']
####################################

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date, volids=volids)


# specifying portfolio
cc1, cc2 = create_straddle('CC  U6.U6', vdf, pdf,
                           pd.to_datetime(start_date), False, 'atm', greek='vega', greekval=50000)

qc1, qc2 = create_straddle('QC  U6.U6', vdf, pdf, pd.to_datetime(
    start_date), True, 'atm', greek='vega', greekval=50000)


pf = Portfolio()
pf.add_security([cc1, cc2, qc1, qc2], 'OTC')

print('portfolio: ', pf)


# specifying hedges.
hedges, roll_portfolio, pf_ttm_tol, pf_roll_pdt, \
    roll_hedges, h_ttm_tol, h_roll_product = generate_hedges('hedging.csv')

print('hedges: ', hedges)

print('pf ttm tol: ', pf_ttm_tol)
print('pf roll product: ', pf_roll_pdt)

# signals = pd.read_csv('signals.csv')
# signals.value_date = pd.to_datetime(signals.value_date)


# running simulation
log = run_simulation(vdf, pdf, edf, pf, hedges,
                     roll_portfolio=roll_portfolio, pf_ttm_tol=pf_ttm_tol,
                     pf_roll_product=pf_roll_pdt,
                     roll_hedges=roll_hedges, h_ttm_tol=h_ttm_tol,
                     h_roll_product=h_roll_product)

# name = str(yr) + '_qc_cc_log'
# log.to_csv('../../QC_CC_Splits/' + str(name) +
#            '_hedge_test.csv', index=False)
