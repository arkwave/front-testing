# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-07-21 15:41:52
# @Last Modified by:   arkwave
# @Last Modified time: 2017-08-17 21:30:25

from scripts.fetch_data import grab_data
from scripts.util import create_straddle, combine_portfolios
import pandas as pd
from simulation import run_simulation
from scripts.prep_data import generate_hedges
from scripts.portfolio import Portfolio


######### variables ################
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']
volids = ['CC  Z7.Z7', 'QC  Z7.Z7']

####################################

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          volids=volids, write_dump=True)


# specifying portfolio
cc1, cc2 = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                           False, 'atm', greek='theta', greekval=10000)

qc1, qc2 = create_straddle('QC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                           True, 'atm', greek='theta', greekval=10000)

# specifying hedges.
hedges, roll_portfolio, pf_ttm_tol, pf_roll_pdt, \
    roll_hedges, h_ttm_tol, h_roll_product = generate_hedges(
        'cc_qc_hedges.csv')

print('hedges: ', hedges)

# pf = Portfolio(hedges, 1)

# pf.add_security([cc1, cc2, qc1, qc2], 'OTC')

pf_cc = Portfolio(hedges, 1)
pf_cc.add_security([cc1, cc2], 'OTC')

# pf = pf_cc

pf_qc = Portfolio(hedges, 2)
pf_qc.add_security([qc1, qc2], 'OTC')


# pf = pf_qc

pf = combine_portfolios([pf_cc, pf_qc], hedges=hedges,
                        name='all', refresh=True)

pf.refresh()

print('portfolio: ', pf)
print('pf ttm tol: ', pf_ttm_tol)
print('pf roll product: ', pf_roll_pdt)

# signals = pd.read_csv('signals.csv')
# signals.value_date = pd.to_datetime(signals.value_date)


# # running simulation
# log = run_simulation(vdf, pdf, edf, pf,
#                      roll_portfolio=roll_portfolio, pf_ttm_tol=pf_ttm_tol,
#                      pf_roll_product=pf_roll_pdt,
#                      roll_hedges=roll_hedges, h_ttm_tol=h_ttm_tol,
#                      h_roll_product=h_roll_product)

# log = run_simulation(vdf, pdf, edf, pf, hedges)


# log.to_csv('30dayroll.csv', index=False)
