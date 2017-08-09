# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-08-09 17:01:19
# @Last Modified by:   arkwave
# @Last Modified time: 2017-08-09 17:21:25

from scripts.util import combine_portfolios, create_straddle 
from scripts.fetch_data import grab_data 
from scripts.portfolio import Portfolio


import pandas as pd 

######### variables ################
start_date = '2017-05-01'
end_date = '2017-07-21'
pdts = ['QC', 'CC']
volids = ['QC  U7.U7', 'QC  Z7.Z7', 'CC  U7.U7', 'CC  Z7.Z7']
####################################

vdf, pdf, edf = grab_data(pdts, start_date, end_date,
	                      volids=volids, write=True)

pdf = pdf[pdf.value_date == pd.to_datetime(start_date)]
vdf = vdf[vdf.value_date == pd.to_datetime(start_date)]


def test_create_underlying():
	pass 


def test_create_vanilla_option():
	pass 


def test_create_straddle():
	pass 


def test_create_strangle():
	pass 


def test_create_butterfly():
	pass 

def test_create_spread():
	pass 

def test_create_skew():
	pass 


def test_create_composites():
	pass 


def test_merge_dicts():
	pass 


def test_merge_lists():
	pass 



def test_combine_portfolios():

	cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
	    start_date), False, 'atm', greek='vega', greekval=20000)

	qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
	    start_date), True, 'atm', greek='vega', greekval=20000)

	pf1 = Portfolio()
	pf1.add_security([qc1, qc2], 'OTC')

	pf2 = Portfolio() 
	pf2.add_security([cc1, cc2], 'OTC')
	
	pf3 = combine_portfolios([pf1, pf2])

	# OTC check 
	otc = pf1.OTC.copy()
	otc.update(pf2.OTC.copy())
	for pdt in otc:
		for mth in otc[pdt]:
			assert pf3.OTC[pdt][mth] == otc[pdt][mth]

	# hedge check 
	hed = pf1.hedges.copy()
	hed.update(pf2.hedges.copy())
	for pdt in hed:
		for mth in hed[pdt]:
			assert pf3.hedges[pdt][mth] == hed[pdt][mth]

	# net check 
	net_tst = pf1.net_greeks.copy() 
	net_tst.update(pf2.net_greeks)
	for pdt in net_tst:
		for mth in net_tst[pdt]:
			assert net_tst[pdt][mth] == pf3.net_greeks[pdt][mth]

	# lists check 
	otcops = pf1.OTC_options.copy()
	otcops.extend(pf2.OTC_options)
	assert otcops == pf3.OTC_options 

	hops = pf1.hedge_options.copy() 
	hops.extend(pf2.hedge_options)
	assert hops == pf3.hedge_options 

	assert pf3.families == [pf1, pf2]
	assert isinstance(pf3, Portfolio)



