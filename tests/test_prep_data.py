"""
File Name      : test_prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 28/3/2017
Python version : 3.5
Description    : File contains tests for the methods in prep_data.py

"""
from scripts.prep_data import *
import os

filepath = 'portfolio_specs.txt'
voldata, pricedata, edf = read_data(filepath)


def test_prep_portfolio():
    vdata, pdata, edf = read_data(filepath)
    pmin = min(pdata['value_date'])
    vmin = min(vdata['value_date'])

    assert pmin == vmin

    sim_start = pmin
    pf = prep_portfolio(vdata, pdata, pmin)

    otc = pf.OTC
    hedge = pf.hedges

    assert len(otc) == 1
    assert len(hedge) == 1


def test_get_rollover_dates():
    ret = get_rollover_dates(pricedata)
    assert len(ret) == 1
    assert set(ret.keys()) == set(['C'])
    val = pd.to_datetime('2017-02-26 00:00:00')
    assert ret['C'][0] == val
