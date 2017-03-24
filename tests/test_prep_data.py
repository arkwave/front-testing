"""
File Name      : test_prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : File contains tests for the methods in prep_data.py

"""
from scripts.prep_data import *
import os


def test_prep_portfolio():
    path = 'portfolio_specs.txt'
    vdata, pdata, edf = read_data(path)
    pmin = min(pdata['value_date'])
    vmin = min(vdata['value_date'])

    assert pmin == vmin

    sim_start = pmin
    pf = prep_portfolio(vdata, pdata, pmin)

    otc = pf.OTC
    hedge = pf.hedges

    assert len(otc) == 1
    assert len(hedge) == 1
