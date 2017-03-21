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
    # try:
    vdata, pdata = read_data(path)
    # except FileNotFoundError:
    #     print('Curr directory: ', os.getcwd())
    vdata = clean_data(vdata, 'vol')
    pdata = clean_data(pdata, 'p')

    pmin = min(pdata['value_date'])
    vmin = min(vdata['value_date'])

    assert pmin == vmin

    sim_start = pmin
    pf = prep_portfolio(vdata, pdata, pmin)

    longs = pf.long_pos
    short = pf.short_pos

    assert len(longs) == 1
    assert len(short) == 1
