"""
File Name      : test_prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 30/3/2017
Python version : 3.5
Description    : File contains tests for the methods in prep_data.py

"""
import scripts.prep_data as pr
import os
import pandas as pd

filepath = 'portfolio_specs.txt'
voldata, pricedata, edf = pr.read_data(filepath)

contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'LCC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V' 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}




def test_prep_portfolio():
    vdata, pdata, edf = pr.read_data(filepath)
    pmin = min(pdata['value_date'])
    vmin = min(vdata['value_date'])

    assert pmin == vmin

    sim_start = pmin
    pf = pr.prep_portfolio(vdata, pdata, pmin)

    otc = pf.OTC
    hedge = pf.hedges

    assert len(otc) == 1
    assert len(hedge) == 1


def test_get_rollover_dates():
    ret = pr.get_rollover_dates(pricedata)
    assert len(ret) == 1
    assert set(ret.keys()) == set(['C'])
    val = pd.to_datetime('2017-02-26 00:00:00')
    assert ret['C'][0] == val


def test_find_cdist():
    pdt = 'C'
    all_mths = contract_mths[pdt]
    m1, m2, m3, m4, m5 = 'H', 'K', 'N', 'U', 'Z'
    try:
        assert pr.find_cdist(m1, m2, all_mths) == 1
        assert pr.find_cdist(m1, m3, all_mths) == 2
        assert pr.find_cdist(m1, m4, all_mths) == 3
        assert pr.find_cdist(m1, m5, all_mths) == 4
        assert pr.find_cdist(m3, m1, all_mths) == 3
        assert pr.find_cdist(m3, m2, all_mths) == 4
        assert pr.find_cdist(m4, m3, all_mths) == 4
    except AssertionError:  
        print('x1: ', m1)
        print('x2: ', m2)
        print('all_mths: ', all_mths)
        print('cdist: ', pr.find_cdist(x1, x2, all_mths))    

    # next product
    pdt = 'LH'
    all_mths = contract_mths[pdt]
    x1, x2, x3, x4, x5, x6, x7, x8 = 'G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'
    assert all_mths == [x1,x2, x3, x4, x5, x6, x7, x8]
    try:
        assert pr.find_cdist(x1, x2, all_mths) == 1
        assert pr.find_cdist(x1, x3, all_mths) == 2
        assert pr.find_cdist(x1, x4, all_mths) == 3
        assert pr.find_cdist(x1, x5, all_mths) == 4
        assert pr.find_cdist(x7, x2, all_mths) == 3
        assert pr.find_cdist(x6, x3, all_mths) == 5
        assert pr.find_cdist(x5, x4, all_mths) == 7
        assert pr.find_cdist(x8, x1, all_mths) == 1
    except AssertionError:  
        print('x1: ', x1)
        print('x2: ', x2)
        print('all_mths: ', all_mths)
        print('cdist: ', pr.find_cdist(x1, x2, all_mths))    
    
