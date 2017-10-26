"""
File Name      : test_prep_data.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 17/4/2017
Python version : 3.5
Description    : File contains tests for the methods in prep_data.py

"""
import scripts.prep_data as pr
# import os
import pandas as pd
import numpy as np
from scripts.classes import Option, Future
# import scripts.global_vars as gv
from scripts.fetch_data import grab_data


############## variables ###########
yr = 2017
start_date = '2014-01-01'
end_date = '2017-08-21'
pdts = ['MW']

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=False, test=True)
####################################


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


def test_find_cdist():
    pdt = 'C'
    all_mths = contract_mths[pdt]
    m1, m2, m3, m4, m5 = 'H7', 'K7', 'N7', 'U7', 'Z7'
    # try:
    assert pr.find_cdist(m1,  m2, all_mths) == 1
    assert pr.find_cdist(m1,  m3, all_mths) == 2
    assert pr.find_cdist(m1,  m4, all_mths) == 3
    assert pr.find_cdist(m1,  m5, all_mths) == 4
    assert pr.find_cdist(m3, 'H8', all_mths) == 3
    assert pr.find_cdist(m3, 'K8', all_mths) == 4
    assert pr.find_cdist(m4, 'N8', all_mths) == 4
    assert pr.find_cdist('H7', 'H8', all_mths) == 5
    assert pr.find_cdist('H7', 'H9', all_mths) == 10
    # except AssertionError:
    #     print('x1: ', m1)
    #     print('x2: ', m2)
    #     print('all_mths: ', all_mths)
    # print('cdist: ', pr.find_cdist(x1, x2, all_mths))

    # next product
    pdt = 'LH'
    all_mths = contract_mths[pdt]
    x1, x2, x3, x4, x5, x6, x7, x8 = 'G7', 'J7', 'K7', 'M7', 'N7', 'Q7', 'V7', 'Z7'
    # assert all_mths == [x1, x2, x3, x4, x5, x6, x7, x8]
    # try:
    assert pr.find_cdist(x1, x2, all_mths) == 1
    assert pr.find_cdist(x1, x3, all_mths) == 2
    assert pr.find_cdist(x1, x4, all_mths) == 3
    assert pr.find_cdist(x1, x5, all_mths) == 4
    assert pr.find_cdist(x7, 'J8', all_mths) == 3
    assert pr.find_cdist(x6, 'K8', all_mths) == 5
    assert pr.find_cdist(x5, 'M8', all_mths) == 7
    assert pr.find_cdist(x8, 'G8', all_mths) == 1
    assert pr.find_cdist('G7', 'G9', all_mths) == 16
    assert pr.find_cdist('G7', 'G8', all_mths) == 8
    # except AssertionError:
    #     print('x1: ', x1)
    #     print('x2: ', x2)
    #     print('all_mths: ', all_mths)
    #     print('cdist: ', pr.find_cdist(x1, x2, all_mths))

    # mth entry not in list
    m1, m2, m3, m4, m5 = 'H7', 'K7', 'N7', 'U7', 'Z7'
    all_mths = contract_mths['C']

    assert pr.find_cdist('G7', 'H7', all_mths) == 1
    assert pr.find_cdist('V7', 'H8', all_mths) == 2
    assert pr.find_cdist('X7', 'H8', all_mths) == 2
    assert pr.find_cdist('H7', 'H10', all_mths) == 15
    assert pr.find_cdist('X7', 'H9', all_mths) == 7
    assert pr.find_cdist('X7', 'Z8', all_mths) == 6
    assert pr.find_cdist('X7', 'Z9', all_mths) == 11


def test_daily_to_bullets():
    ft1 = Future('H7', 30, 'C')
    op1 = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer', False, 'Z7')
    op2 = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer', False, 'Z7', bullet=False)
    dic1 = {'hedge': [], 'OTC': [op1]}
    dic2 = {'hedge': [], 'OTC': [op2]}
    # applying function
    sim_start = pd.to_datetime(vdf.value_date.min())
    print('simstart: ', sim_start)
    ret1 = pr.handle_dailies(dic1, sim_start)
    ret2 = pr.handle_dailies(dic2, sim_start)

    assert len(ret1) == 2
    assert len(ret1['OTC']) == 1
    # try:
    assert len(ret2['OTC']) == 13
    # except AssertionError:
    #     print('actual len: ', len(ret2['OTC']))
    #     print('desired: ', 13)


def test_reorder_ohlc_data():
    pass
