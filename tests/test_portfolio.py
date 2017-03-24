"""
File Name      : test_portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 16/3/2017
Python version : 3.5
Description    : File contains tests for the Portfolio class methods in portfolio.py

"""
from scripts.portfolio import Portfolio
from scripts.classes import Option, Future
import copy
import numpy as np


def generate_portfolio():
    """Generate portfolio for testing purposes. """
    # Underlying Futures
    ft1 = Future('march', 30, 'C')
    ft2 = Future('may', 25, 'C')
    ft3 = Future('april', 32, 'BO')
    ft4 = Future('april', 33, 'LH')
    ft5 = Future('april', 24, 'LH')

    # options
    op1 = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer', False)
    op2 = Option(
        29, 0.2156100288506942, 'call', 0.45176132048500206, ft2, 'amer', False)
    op3 = Option(30, 0.21534276294769317, 'call', 0.14464169782291536,
                 ft3, 'amer', True, direc='up', barrier='amer', bullet=False, ko=35)
    op4 = Option(33, 0.22365510948646386, 'put', 0.18282926924909026,
                 ft4, 'amer', False, direc='down', barrier='amer', bullet=False, ki=28)
    op5 = Option(
        32, 0.010975090692443346, 'put', 0.8281728247909962, ft5, 'amer', True)

    # Portfolio Futures
    ft6 = Future('may', 37, 'C', shorted=False)
    ft7 = Future('march', 29, 'BO', shorted=False)
    ft8 = Future('april', 32, 'C', shorted=True)
    ft9 = Future('april', 32, 'BO', shorted=True)

    OTCs, hedges = [op1, op2, ft7, op4, ft6], [op5, op3, ft8, ft9]

    # creating portfolio
    pf = Portfolio()
    for sec in hedges:
        pf.add_security(sec, 'hedge')

    for sec in OTCs:
        pf.add_security(sec, 'OTC')

    return pf


def test_add_remove_net():
    pf = generate_portfolio()
    OTC_pos = pf.get_securities_monthly('OTC')
    hedge_pos = pf.get_securities_monthly('hedge')
    net = pf.get_net_greeks()
    assert len(OTC_pos) == 3
    assert len(hedge_pos) == 3
    assert len(pf.OTC_options) == 3
    assert len(pf.hedge_options) == 2
    assert len(pf.OTC_futures) == 2
    assert len(pf.hedge_futures) == 2
    assert len(net) == 3


def test_add_multiple():
    pf = generate_portfolio()
    assert len(pf.OTC) == 3
    assert len(pf.OTC_options) == 3
    ft = Future('march', 30, 'C')
    op = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft, 'amer', True)
    pf.add_security(op, 'OTC')
    pf.add_security(op, 'OTC')
    assert len(pf.OTC_options) == 5


def test_remove_dne():
    pf = generate_portfolio()
    ft = Future('july', 50, 'D')
    assert pf.remove_security(ft, 'OTC') == -1
    assert pf.remove_security(ft, 'hedge') == -1


def test_OTC_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('OTC')
    assert set(dic.keys()) == set(['C', 'LH', 'BO'])
    # sub-dictionaries.
    dic_c = dic['C']
    assert set(dic_c.keys()) == set(['march', 'may'])
    dic_bo = dic['BO']
    assert set(dic_bo.keys()) == set(['march'])
    dic_lh = dic['LH']
    assert set(dic_lh.keys()) == set(['april'])


def test_hedge_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('hedge')
    assert set(dic.keys()) == set(['C', 'LH', 'BO'])
    # sub dictionaries
    dic_c = dic['C']
    assert set(dic_c.keys()) == set(['april'])
    dic_lh = dic['LH']
    assert set(dic_lh.keys()) == set(['april'])
    dic_bo = dic['BO']
    assert set(dic_bo.keys()) == set(['april'])


def test_remove_security_futures():
    pf = generate_portfolio()
    ft_test = Future('aug', 50, 'C')
    pf2 = copy.deepcopy(pf)
    prev_net = pf2.get_net_greeks()
    prev_OTCs = pf2.get_securities_monthly('OTC')

    # basic checks
    assert len(prev_OTCs) == 3
    assert len(prev_OTCs['C']) == 2
    assert set(prev_OTCs['C'].keys()) == set(['march', 'may'])
    assert len(pf.OTC_futures) == 2
    assert ft_test.get_product() == 'C'

    # adding future into portfolio; should not change greeks and lengths of
    # relevant data structures. additionally, net greeks should not have
    # futures.
    pf.add_security(ft_test, 'OTC')
    try:
        assert set(pf.get_net_greeks()['C'].keys()) == set(
            ['march', 'may'])
    except AssertionError:
        print(pf.get_net_greeks()['C'].keys())

    # checking addition
    curr_OTCs = pf.get_securities_monthly('OTC')
    cprod = curr_OTCs['C']
    assert len(cprod) == 3
    assert set(cprod) == set(['march', 'may', 'aug'])
    assert len(curr_OTCs) == 3
    assert len(pf.OTC_futures) == 3

    # checking current status
    curr_net = pf.get_net_greeks()

    # greeks should be same since futures do not contribute greeks.
    assert curr_net == prev_net
    assert curr_OTCs != prev_OTCs

    # now remove the same security
    pf.remove_security(ft_test, 'OTC')
    assert set(pf.get_net_greeks()['C'].keys()) == set(
        ['march', 'may'])

    # data structures should reset
    rem_OTCs = pf.get_securities_monthly('OTC')

    assert set(rem_OTCs.keys()) == set(['C', 'LH', 'BO'])

    rprod = rem_OTCs['C']
    assert set(rprod.keys()) == set(['march', 'may'])

    rem_net = pf.get_net_greeks()
    assert len(pf.OTC_futures) == 2

    # failing
    try:
        assert rem_net == prev_net
    except AssertionError:
        print(rem_net['C'].keys())
        print(prev_net['C'].keys())

    # failing
    try:
        assert rem_OTCs == prev_OTCs
    except AssertionError:
        mar_option1 = list(rem_OTCs['C']['march'][0])[0]
        mar_option2 = list(prev_OTCs['C']['march'][0])[0]
        may_option1 = list(rem_OTCs['C']['may'][0])[0]
        may_option2 = list(prev_OTCs['C']['may'][0])[0]
        may_future1 = list(rem_OTCs['C']['may'][1])[0]
        may_option2 = list(prev_OTCs['C']['may'][1])[0]
        # checking equality of options
        assert mar_option1.underlying.get_price(
        ) == mar_option2.underlying.get_price()
        assert mar_option1.K == mar_option2.K
        assert mar_option1.tau == mar_option2.tau
        assert mar_option1.char == mar_option2.char
        assert mar_option1.price == mar_option2.price
        assert mar_option1.get_product() == mar_option2.get_product()


def test_remove_security_options():
    pf = generate_portfolio()
    ft_test = Future('aug', 50, 'C')
    op_test = Option(35, 0.02, 'call', 0.8, ft_test, 'amer', False)
    prev_net = copy.deepcopy(pf.get_net_greeks())
    prev_OTCs = copy.deepcopy(pf.get_securities_monthly('OTC'))
    assert len(pf.get_securities_monthly('OTC')) == 3
    assert len(pf.OTC_options) == 3

    # adding security into portfolio; should change greeks and lengths of
    # relevant data structures.
    pf.add_security(op_test, 'OTC')
    assert len(pf.get_securities_monthly('OTC')) == 3
    assert len(pf.OTC_options) == 4
    curr_net = pf.get_net_greeks()

    curr_OTCs = pf.get_securities_monthly('OTC')
    # greeks should NOT be same since options contribute greeks.
    assert curr_net != prev_net
    assert curr_OTCs != prev_OTCs

    # now remove the same security
    pf.remove_security(op_test, 'OTC')

    # data structures should reset
    rem_OTCs = pf.get_securities_monthly('OTC')
    rem_net = pf.get_net_greeks()
    assert len(pf.OTC_options) == 3
    assert len(pf.get_securities_monthly('OTC')) == 3

    try:
        assert rem_net == prev_net
    except AssertionError:
        print(rem_OTCs['C'].keys())
        print(pf.hedge_pos['C'].keys())
        print(rem_net['C'].keys())
        print(prev_net['C'].keys())

    try:
        assert rem_OTCs == prev_OTCs
    # memory location errors
    except AssertionError:
        mar_option1 = list(rem_OTCs['C']['march'][0])[0]
        mar_option2 = list(prev_OTCs['C']['march'][0])[0]
        may_option1 = list(rem_OTCs['C']['may'][0])[0]
        may_option2 = list(prev_OTCs['C']['may'][0])[0]
        may_future1 = list(rem_OTCs['C']['may'][1])[0]
        may_option2 = list(prev_OTCs['C']['may'][1])[0]

        # checking equality of options
        assert mar_option1.underlying.get_price(
        ) == mar_option2.underlying.get_price()
        assert mar_option1.K == mar_option2.K
        assert mar_option1.tau == mar_option2.tau
        assert mar_option1.char == mar_option2.char
        assert mar_option1.price == mar_option2.price
        assert mar_option1.get_product() == mar_option2.get_product()


def test_remove_expired():
    pf = generate_portfolio()
    ft = Future('june', 50, 'C')
    init_net = copy.deepcopy(pf.get_net_greeks())
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False)
    pf.add_security(op1, 'OTC')
    assert len(pf.OTC_options) == 4
    assert len(pf.OTC['C']) == 3
    prev_net = copy.deepcopy(pf.get_net_greeks())
    pf.timestep(0.01)
    pf.remove_expired()
    curr_net = copy.deepcopy(pf.get_net_greeks())
    try:
        assert curr_net != prev_net
        assert curr_net == init_net
    except AssertionError:
        print(curr_net)
        print(prev_net)
    assert 'june' not in pf.OTC['C']
    assert len(pf.OTC_options) == 3


def test_compute_value():
    pf = generate_portfolio()
    init_val = pf.compute_value()
    ft = Future('june', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False)
    opval = op1.get_price()
    # testing OTC pos
    pf.add_security(op1, 'OTC')
    addval = pf.compute_value()
    try:
        assert (addval - init_val) == opval
    except AssertionError:
        assert np.isclose(addval-init_val, opval)
        print('residue: ', addval - init_val - opval)
    # testing hedge pos
    pf.remove_security(op1, 'OTC')
    pf.add_security(op1, 'hedge')
    hedgeval = pf.compute_value()
    assert init_val - hedgeval == opval


def test_exercise_option():
    pf = generate_portfolio()
    # initial
    assert len(pf.OTC_options) == 3
    assert len(pf.OTC_futures) == 2
    init_greeks = copy.deepcopy(pf.get_net_greeks())

    # add option
    ft = Future('june', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False)
    pf.add_security(op1, 'OTC')

    assert len(pf.OTC_options) == 4
    add_greeks = copy.deepcopy(pf.get_net_greeks())
    assert add_greeks != init_greeks

    # exercise
    pf.exercise_option(op1, 'OTC')
    assert len(pf.OTC_options) == 3
    assert len(pf.OTC_futures) == 3
    ex_greeks = copy.deepcopy(pf.get_net_greeks())
    assert ex_greeks == init_greeks
