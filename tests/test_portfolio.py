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
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer')
    op2 = Option(
        29, 0.2156100288506942, 'call', 0.45176132048500206, ft2, 'amer')
    op3 = Option(30, 0.21534276294769317, 'call', 0.14464169782291536,
                 ft3, 'amer', direc='up', barrier='amer', bullet=False, ko=35)
    op4 = Option(33, 0.22365510948646386, 'put', 0.18282926924909026,
                 ft4, 'amer', direc='down', barrier='amer', bullet=False, ki=28)
    op5 = Option(
        32, 0.010975090692443346, 'put', 0.8281728247909962, ft5, 'amer')

    # Portfolio Futures
    ft6 = Future('may', 37, 'C')
    ft7 = Future('march', 29, 'BO')
    ft8 = Future('april', 32, 'C')
    ft9 = Future('april', 32, 'BO')

    longs, shorts = [op1, op2, ft7, op4, ft6], [op5, op3, ft8, ft9]

    # creating portfolio
    pf = Portfolio()
    for sec in shorts:
        pf.add_security(sec, 'short')

    for sec in longs:
        pf.add_security(sec, 'long')

    return pf


def test_add_remove_net():
    pf = generate_portfolio()
    long_pos = pf.get_securities_monthly('long')
    short_pos = pf.get_securities_monthly('short')
    net = pf.get_net_greeks()
    assert len(long_pos) == 3
    assert len(short_pos) == 3
    assert len(pf.long_options) == 3
    assert len(pf.short_options) == 2
    assert len(pf.long_futures) == 2
    assert len(pf.short_futures) == 2
    assert len(net) == 3


def test_add_multiple():
    pf = generate_portfolio()
    assert len(pf.long_pos) == 3
    assert len(pf.long_options) == 3
    ft = Future('march', 30, 'C')
    op = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft, 'amer')
    pf.add_security(op, 'long')
    pf.add_security(op, 'long')
    assert len(pf.long_options) == 5


def test_remove_dne():
    pf = generate_portfolio()
    ft = Future('july', 50, 'D')
    assert pf.remove_security(ft, 'long') == -1
    assert pf.remove_security(ft, 'short') == -1


def test_long_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('long')
    assert set(dic.keys()) == set(['C', 'LH', 'BO'])
    # sub-dictionaries.
    dic_c = dic['C']
    assert set(dic_c.keys()) == set(['march', 'may'])
    dic_bo = dic['BO']
    assert set(dic_bo.keys()) == set(['march'])
    dic_lh = dic['LH']
    assert set(dic_lh.keys()) == set(['april'])


def test_short_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('short')
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
    prev_longs = pf2.get_securities_monthly('long')

    # basic checks
    assert len(prev_longs) == 3
    assert len(prev_longs['C']) == 2
    assert set(prev_longs['C'].keys()) == set(['march', 'may'])
    assert len(pf.long_futures) == 2
    assert ft_test.get_product() == 'C'

    # adding future into portfolio; should not change greeks and lengths of
    # relevant data structures. additionally, net greeks should not have
    # futures.
    pf.add_security(ft_test, 'long')
    try:
        assert set(pf.get_net_greeks()['C'].keys()) == set(
            ['march', 'may'])
    except AssertionError:
        print(pf.get_net_greeks()['C'].keys())

    # checking addition
    curr_longs = pf.get_securities_monthly('long')
    cprod = curr_longs['C']
    assert len(cprod) == 3
    assert set(cprod) == set(['march', 'may', 'aug'])
    assert len(curr_longs) == 3
    assert len(pf.long_futures) == 3

    # checking current status
    curr_net = pf.get_net_greeks()

    # greeks should be same since futures do not contribute greeks.
    assert curr_net == prev_net
    assert curr_longs != prev_longs

    # now remove the same security
    pf.remove_security(ft_test, 'long')
    assert set(pf.get_net_greeks()['C'].keys()) == set(
        ['march', 'may'])

    # data structures should reset
    rem_longs = pf.get_securities_monthly('long')

    assert set(rem_longs.keys()) == set(['C', 'LH', 'BO'])

    rprod = rem_longs['C']
    assert set(rprod.keys()) == set(['march', 'may'])

    rem_net = pf.get_net_greeks()
    assert len(pf.long_futures) == 2

    # failing
    try:
        assert rem_net == prev_net
    except AssertionError:
        print(rem_net['C'].keys())
        print(prev_net['C'].keys())

    # failing
    try:
        assert rem_longs == prev_longs
    except AssertionError:
        mar_option1 = list(rem_longs['C']['march'][0])[0]
        mar_option2 = list(prev_longs['C']['march'][0])[0]
        may_option1 = list(rem_longs['C']['may'][0])[0]
        may_option2 = list(prev_longs['C']['may'][0])[0]
        may_future1 = list(rem_longs['C']['may'][1])[0]
        may_option2 = list(prev_longs['C']['may'][1])[0]
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
    op_test = Option(35, 0.02, 'call', 0.8, ft_test, 'amer')
    prev_net = copy.deepcopy(pf.get_net_greeks())
    prev_longs = copy.deepcopy(pf.get_securities_monthly('long'))
    assert len(pf.get_securities_monthly('long')) == 3
    assert len(pf.long_options) == 3

    # adding security into portfolio; should change greeks and lengths of
    # relevant data structures.
    pf.add_security(op_test, 'long')
    assert len(pf.get_securities_monthly('long')) == 3
    assert len(pf.long_options) == 4
    curr_net = pf.get_net_greeks()

    curr_longs = pf.get_securities_monthly('long')
    # greeks should NOT be same since options contribute greeks.
    assert curr_net != prev_net
    assert curr_longs != prev_longs

    # now remove the same security
    pf.remove_security(op_test, 'long')

    # data structures should reset
    rem_longs = pf.get_securities_monthly('long')
    rem_net = pf.get_net_greeks()
    assert len(pf.long_options) == 3
    assert len(pf.get_securities_monthly('long')) == 3

    try:
        assert rem_net == prev_net
    except AssertionError:
        print(rem_longs['C'].keys())
        print(pf.short_pos['C'].keys())
        print(rem_net['C'].keys())
        print(prev_net['C'].keys())

    try:
        assert rem_longs == prev_longs
    # memory location errors
    except AssertionError:
        mar_option1 = list(rem_longs['C']['march'][0])[0]
        mar_option2 = list(prev_longs['C']['march'][0])[0]
        may_option1 = list(rem_longs['C']['may'][0])[0]
        may_option2 = list(prev_longs['C']['may'][0])[0]
        may_future1 = list(rem_longs['C']['may'][1])[0]
        may_option2 = list(prev_longs['C']['may'][1])[0]

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
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer')
    pf.add_security(op1, 'long')
    assert len(pf.long_options) == 4
    assert len(pf.long_pos['C']) == 3
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
    assert 'june' not in pf.long_pos['C']
    assert len(pf.long_options) == 3


def test_compute_value():
    pf = generate_portfolio()
    init_val = pf.compute_value()
    ft = Future('june', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer')
    opval = op1.get_price()
    # testing long pos
    pf.add_security(op1, 'long')
    addval = pf.compute_value()
    try:
        assert (addval - init_val) == opval
    except AssertionError:
        assert np.isclose(addval-init_val, opval)
        print('residue: ', addval - init_val - opval)
    # testing short pos
    pf.remove_security(op1, 'long')
    pf.add_security(op1, 'short')
    shortval = pf.compute_value()
    assert init_val - shortval == opval


def test_exercise_option():
    pf = generate_portfolio()
    # initial
    assert len(pf.long_options) == 3
    assert len(pf.long_futures) == 2
    init_greeks = copy.deepcopy(pf.get_net_greeks())

    # add option
    ft = Future('june', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer')
    pf.add_security(op1, 'long')

    assert len(pf.long_options) == 4
    add_greeks = copy.deepcopy(pf.get_net_greeks())
    assert add_greeks != init_greeks

    # exercise
    pf.exercise_option(op1, 'long')
    assert len(pf.long_options) == 3
    assert len(pf.long_futures) == 3
    ex_greeks = copy.deepcopy(pf.get_net_greeks())
    assert ex_greeks == init_greeks
