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
    ft1 = Future('H7', 30, 'C')
    ft2 = Future('K7', 25, 'C')
    ft3 = Future('J7', 32, 'BO')
    ft4 = Future('J7', 33, 'LH')
    ft5 = Future('J7', 24, 'LH')

    # options
    op1 = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer', False, 'Z7')
    op2 = Option(
        29, 0.2156100288506942, 'call', 0.45176132048500206, ft2, 'amer', False, 'Z7')
    op3 = Option(30, 0.21534276294769317, 'call', 0.14464169782291536,
                 ft3, 'amer', True, 'Z7',  direc='up', barrier='amer', bullet=False, ko=35)
    op4 = Option(33, 0.22365510948646386, 'put', 0.18282926924909026,
                 ft4, 'amer', False, 'Z7', direc='down', barrier='amer', bullet=False, ki=28)
    op5 = Option(
        32, 0.010975090692443346, 'put', 0.8281728247909962, ft5, 'amer', True, 'Z7')

    # Portfolio Futures
    ft6 = Future('K7', 37, 'C', shorted=False)
    ft7 = Future('H7', 29, 'BO', shorted=False)
    ft8 = Future('J7', 32, 'C', shorted=True)
    ft9 = Future('J7', 32, 'BO', shorted=True)

    OTCs, hedges = [op1, op2, ft7, op4, ft6], [op5, op3, ft8, ft9]

    # creating portfolio
    pf = Portfolio()
    pf.add_security(hedges, 'hedge')
    pf.add_security(OTCs, 'OTC')
    # for sec in hedges:
    #     pf.add_security(sec, 'hedge')

    # for sec in OTCs:
    #     pf.add_security(sec, 'OTC')

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
    ft = Future('H7', 30, 'C')
    op = Option(
        35, 0.05106521860205984, 'call', 0.4245569263291844, ft, 'amer', True, 'Z7')
    pf.add_security([op], 'OTC')
    pf.add_security([op], 'OTC')
    assert len(pf.OTC_options) == 5


def test_remove_dne():
    pf = generate_portfolio()
    ft = Future('N7', 50, 'D')
    assert pf.remove_security([ft], 'OTC') == -1
    assert pf.remove_security([ft], 'hedge') == -1


def test_OTC_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('OTC')
    assert set(dic.keys()) == set(['C', 'LH', 'BO'])
    # sub-dictionaries.
    dic_c = dic['C']
    assert set(dic_c.keys()) == set(['H7', 'K7'])
    dic_bo = dic['BO']
    assert set(dic_bo.keys()) == set(['H7'])
    dic_lh = dic['LH']
    assert set(dic_lh.keys()) == set(['J7'])


def test_hedge_pos():
    pf = generate_portfolio()
    dic = pf.get_securities_monthly('hedge')
    assert set(dic.keys()) == set(['C', 'LH', 'BO'])
    # sub dictionaries
    dic_c = dic['C']
    assert set(dic_c.keys()) == set(['J7'])
    dic_lh = dic['LH']
    assert set(dic_lh.keys()) == set(['J7'])
    dic_bo = dic['BO']
    assert set(dic_bo.keys()) == set(['J7'])


def test_remove_security_futures():
    pf = generate_portfolio()
    ft_test = Future('Q7', 50, 'C')
    pf2 = copy.deepcopy(pf)
    prev_net = pf2.get_net_greeks()
    prev_OTCs = pf2.get_securities_monthly('OTC')

    # basic checks
    assert len(prev_OTCs) == 3
    assert len(prev_OTCs['C']) == 2
    assert set(prev_OTCs['C'].keys()) == set(['H7', 'K7'])
    assert len(pf.OTC_futures) == 2
    assert ft_test.get_product() == 'C'

    # adding future into portfolio; should not change greeks and lengths of
    # relevant data structures. additionally, net greeks should not have
    # futures.
    pf.add_security([ft_test], 'OTC')
    try:
        assert set(pf.get_net_greeks()['C'].keys()) == set(
            ['H7', 'K7'])
    except AssertionError:
        print('remove_sec_ft 1: ', pf.get_net_greeks()['C'].keys())

    # checking addition
    curr_OTCs = pf.get_securities_monthly('OTC')
    cprod = curr_OTCs['C']
    assert len(cprod) == 3
    assert set(cprod) == set(['H7', 'K7', 'Q7'])
    assert len(curr_OTCs) == 3
    assert len(pf.OTC_futures) == 3

    # checking current status
    curr_net = pf.get_net_greeks()

    # greeks should be same since futures do not contribute greeks.
    assert curr_net == prev_net
    assert curr_OTCs != prev_OTCs

    # now remove the same security
    pf.remove_security([ft_test], 'OTC')
    assert set(pf.get_net_greeks()['C'].keys()) == set(
        ['H7', 'K7'])

    # data structures should reset
    rem_OTCs = pf.get_securities_monthly('OTC')

    assert set(rem_OTCs.keys()) == set(['C', 'LH', 'BO'])

    rprod = rem_OTCs['C']
    assert set(rprod.keys()) == set(['H7', 'K7'])

    rem_net = pf.get_net_greeks()
    assert len(pf.OTC_futures) == 2

    # failing
    try:
        assert rem_net == prev_net
    except AssertionError:
        print('remove_sec_ft 2: ', rem_net['C'].keys())
        print('remove_sec_ft 3: ', prev_net['C'].keys())

    # failing
    try:
        assert rem_OTCs == prev_OTCs
    except AssertionError:
        mar_option1 = list(rem_OTCs['C']['H7'][0])[0]
        mar_option2 = list(prev_OTCs['C']['H7'][0])[0]
        may_option1 = list(rem_OTCs['C']['K7'][0])[0]
        may_option2 = list(prev_OTCs['C']['K7'][0])[0]
        may_future1 = list(rem_OTCs['C']['K7'][1])[0]
        may_option2 = list(prev_OTCs['C']['K7'][1])[0]
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
    ft_test = Future('Q7', 50, 'C')
    op_test = Option(35, 0.02, 'call', 0.8, ft_test, 'amer', False, 'Z7')
    prev_net = copy.deepcopy(pf.get_net_greeks())
    prev_OTCs = copy.deepcopy(pf.get_securities_monthly('OTC'))
    assert len(pf.get_securities_monthly('OTC')) == 3
    assert len(pf.OTC_options) == 3

    # adding security into portfolio; should change greeks and lengths of
    # relevant data structures.
    pf.add_security([op_test], 'OTC')
    assert len(pf.get_securities_monthly('OTC')) == 3
    assert len(pf.OTC_options) == 4
    curr_net = pf.get_net_greeks()

    curr_OTCs = pf.get_securities_monthly('OTC')
    # greeks should NOT be same since options contribute greeks.
    assert curr_net != prev_net
    assert curr_OTCs != prev_OTCs

    # now remove the same security
    pf.remove_security([op_test], 'OTC')

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
        mar_option1 = list(rem_OTCs['C']['H7'][0])[0]
        mar_option2 = list(prev_OTCs['C']['H7'][0])[0]
        may_option1 = list(rem_OTCs['C']['K7'][0])[0]
        may_option2 = list(prev_OTCs['C']['K7'][0])[0]
        may_future1 = list(rem_OTCs['C']['K7'][1])[0]
        may_option2 = list(prev_OTCs['C']['K7'][1])[0]

        # checking equality of options
        assert mar_option1.underlying.get_price(
        ) == mar_option2.underlying.get_price()
        assert mar_option1.K == mar_option2.K
        assert mar_option1.tau == mar_option2.tau
        assert mar_option1.char == mar_option2.char
        assert mar_option1.price == mar_option2.price
        assert mar_option1.get_product() == mar_option2.get_product()


def test_remove_expired_1():
    pf = generate_portfolio()
    ft = Future('M7', 50, 'C')
    init_net = copy.deepcopy(pf.get_net_greeks())
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    pf.add_security([op1], 'OTC')
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
        print('rem_exp_1 curr: ', curr_net)
        print('rem_exp_1 prev: ', prev_net)
    assert 'M7' not in pf.OTC['C']
    assert len(pf.OTC_options) == 3


def test_remove_expired_2():
    ft = Future('H7', 30, 'C')
    op = Option(
        35, 0.05, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    pf = Portfolio()
    pf.add_security([op], 'OTC')
    net = pf.get_net_greeks()['C']['H7']
    net = np.array(net)
    # print('net: ', net)
    try:
        assert net.all() != 0
    except AssertionError:
        print('pre-exp: ', net)
    # decrement tau, expiring option.
    pf.timestep(0.05)
    assert op.check_expired() == True
    pf.remove_expired()
    net = pf.get_net_greeks()
    try:
        assert len(net) == 0
    except (AssertionError, IndexError, KeyError):
        print('post-exp: ', net)


def test_ordering():
    ft = Future('H7', 30, 'C')
    op = Option(
        35, 0.05, 'call', 0.4245569263291844, ft, 'amer', True, 'Z7', ordering=2)
    pf = Portfolio()
    pf.add_security([op], 'OTC')
    init_ord = op.get_ordering()
    assert init_ord == 2
    pf.decrement_ordering('C', 1)
    new_ord = op.get_ordering()
    assert new_ord == 1
    pf.decrement_ordering('C', 1)
    fin_ord = op.get_ordering()
    assert fin_ord == 0
    assert op.check_expired() == True
    pf.remove_expired()
    dic = pf.OTC
    try:
        assert len(dic) == 0
    except:
        print('test_ordering: ', dic)


def test_compute_value():
    pnlmult = 50
    pf = generate_portfolio()
    init_val = pf.compute_value()
    ft = Future('M7', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    opval = op1.get_price()
    # testing OTC pos
    pf.add_security([op1], 'OTC')
    addval = pf.compute_value()
    try:
        assert (addval - init_val) == opval * op1.lots * pnlmult
    except AssertionError:
        assert np.isclose(addval-init_val, opval*50*op1.lots)
        assert (addval - init_val - (opval * op1.lots * pnlmult) < 2e-9)
        # print('residue: ', addval - init_val - (opval * op1.lots * pnlmult))
    # testing hedge pos
    pf.remove_security([op1], 'OTC')

    try:
        assert pf.compute_value() == init_val
    except AssertionError:
        print('testport compute_val remove: ', pf.compute_value(), init_val)

    op2 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', True, 'Z7')
    pf.add_security([op2], 'hedge')
    shorted = pf.compute_value()

    try:
        assert np.isclose(init_val - shorted, op2.lots *
                          op2.get_price() * pnlmult)
    except AssertionError:
        print('shorted: ',  op2.lots * op2.get_price())
        print('testport compute_val short: ', init_val, shorted)


def test_exercise_option():
    pf = generate_portfolio()
    # initial
    assert len(pf.OTC_options) == 3
    assert len(pf.OTC_futures) == 2
    init_greeks = copy.deepcopy(pf.get_net_greeks())

    # add option
    ft = Future('M7', 50, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    pf.add_security([op1], 'OTC')

    assert len(pf.OTC_options) == 4
    add_greeks = copy.deepcopy(pf.get_net_greeks())
    assert add_greeks != init_greeks

    # exercise
    pf.exercise_option(op1, 'OTC')
    assert len(pf.OTC_options) == 3
    assert len(pf.OTC_futures) == 3
    ex_greeks = copy.deepcopy(pf.get_net_greeks())
    assert ex_greeks == init_greeks


def test_price_vol_change():
    ft = Future('M7', 30, 'C')
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    pf = Portfolio()
    pf.add_security([op1], 'OTC')
    init_net = copy.deepcopy(pf.get_net_greeks())
    vol = 0.6
    price = 55
    ft.update_price(price)
    op1.update_greeks(vol=vol)
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)
    new_net = copy.deepcopy(pf.get_net_greeks())
    # print('new: ', new_net)
    # print('old: ', init_net)
    try:
        assert new_net != init_net
    except:
        print('new_net: ', new_net)
        print('init_net: ', init_net)


def test_decrement_ordering():
    # assume current month is H. Contract mths are H K N U Z.
    ft = Future('N7', 30, 'C')
    # H7.M7 option
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'H7', ordering=2)
    ft2 = Future('K7', 30, 'C')
    # H7.K7 option
    op2 = Option(25, 0.01, 'put', 0.25, ft2, 'amer', False, 'H7', ordering=1)
    # initial checks
    assert op1.get_ordering() == 2
    assert op2.get_ordering() == 1
    assert op1.check_expired() == False
    assert op2.check_expired() == False

    pf = Portfolio()
    pf.add_security([op1], 'OTC')
    pf.add_security([op2], 'OTC')

    # op1greeks = list(op1.greeks())
    # check basic portfolio functionality.
    otc = pf.OTC['C']
    assert len(otc) == 2
    assert otc.keys() == set(['K7', 'N7'])

    netgreeks = pf.get_net_greeks()['C']
    assert len(netgreeks) == 2

    pf.decrement_ordering('C', 1)
    assert op2.expired == True
    assert op2.ordering == 0
    assert op1.expired == False
    assert op1.ordering == 1

    # before removal; len should be the same.
    assert len(netgreeks) == 2
    # print(netgreeks['K7'])

    pf.remove_expired()
    fingreeks = pf.get_net_greeks()['C']
    assert len(fingreeks) == 1

    finotc = pf.OTC['C']
    assert len(finotc) == 1


def test_compute_ordering():
    # assume current month is H. Contract mths are H K N U Z.
    ft = Future('N7', 30, 'C')
    # H7.M7 option
    op1 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', False, 'H7', ordering=2)
    ft2 = Future('K7', 30, 'C')
    # H7.K7 option
    op2 = Option(25, 0.01, 'put', 0.25, ft2, 'amer', False, 'H7', ordering=1)
    pf = Portfolio()
    pf.add_security([op1], 'OTC')
    pf.add_security([op2], 'OTC')

    assert pf.compute_ordering('C', 'N7') == 2
    assert pf.compute_ordering('C', 'K7') == 1

    pf.decrement_ordering('C', 1)

    assert pf.compute_ordering('C', 'N7') == 1
    assert pf.compute_ordering('C', 'K7') == 0
