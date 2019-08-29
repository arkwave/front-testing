"""
File Name      : test_portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 16/3/2017
Python version : 3.5
Description    : File contains tests for the Portfolio class methods in portfolio.py

"""
import pandas as pd
from scripts.util import create_straddle, combine_portfolios, create_underlying, assign_hedge_objects
from scripts.fetch_data import grab_data
from scripts.portfolio import Portfolio
from scripts.classes import Option, Future
from scripts.hedge import Hedge
from scripts.simulation import roll_over
from collections import OrderedDict
import copy
import numpy as np
import unittest as un
# import p# print


######### variables ################
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']

####################################

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=False)
# print(vdf.value_date.min())
# print(start_date)
# assert 1 == 0


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
                 ft3, 'amer', True, 'Z7',  direc='up', barrier='amer', bullet=True, ko=35)
    op4 = Option(33, 0.22365510948646386, 'put', 0.18282926924909026,
                 ft4, 'amer', False, 'Z7', direc='down', barrier='amer', bullet=True, ki=28)
    op5 = Option(
        32, 0.010975090692443346, 'put', 0.8281728247909962, ft5, 'amer', True, 'Z7')

    # Portfolio Futures
    ft6 = Future('K7', 37, 'C', shorted=False)
    ft7 = Future('H7', 29, 'BO', shorted=False)
    ft8 = Future('J7', 32, 'C', shorted=True)
    ft9 = Future('J7', 32, 'BO', shorted=True)

    OTCs, hedges = [op1, op2, ft7, op4, ft6], [op5, op3, ft8, ft9]

    # creating portfolio
    pf = Portfolio(None)
    pf.add_security(hedges, 'hedge')
    pf.add_security(OTCs, 'OTC')

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
    tc = un.TestCase
    with tc.assertRaises(pf.remove_security, ValueError) as e1:
        pf.remove_security([ft], 'OTC')
    with tc.assertRaises(pf.remove_security, ValueError) as e2:
        pf.remove_security([ft], 'hedge')

    assert isinstance(e1.exception, ValueError)
    assert isinstance(e2.exception, ValueError)


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
    # try:
    assert set(pf.get_net_greeks()['C'].keys()) == set(
        ['H7', 'K7'])
    # except AssertionError:
    #     # # print('remove_sec_ft 1: ', pf.get_net_greeks()['C'].keys())

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
    # try:
    assert rem_net == prev_net
    # except AssertionError:
    #     # # print('remove_sec_ft 2: ', rem_net['C'].keys())
    #     # # print('remove_sec_ft 3: ', prev_net['C'].keys())

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

    # try:
    assert rem_net == prev_net
    # except AssertionError:
    #     # # print(rem_OTCs['C'].keys())
    #     # # print(pf.hedge_pos['C'].keys())
    #     # # print(rem_net['C'].keys())
    #     # # print(prev_net['C'].keys())

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
    assert len(pf.OTC['C']) == 2
    curr_net = copy.deepcopy(pf.get_net_greeks())
    # try:
    assert curr_net != prev_net
    # except AssertionError:
    #     # # print('rem_exp_1 curr: ', curr_net)
    #     # # print('rem_exp_1 prev: ', prev_net)
    #     # # print('rem_exp_1 init: ', init_net)
    assert 'M7' not in pf.OTC['C']
    assert len(pf.OTC_options) == 3


def test_remove_expired_2():
    ft = Future('H7', 30, 'C')
    op = Option(
        35, 0.05, 'call', 0.4245569263291844, ft, 'amer', False, 'Z7')
    pf = Portfolio(None)
    pf.add_security([op], 'OTC')
    net = pf.get_net_greeks()['C']['H7']
    net = np.array(net)
    # # # print('net: ', net)
    # try:
    assert net.all() != 0
    # except AssertionError:
    #     # # print('pre-exp: ', net)
    # decrement tau, expiring option.
    pf.timestep(0.05)
    assert op.check_expired()
    pf.remove_expired()
    net = pf.get_net_greeks()
    # try:
    assert len(net) == 0
    # except (AssertionError, IndexError, KeyError):
    #     # # print('post-exp: ', net)


def test_ordering():
    ft = Future('H7', 30, 'C', ordering=2)
    op = Option(
        35, 0.05, 'call', 0.4245569263291844, ft, 'amer', True, 'Z7', ordering=2)
    pf = Portfolio(None)
    pf.add_security([op], 'OTC')
    init_ord = op.get_ordering()
    assert init_ord == 2
    pf.decrement_ordering('C', 1)
    new_ord = op.get_ordering()
    assert new_ord == 1
    pf.decrement_ordering('C', 1)
    assert op.ordering == 0


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
        # # # print('residue: ', addval - init_val - (opval * op1.lots * pnlmult))
    # testing hedge pos
    pf.remove_security([op1], 'OTC')

    # try:
    assert pf.compute_value() == init_val
    # except AssertionError:
    #     # # print('testport compute_val remove: ', pf.compute_value(), init_val)

    op2 = Option(
        35, 0.01, 'call', 0.4245569263291844, ft, 'amer', True, 'Z7')
    pf.add_security([op2], 'hedge')
    shorted = pf.compute_value()

    # try:
    assert np.isclose(init_val - shorted, op2.lots *
                      op2.get_price() * pnlmult)
    # except AssertionError:
    #     # # print('shorted: ',  op2.lots * op2.get_price())
    #     # # print('testport compute_val short: ', init_val, shorted)


def test_exercise_option():
    pf = generate_portfolio()
    # initial
    assert len(pf.OTC_options) == 3
    assert len(pf.OTC_futures) == 2
    init_greeks = copy.deepcopy(pf.get_net_greeks())

    # add option
    ft = Future('M7', 50, 'C')
    op1 = Option(35, 0.01, 'call', 0.4245569263291844,
                 ft, 'amer', False, 'Z7')
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
    op1 = Option(35, 0.01, 'call', 0.4245569263291844,
                 ft, 'amer', False, 'Z7')
    pf = Portfolio(None)
    pf.add_security([op1], 'OTC')
    init_net = copy.deepcopy(pf.get_net_greeks())
    vol = 0.6
    price = 55
    ft.update_price(price)
    op1.update_greeks(vol=vol)
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)
    new_net = copy.deepcopy(pf.get_net_greeks())
    # # # print('new: ', new_net)
    # # # print('old: ', init_net)
    # try:
    assert new_net != init_net
    # except AssertionError:
    #     # # print('new_net: ', new_net)
    #     # # print('init_net: ', init_net)


def test_decrement_ordering():
    # assume current month is H. Contract mths are H K N U Z.
    ft = Future('N7', 30, 'C', ordering=2)
    # H7.M7 option
    op1 = Option(35, 0.01, 'call', 0.4245569263291844,
                 ft, 'amer', False, 'H7', ordering=2)
    ft2 = Future('K7', 30, 'C', ordering=1)
    # H7.K7 option
    op2 = Option(25, 0.01, 'put', 0.25, ft2,
                 'amer', False, 'H7', ordering=1)
    # initial checks
    assert op1.get_ordering() == 2
    assert op2.get_ordering() == 1
    assert not op1.check_expired()
    assert not op2.check_expired()

    pf = Portfolio(None)
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
    assert op2.ordering == 0
    assert not op1.expired
    assert op1.ordering == 1

    # before removal; len should be the same.
    assert len(netgreeks) == 2
    # # # print(netgreeks['K7'])


def test_compute_ordering():
    # assume current month is H. Contract mths are H K N U Z.
    ft = Future('N7', 30, 'C', ordering=2)
    # H7.M7 option
    op1 = Option(35, 0.01, 'call', 0.4245569263291844, ft, 'amer',
                 False, 'H7', ordering=ft.get_ordering())
    ft2 = Future('K7', 30, 'C', ordering=1)
    # H7.K7 option
    op2 = Option(25, 0.01, 'put', 0.25, ft2, 'amer',
                 False, 'H7', ordering=ft2.get_ordering())
    pf = Portfolio(None)
    pf.add_security([op1], 'OTC')
    pf.add_security([op2], 'OTC')

    # # # print('actual: ', pf.compute_ordering('C', 'N7'))

    assert pf.compute_ordering('C', 'N7') == 2
    assert pf.compute_ordering('C', 'K7') == 1

    pf.decrement_ordering('C', 1)

    assert pf.compute_ordering('C', 'N7') == 1
    assert pf.compute_ordering('C', 'K7') == 0


""" To be added:
1) Testing families access etc.
2) Testing refresh.
3) Test family containing.
"""


def comp_portfolio(refresh=False):
    # creating the options.
    ccops = create_straddle('CC  Z7.Z7', vdf, pdf,
                            False, 'atm', greek='theta', greekval=10000)
    qcops = create_straddle('QC  Z7.Z7', vdf, pdf,
                            True, 'atm', greek='theta', greekval=10000)
    # create the hedges.
    gen_hedges = OrderedDict({'delta': [['static', 'zero', 1]]})
    cc_hedges_s = {'delta': [['static', 'zero', 1],
                             ['roll', 50, 1, (-10, 10)]]}

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    qc_hedges = {'delta': [['roll', 50, 1, (-15, 15)]],
                 'theta': [['bound', (9000, 11000), 1, 'straddle',
                            'strike', 'atm', 'uid']]}
    cc_hedges_s = OrderedDict(cc_hedges_s)
    cc_hedges_c = OrderedDict(cc_hedges_c)
    qc_hedges = OrderedDict(qc_hedges)

    # create one simple and one complex portfolio.
    pf_simple = Portfolio(cc_hedges_s, name='cc_simple')
    pf_simple.add_security(ccops, 'OTC')

    pfcc = Portfolio(cc_hedges_c, name='cc_comp')
    pfcc.add_security(ccops, 'OTC')
    pfqc = Portfolio(qc_hedges, name='qc_comp')
    pfqc.add_security(qcops, 'OTC')

    pf_comp = combine_portfolios(
        [pfcc, pfqc], hedges=gen_hedges, refresh=refresh, name='full')

    pf_simple = assign_hedge_objects(pf_simple, vdf=vdf, pdf=pdf)
    pfcc = assign_hedge_objects(pfcc, vdf=vdf, pdf=pdf)
    pfqc = assign_hedge_objects(pfqc, vdf=vdf, pdf=pdf)
    pf_comp = assign_hedge_objects(pf_comp, vdf=vdf, pdf=pdf)

    assert pf_simple.get_hedger() is not None
    assert pfcc.get_hedger() is not None
    assert pfqc.get_hedger() is not None
    assert pf_comp.get_hedger() is not None

    return pf_simple, pf_comp, ccops, qcops, pfcc, pfqc


def test_refresh():
    # checking that values are passed the right way during refresh.
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(False)

    init_net = pf_comp.get_net_greeks()

    pf_comp.refresh()

    post_refresh = pf_comp.get_net_greeks()

    assert init_net != post_refresh

    pftest = Portfolio(None)
    pftest.add_security(ccops + qcops, 'OTC')

    # check net greeks.
    assert pftest.get_net_greeks() == pf_comp.get_net_greeks()

    # check options.
    assert pftest.get_all_options() == pf_comp.get_all_options()

    # check OTC and Hedge dictionaries.
    assert pftest.OTC == pf_comp.OTC
    assert pftest.hedges == pf_comp.hedges

    # check the individual constituent families to make sure refresh doesn't
    # affect them
    qcfam = [x for x in pf_comp.families if x.get_unique_products() == {'QC'}][
        0]
    qctest = Portfolio(None)
    qctest.add_security(qcops, 'OTC')

    assert qcfam.OTC == qctest.OTC
    assert qcfam.net_greeks == qctest.net_greeks
    assert qcfam.OTC_options == qctest.OTC_options

    ccfam = [x for x in pf_comp.families if x.get_unique_products() == {'CC'}][
        0]
    cctest = Portfolio(None)
    cctest.add_security(ccops, 'OTC')

    assert ccfam.OTC == cctest.OTC
    assert ccfam.net_greeks == cctest.net_greeks
    assert ccfam.OTC_options == cctest.OTC_options


def test_refresh_advanced():
    # more complicated tests for refresh.
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # first: time step forward and back should not change nets.
    pf = copy.deepcopy(pf_comp)
    init_otc = pf.OTC.copy()
    init_net = copy.deepcopy(pf.get_net_greeks())

    pf.timestep(1/365)

    assert pf.get_net_greeks() != init_net

    pf.timestep(-1/365)
    new_otc = pf.OTC.copy()
    new_net = pf.get_net_greeks()
    assert new_net == init_net
    assert init_otc == new_otc


def test_family_containing():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    cop1, cop2 = ccops
    qop1, qop2 = qcops
    assert pf_comp.get_family_containing(cop1) == pfcc
    assert pf_comp.get_family_containing(cop2) == pfcc
    assert pf_comp.get_family_containing(qop1) == pfqc
    assert pf_comp.get_family_containing(qop2) == pfqc


def test_removing_from_composite():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    cop1, cop2 = ccops
    qop1, qop2 = qcops

    init = pf_comp.net_greeks.copy()
    # # # print('initial net: ', pf_comp.net_greeks)

    # first test
    pf_comp.remove_security([cop1], 'OTC')

    assert cop1 not in pfcc.OTC_options
    assert cop1 not in pfcc.OTC['CC']['Z7'][0]

    # intermediate test
    x1 = pfcc.net_greeks.copy()
    x1.update(pfqc.net_greeks.copy())

    # # # print('x1: ', x1)
    # # # print('actual: ', pf_comp.net_greeks)

    assert x1 == pf_comp.get_net_greeks()

    # # # print('pre-add net: ', pf_comp.net_greeks)
    pfcc.add_security([cop1], 'OTC')

    assert pf_comp.net_greeks != init

    # # # print('bef refresh: ', pf_comp.net_greeks)
    pf_comp.refresh()

    # print('pf_comp before QC removal: ', pf_comp)
    # print('pfqc before qc_removal: ', pfqc)

    assert pf_comp.net_greeks == init
    # # # print('aft refresh: ', pf_comp.net_greeks)
    # second test.
    pf_comp.remove_security([qop1], 'OTC')
    assert qop1 not in pfqc.OTC_options
    assert qop1 not in pfqc.OTC['QC']['Z7'][0]

    # net greeks test.
    # try:
    x = pfcc.net_greeks.copy()
    x.update(pfqc.net_greeks.copy())
    assert x == pf_comp.net_greeks
    # except AssertionError:
    #     # # print('FAILURE: test_portfolio.test_removing_from_composite')
    #     # # print('netnewgreeks: ', x)
    #     # # print('actual: ', pf_comp.net_greeks)


def test_adding_to_composites():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # get initialization parameters
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # first: add futures to overall pf_comp. should not appear in either pfcc
    # or pfqc
    ccdelta2 = int(pfcc.net_greeks['CC']['Z7'][0])
    shorted = False if ccdelta2 < 0 else True
    ccft2, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=abs(ccdelta2), shorted=shorted)
    pf = copy.deepcopy(pf_comp)

    cc_pf = [fam for fam in pf.families if fam.name == 'cc_comp'][0]
    qc_pf = [fam for fam in pf.families if fam.name == 'qc_comp'][0]

    pf.add_security([ccft2], 'hedge')
    pf.refresh()
    try:
        assert ccft2 not in cc_pf.hedges['CC']['Z7'][1]
    except KeyError:
        assert 'CC' not in cc_pf.hedges

    assert ccft2 in pf.hedges['CC']['Z7'][1]
    assert 'CC' not in qc_pf.hedges

    # get the greeks for the heck of it.
    init_greeks = copy.deepcopy(pf.get_net_greeks())

    # print('============== first add ==================')
    # second test: add an option to pfqc, check if it is updated in pf.
    qc_straddle = create_straddle('QC  Z7.Z7', r_vdf, r_pdf,
                                  True, 'atm', greek='theta', greekval=2000, date=date)
    qc_pf.add_security(qc_straddle, 'OTC')

    pf.refresh()

    assert len(pf.OTC['QC']['Z7'][0]) == 4
    assert len(pf.OTC_options) == 6
    for op in qc_straddle:
        assert op in pf.OTC['QC']['Z7'][0]
        assert op in pf.OTC_options
        assert op in qc_pf.OTC_options
        assert op in qc_pf.OTC['QC']['Z7'][0]

    # third: remove the QC options
    # print('================== second: remove added QC ====================')
    pf.remove_security(qc_straddle, 'OTC')
    assert len(pf.OTC['QC']['Z7'][0]) == 2
    assert len(pf.OTC_options) == 4
    for op in qc_straddle:
        assert op not in pf.OTC['QC']['Z7'][0]
        assert op not in pf.OTC_options
        assert op not in qc_pf.OTC_options
        assert op not in qc_pf.OTC['QC']['Z7'][0]
    # print('============================== end =============================')
    assert pf.get_net_greeks() == init_greeks

    # fourth: remove the cc options, check to see if cc_pf is empty.
    ccops2 = [op for op in pf.OTC_options if op.get_product() == 'CC']
    # print('====================== third: remove init cc =====================')
    pf.remove_security(ccops2, 'OTC')
    pf.refresh()
    assert 'CC' not in pf.OTC
    assert 'CC' not in cc_pf.OTC
    assert cc_pf.empty()
    # print('============================== end =============================')


def test_degenerate_case():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    init_net_greeks = pf_comp.get_net_greeks().copy()

    # # # print('----- removing -----')
    pf_comp.remove_security(ccops, 'OTC')
    # # # print('------ done removing ------')

    try:
        assert pfcc.empty()
    except AssertionError as e:
        raise AssertionError(
            'test_portfolio.test_degenerate_case - failure: ', pfcc.OTC, pfcc.hedges) from e

    # check to see if 'CC' is still in the dict.
    assert 'CC' not in pf_comp.OTC
    assert 'CC' not in pf_comp.net_greeks
    assert 'CC' not in pf_comp.hedges

    pfcc.add_security(ccops, 'OTC')

    # check that net_greeks are still that of qc before refresh.
    assert pf_comp.net_greeks == pfqc.net_greeks

    pf_comp.refresh()

    # check to ensure that the net greeks stay the same
    assert init_net_greeks == pf_comp.net_greeks


def test_family_timestep():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    cc_old_ttm = ccops[0].tau
    qc_old_ttm = qcops[0].tau
    pf_comp.timestep(1/365)

    for op in pfcc.OTC_options:
        assert op.tau == cc_old_ttm - (1/365)

    for op in pfqc.OTC_options:
        assert op.tau == qc_old_ttm - (1/365)


def test_get_volid_mappings():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    x = pf_comp.get_volid_mappings()

    assert set(x['CC  Z7.Z7']) == set(ccops)
    assert set(x['QC  Z7.Z7']) == set(qcops)

    y = pf_simple.get_volid_mappings()
    assert set(y['CC  Z7.Z7']) == set(ccops)


def test_get_unique_products():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    assert pf_comp.get_unique_products() == {'CC', 'QC'}


def test_timestep():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # # # print('pf.net_greeks: ', pf_simple.net_greeks)

    # testing simple portfolio
    pf = pf_simple
    init_val = pf.compute_value()
    init_netgreeks = copy.deepcopy(pf.get_net_greeks())
    # # # print('init_netgreeks: ', init_netgreeks)
    # timestep
    pf.timestep(1/365)
    # pf.update_sec_by_month(None, 'OTC', update=True)
    # pf.update_sec_by_month(None, 'hedge', update=True)

    assert pf.net_greeks != init_netgreeks
    assert pf.get_net_greeks() != init_netgreeks

    pf.timestep(-1/365)
    # pf.update_sec_by_month(None, 'OTC', update=True)
    # pf.update_sec_by_month(None, 'hedge', update=True)

    newval = pf.compute_value()
    new_netgreeks = pf.get_net_greeks().copy()
    # assert newval == init_val
    # try:
    assert init_netgreeks == new_netgreeks
    # except AssertionError:
    #     # # print('init: ', init_netgreeks)
    #     # # print('new: ', new_netgreeks)

    # test composite portfolios.
    init_val = pf_comp.compute_value()
    init_netgreeks = pf_comp.get_net_greeks().copy()
    # timestep
    pf_comp.timestep(1/365)
    pf_comp.timestep(-1/365)
    newval = pf_comp.compute_value()
    new_netgreeks = pf_comp.get_net_greeks().copy()
    assert newval == init_val
    assert init_netgreeks == new_netgreeks


def test_timestep_delta_hedging_simple():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # replicated the exact steps in simulation.
    # first: test simple portfolio.
    init_ttm = pf_simple.OTC_options[0].tau

    relevant_vols = vdf[vdf.value_date == vdf.value_date.min()]
    relevant_prices = pdf[pdf.value_date == pdf.value_date.min()]

    hedge_engine = Hedge(pf_simple, pf_simple.hedge_params,
                         relevant_vols, relevant_prices)

    # time step before hedging delta.
    pf = pf_simple
    pf.timestep(1/365)
    intermediate_ttm = pf.OTC_options[0].tau
    fee = hedge_engine.apply('delta')
    # reverse the timestep.
    pf.timestep(-1/365, allops=False)

    new_ttm = pf.OTC_options[0].tau

    assert new_ttm == init_ttm
    assert np.isclose(intermediate_ttm, new_ttm - 1/365)


def test_timestep_delta_hedging_comp():

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # replicated the exact steps in simulation.
    # first: test simple portfolio.

    pf = pf_comp

    cc_init_ttm = ccops[0].tau
    qc_init_ttm = qcops[0].tau

    relevant_vols = vdf[vdf.value_date == vdf.value_date.min()]
    relevant_prices = pdf[pdf.value_date == pdf.value_date.min()]

    hedge_engine = Hedge(pf, pf.hedge_params,
                         relevant_vols, relevant_prices)

    # time step before hedging delta.
    pf.timestep(1/365)

    pf.refresh()

    cc_intermediate_ttm = ccops[0].tau

    # # # print('cc_intermediate: ', cc_intermediate_ttm)
    qc_intermediate_ttm = qcops[0].tau

    fee = hedge_engine.apply('delta')

    pf.refresh()

    cc_ttm = ccops[0].tau
    qc_ttm = qcops[0].tau

    # # # print('cc_intermediate post hedge: ', cc_ttm)
    # ensure the refresh doesn't fuck with the ttm.
    assert cc_ttm == cc_intermediate_ttm
    assert qc_ttm == qc_intermediate_ttm

    # reverse the timestep.
    pf.timestep(-1/365, allops=False)

    cc_new = ccops[0].tau
    qc_new = qcops[0].tau

    # # # print('cc_new: ', cc_new)

    assert qc_new == qc_init_ttm
    assert cc_new == cc_init_ttm

    assert np.isclose(cc_intermediate_ttm, cc_new - 1/365)
    assert np.isclose(qc_intermediate_ttm, qc_new - 1/365)


def test_rollovers_OTC_representation_simple():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    pf = copy.deepcopy(pfcc)
    cchedges = create_straddle('CC  Z7.Z7', vdf, pdf,
                               True, 'atm', greek='theta', greekval=5000)
    pf.add_security(cchedges, 'hedge')

    pf.roll = True
    pf.ttm_tol = (ccops[0].tau * 365) + 1
    pf.refresh()

    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # # print('pf after rollovers: ', pf)

    assert 'Z7' not in pf.net_greeks['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.hedges['CC']

    assert 'H8' in pf.net_greeks['CC']
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.hedges['CC']


##########################################################################
##########################################################################
##########################################################################
##########################################################################


def test_refresh_securities_diff_pdts():
    """Summary: tests to ensure that adding futures into overall portfolio does not
    modify the dictionaries of constituent families. 
    Used: CC/QC, added in CC future, checked to ensure that Z7 exists in pf but not
    cc_pf or qc_pf """

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    pf = copy.deepcopy(pf_comp)

    cc_pf = [fam for fam in pf.get_families() if
             fam.get_unique_products() == set(['CC'])][0]
    qc_pf = [fam for fam in pf.get_families() if
             fam.get_unique_products() == set(['QC'])][0]

    ccops2 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf,
                             True, 'atm', greek='theta', greekval=5000, date=date)

    cc_pf.add_security(ccops2, 'OTC')

    # second: refresh shouldn't transfer securities around for whatever reason.
    # initializing the futures to zero out deltas.
    cc_delta = int(pf.net_greeks['CC']['Z7'][0])
    cc_shorted = False if cc_delta < 0 else True

    ccft, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=cc_delta, shorted=cc_shorted)

    pf.add_security([ccft], 'hedge')

    try:
        assert 'CC' not in cc_pf.hedges
    except AssertionError as e:
        raise AssertionError('cc_pf.hedges: ', cc_pf.hedges)
    assert 'QC' not in qc_pf.hedges
    try:
        assert 'Z7' in pf.hedges['CC']
    except AssertionError as e:
        raise AssertionError('pf.hedges: ', pf.hedges)

    pf.refresh()

    assert 'CC' not in cc_pf.hedges
    assert 'QC' not in qc_pf.hedges
    assert 'Z7' in pf.hedges['CC']


def test_refresh_securities_same_pdts():
    """Summary: tests to ensure that adding futures into overall portfolio does not
    modify the dictionaries of constituent families. 
    Used: CC_Z and CC_H, added in CC_H future, checked to ensure that H exists in pf but not
    Z_pf or H_pf """

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initializing the portfolios and options
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    pf1 = Portfolio(cc_hedges_c, name='cc_1')
    pf2 = Portfolio(cc_hedges_c, name='cc_2')

    straddle1 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf,
                                True, 'atm', greek='theta', greekval=5000, date=date)
    straddle2 = create_straddle('CC  H8.H8', r_vdf, r_pdf,
                                True, 'atm', greek='theta', greekval=5000, date=date)
    straddle3 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf,
                                False, 'atm', greek='theta', greekval=2500, date=date)

    pf1.add_security(straddle1, 'OTC')
    pf1.add_security(straddle3, 'hedge')

    pf2.add_security(straddle2, 'OTC')

    pf1.refresh()
    pf2.refresh()

    # create the super-portfolio and add hedging futures
    pf = combine_portfolios([pf1, pf2], hedges=None,
                            refresh=True, name='full')

    cc_delta = int(pf.net_greeks['CC']['H8'][0])
    cc_shorted = False if cc_delta < 0 else True

    ccft, _ = create_underlying(
        'CC', 'H8', r_pdf, date, lots=cc_delta, shorted=cc_shorted)

    # # print('ccft: ', str(ccft))

    # # print('pf1.hedges pre_add: ', pf1.hedges['CC'].keys())

    pf.add_security([ccft], 'hedge')

    # # print('pf1.hedges: ', pf1.hedges['CC'].keys())
    # # print('pf1.hedge_futures: ', [str(x) for x in pf1.hedge_futures])

    # adding hedging futures should trigger 'H8' in pf.hedges, but NOT in
    # either of the sub-portfolios.
    try:
        assert 'H8' in pf.hedges['CC']
    except AssertionError as e:
        raise AssertionError('pf.hedges: ', pf.hedges)

    assert 'CC' not in pf2.hedges

    try:
        assert 'H8' not in pf1.hedges['CC']
    except AssertionError as e:
        raise AssertionError('cc_pf.hedges: ', pf1.hedges)

    assert 'CC' not in pf2.hedges

    pf.refresh()

    try:
        assert 'H8' in pf.hedges['CC']
    except AssertionError as e:
        raise AssertionError('pf.hedges: ', pf.hedges)

    try:
        assert 'H8' not in pf1.hedges['CC']
    except AssertionError as e:
        raise AssertionError('cc_pf.hedges: ', pf1.hedges)

    assert 'CC' not in pf2.hedges


def test_add_futures_comp():
    """Adds future into composite portfolio and checks that future month is not added to 
    constituent family dictionaries. 
    """
    # print('================== init refresh ====================')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    # print('================== end init refresh ====================')
    # creating portfolio
    pf = copy.deepcopy(pf_comp)
    # initialization parameters
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    cc_pf = [fam for fam in pf.get_families() if fam.name == 'cc_comp'][0]

    cchedges = create_straddle('CC  Z7.Z7', vdf, pdf,
                               True, 'atm', greek='theta', greekval=5000, date=date)
    cc_pf.add_security(cchedges, 'hedge')

    # # print('ccpf pre-refresh: ', cc_pf)
    # # print('ccpf.OTC: ', cc_pf.OTC)
    # # print('ccpf.hedges: ', cc_pf.hedges)
    # print('=============== comp refresh ==================')
    pf.refresh()
    # print('=============== refresh completed ================')
    # # print('ccpf post-refresh: ', cc_pf)
    # # print('ccpf.OTC: ', cc_pf.OTC)
    # # print('ccpf.hedges: ', cc_pf.hedges)

    cc_delta = int(cc_pf.net_greeks['CC']['Z7'][0])
    cc_shorted = True if cc_delta > 0 else False

    ccft, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=abs(cc_delta), shorted=cc_shorted)
    # print('==================== begin add =================')
    pf.add_security([ccft], 'hedge')
    # print('===================== end add ==================')

    # print('============== final refresh ================')
    pf.refresh()
    # print('========== final refresh completed =============')

    try:
        assert ccft not in cc_pf.hedges['CC']['Z7'][1]
    except KeyError:
        assert 'CC' not in cc_pf.hedges

    # assert 'CC' not in cc_pf.hedges
    assert 'CC' in pf.hedges
    assert 'Z7' in pf.hedges['CC']


def test_rollovers_OTC_representation_comp_no_fts():
    # data for option/future construction.
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initial portfolio
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # isolate cc_porfolio, adding hedge options.
    pf = copy.deepcopy(pf_comp)
    cc_pf = [fam for fam in pf.families
             if fam.get_unique_products() == set(['CC'])][0]

    cchedges = create_straddle('CC  Z7.Z7', vdf, pdf,
                               True, 'atm', greek='theta', greekval=5000, date=date)

    cc_pf.add_security(cchedges, 'hedge')

    # setting roll conditions for this particular family.
    cc_pf.roll = True
    cc_pf.ttm_tol = (ccops[0].tau * 365) + 2

    # # print('cc_pf: ', cc_pf)

    pf.refresh()

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # # print('pf after rollovers: ', pf)
    # # print('cc_pf after rollovers: ', cc_pf)

    assert 'Z7' not in pf.net_greeks['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.hedges['CC']
    assert 'H8' in pf.net_greeks['CC']
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.hedges['CC']

    assert 'Z7' not in cc_pf.net_greeks['CC']
    assert 'Z7' not in cc_pf.OTC['CC']
    assert 'Z7' not in cc_pf.hedges['CC']
    assert 'H8' in cc_pf.net_greeks['CC']
    assert 'H8' in cc_pf.OTC['CC']
    assert 'H8' in cc_pf.hedges['CC']


def test_rollovers_OTC_representation_comp_fts():
    # data for option/future construction.
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initial portfolio
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    # isolate cc_porfolio, adding hedge options.
    pf = copy.deepcopy(pf_comp)
    cc_pf = [fam for fam in pf.families
             if fam.get_unique_products() == set(['CC'])][0]

    cchedges = create_straddle('CC  Z7.Z7', vdf, pdf,
                               True, 'atm', greek='theta', greekval=5000, date=date)

    cc_pf.add_security(cchedges, 'hedge')
    # setting roll conditions for this particular family.
    cc_pf.roll = True
    cc_pf.ttm_tol = (ccops[0].tau * 365) + 2

    pf.refresh()

    # creating futures to test close_out_deltas position
    cc_delta = int(cc_pf.net_greeks['CC']['Z7'][0])
    cc_shorted = True if cc_delta > 0 else False

    ccft, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=cc_delta, shorted=cc_shorted)

    # print('ccpf before futures: ', cc_pf.hedges)
    pf.add_security([ccft], 'hedge')
    # pf.refresh()
    # print('ccpf after futures: ', cc_pf.hedges)

    assert ccft not in cc_pf.hedge_futures
    try:
        assert cc_pf.hedge_futures == []
    except AssertionError as e:
        raise AssertionError("cc_pf hedge options: ",
                             cc_pf.hedge_options) from e

    assert ccft not in cc_pf.hedges['CC']['Z7'][1]

    # # print('ccpf after futures: ', cc_pf)
    pf.refresh()
    # print('ccpf after refresh: ', cc_pf)

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)
    pf.refresh()

    # print('pf after rollovers: ', pf)
    # print('pf.OTC: ', pf.OTC)
    # print('pf.hedge: ', pf.hedges)
    # # print('cc_pf after rollovers: ', cc_pf)
    try:
        assert 'Z7' not in pf.net_greeks['CC']
    except AssertionError as e:
        raise AssertionError('pf.net greeks: ', pf.net_greeks)
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.hedges['CC']
    assert 'H8' in pf.net_greeks['CC']
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.hedges['CC']

    assert 'Z7' not in cc_pf.OTC['CC']
    assert 'Z7' not in cc_pf.hedges['CC']
    assert 'Z7' not in cc_pf.net_greeks['CC']
    assert 'H8' in cc_pf.net_greeks['CC']
    assert 'H8' in cc_pf.OTC['CC']
    assert 'H8' in cc_pf.hedges['CC']


def test_get_all_options_comp():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    assert set(pf_comp.get_all_options()) == set(
        pfcc.OTC_options + pfqc.OTC_options)
    ccops = [x for x in pf_comp.get_all_options() if x.get_product() == 'CC']

    for op in ccops:
        op.underlying.update_price(50)
        op.update_greeks(vol=0.25)

    for op in pfcc.OTC_options:
        assert op.underlying.get_price() == 50
        assert op.vol == 0.25


def test_refresh_vol_price_updates():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)

    for ft in pf_comp.get_all_futures():
        ft.update_price(50)

    # print('pf after price pre refresh: ', pf_comp)
    pf_comp.refresh()
    # print('pf after price post refresh: ', pf_comp)

    for op in pf_comp.get_all_options():
        op.update_greeks(vol=0.25)

    # print('pf after vol pre refresh: ', pf_comp)
    pf_comp.refresh()
    # print('pf after vol post refresh: ', pf_comp)


def test_hedges_pf_updates():
    from scripts.util import assign_hedge_objects
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    pf = copy.deepcopy(pf_comp)
    # first: assign hedges to all.
    pf = assign_hedge_objects(pf)

    cc_pf = [fam for fam in pf.families if fam.name == 'cc_comp'][0]
    qc_pf = [fam for fam in pf.families if fam.name == 'qc_comp'][0]

    # assert hedger objects associated with each portfolio are not none
    cc_hedger, qc_hedger, pf_hedger = cc_pf.get_hedger(
    ), qc_pf.get_hedger(), pf.get_hedger()
    assert cc_hedger is not None
    assert qc_hedger is not None
    assert pf_hedger is not None

    # update step.
    for ft in pf.get_all_futures():
        if ft.get_product() == 'CC':
            ft.update_price(50)
        elif ft.get_product() == 'QC':
            ft.update_price(75)

    # ensure that price updates are passed into the relevant hedger's copy.
    for ft in cc_hedger.pf.get_all_futures():
        assert ft.get_price() == 50

    for ft in qc_hedger.pf.get_all_futures():
        assert ft.get_price() == 75

    for op in pf.get_all_options():
        if op.get_product() == 'CC':
            op.update_greeks(vol=0.75)
        elif op.get_product() == 'QC':
            op.update_greeks(vol=0.65)

    for op in cc_hedger.pf.get_all_options():
        assert op.underlying.get_price() == 50
        assert op.vol == 0.75

    for op in qc_hedger.pf.get_all_options():
        assert op.underlying.get_price() == 75
        assert op.vol == 0.65


def test_pf_assign_hedge_dataframes():
    from scripts.util import assign_hedge_objects
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(True)
    pf = copy.deepcopy(pf_comp)
    # get initialization parameters
    date = pdf.value_date.min()
    maxdate = pdf.value_date.max()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # # print('r_vdf: ', r_vdf)
    # # print('r_pdf: ', r_pdf)

    # print('r_pdf columns: ', r_pdf.columns)
    r_vdf.sort_values(by='value_date', inplace=True)
    r_vdf.reset_index(drop=True, inplace=True)

    r_pdf.sort_values(by='value_date', inplace=True)
    r_pdf.reset_index(drop=True, inplace=True)
    # first: assign hedges to all.
    pf = assign_hedge_objects(pf)
    cc_pf = [fam for fam in pf.families if fam.name == 'cc_comp'][0]
    qc_pf = [fam for fam in pf.families if fam.name == 'qc_comp'][0]

    # assert hedger objects associated with each portfolio are not none
    # sanity check baseline inputs.
    cc_hedger, qc_hedger, pf_hedger = cc_pf.get_hedger(
    ), qc_pf.get_hedger(), pf.get_hedger()
    assert cc_hedger is not None
    assert cc_hedger.vdf is None
    assert cc_hedger.pdf is None
    assert qc_hedger is not None
    assert pf_hedger is not None
    assert qc_hedger.vdf is None
    assert qc_hedger.pdf is None
    assert cc_hedger.mappings == {}
    assert qc_hedger.mappings == {}

    pf.assign_hedger_dataframes(r_vdf, r_pdf)

    assert np.array_equal(cc_hedger.vdf, r_vdf)
    # try:
    assert np.array_equal(cc_hedger.pdf, r_pdf)
    # except AssertionError as e:
    #     for col in cc_hedger.pdf.columns:
    #         # print('col: ', col)
    #         # print(np.array_equal(cc_hedger.pdf[col], r_pdf[col]))
    #         pass

    assert np.array_equal(qc_hedger.vdf, r_vdf)
    assert np.array_equal(qc_hedger.pdf, r_pdf)

    # check that the other computations are made as well.
    assert cc_hedger.mappings != {}
    assert qc_hedger.mappings != {}
    # print('cc_mappings: ', cc_hedger.mappings)
    # print('qc_mappings: ', qc_hedger.mappings)

    r_vdf = vdf[vdf.value_date == maxdate]
    r_pdf = pdf[pdf.value_date == maxdate]

    # # print('r_pdf columns: ', r_pdf.columns)
    r_vdf.sort_values(by='value_date', inplace=True)
    r_vdf.reset_index(drop=True, inplace=True)

    r_pdf.sort_values(by='value_date', inplace=True)
    r_pdf.reset_index(drop=True, inplace=True)
    # update dataframe, check that they are equal again.
    pf.assign_hedger_dataframes(r_vdf, r_pdf)
    assert np.array_equal(cc_hedger.vdf, r_vdf)
    assert np.array_equal(cc_hedger.pdf, r_pdf)
    assert np.array_equal(qc_hedger.vdf, r_vdf)
    assert np.array_equal(qc_hedger.pdf, r_pdf)
