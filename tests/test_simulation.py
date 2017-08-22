"""
File Name      : test_simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 4/4/2017
Python version : 3.5
Description    : File contains tests for methods in simulation.py

"""

# Imports
from scripts.fetch_data import grab_data 
from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
from scripts.simulation import hedge_delta_roll, check_roll_status, \
                               handle_exercise, contract_roll
from scripts.util import create_straddle, combine_portfolios 
from collections import OrderedDict 
import numpy as np
import pandas as pd
import copy 



############## variables ###########
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date, write_dump=True)
####################################


def generate_portfolio(flag):
    """Generate portfolio for testing purposes. """
    # Underlying Futures
    ft1 = Future('K7', 300, 'C')
    ft2 = Future('K7', 250, 'C')
    ft3 = Future('N7', 320, 'C')
    ft4 = Future('N7', 330, 'C')
    ft5 = Future('N7', 240, 'C')

    # options
    short = False if flag == 'long' else True
    # options

    op1 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', short,
        'K7', ordering=2)

    op2 = Option(
        290, 0.301369863013698, 'call', 0.45176132048500206, ft2, 'amer', short,
        'K7', ordering=2)

    op3 = Option(300, 0.473972602739726, 'call', 0.14464169782291536,
                 ft3, 'amer', short, 'N7',  direc='up', barrier='amer', bullet=False,
                 ko=350, ordering=2)

    op4 = Option(330, 0.473972602739726, 'put', 0.18282926924909026,
                 ft4, 'amer', short, 'N7', direc='down', barrier='amer', bullet=False,
                 ki=280, ordering=2)
    op5 = Option(
        320, 0.473972602739726, 'put', 0.8281728247909962, ft5, 'amer', short,
        'N7', ordering=2)

    # Portfolio Futures
    # ft6 = Future('K7', 370, 'C', shorted=False, ordering=1)
    # ft7 = Future('N7', 290, 'C', shorted=False, ordering=2)
    # ft8 = Future('Z7', 320, 'C', shorted=True, ordering=4)
    # ft9 = Future('Z7', 320, 'C', shorted=True, ordering=4)

    OTCs, hedges = [op1, op2, op3], [op4, op5]

    # creating portfolio
    pf = Portfolio(None)
    pf.add_security(OTCs, 'OTC')
    pf.add_security(hedges, 'OTC')
    # for sec in hedges:
    #     pf.add_security(sec, 'OTC')

    # for sec in OTCs:
    #     pf.add_security(sec, 'OTC')

    return pf


def test_generate_portfolio():
    pf1 = generate_portfolio('long')
    for op in pf1.get_all_options():
        assert op.shorted == False
    pf2 = generate_portfolio('short')
    for op in pf2.get_all_options():
        assert op.shorted == True
    net1, net2 = pf1.net_greeks['C']['K7'], pf2.net_greeks['C']['K7']
    gamma1, gamma2 = net1[1], net2[1]
    vega1, vega2 = net1[3], net2[3]
    assert gamma1 == -gamma2
    assert vega1 == -vega2


def test_shorted_greeks():
    ft1 = Future('K7', 300, 'C')
    op1 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', False, 'K7', ordering=1)
    op2 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', True, 'K7', ordering=1)
    g1 = np.array((op1.greeks()))
    g2 = -np.array((op2.greeks()))
    # print(g1, g2)
    assert np.isclose(g1.all(), g2.all())


def test_feed_data_updates():
    pf = generate_portfolio('long')
    init_val = pf.compute_value()
    all_underlying = pf.get_underlying()
    for ft in all_underlying:
        ft.update_price(ft.price + 10)

    new_val = pf.compute_value()
    try:
        assert new_val != init_val
    except AssertionError:
        print('new: ', new_val)
        print('init: ', init_val)

    pf2 = generate_portfolio('long')
    init2 = pf2.compute_value()
    all_ft = pf2.get_all_futures()
    for ft in all_ft:
        ft.update_price(60)

    new2 = pf2.compute_value()
    try:
        assert new2 != init2
    except AssertionError:
        print('new2: ', new2)
        print('init2: ', init2)


def comp_portfolio(refresh=False):
    # creating the options.
    ccops = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                            False, 'atm', greek='theta', greekval=10000)
    qcops = create_straddle('QC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
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

    return pf_simple, pf_comp, ccops, qcops, pfcc, pfqc


def test_handle_exercise_simple():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    # testing exercise for simple portfolio. 
    pf = copy.deepcopy(pf_simple)
    ref_op = pf.OTC_options[0]
    cc_ttm = ref_op.tau  
    
    # check to ensure that tau > tol does not trigger exercise. 
    diff = cc_ttm - 2/365 
    pf.timestep(diff)
    # check to ensure timestep happened.
    assert np.isclose(ref_op.tau, 2/365)

    # first exercise.  
    profit, pf, exercised, tobeadded = handle_exercise(pf)
    assert not exercised 

    # now check that it does handle exercises when within tol. 
    pf.timestep(1/365)
    assert np.isclose(ref_op.tau, 1/365)
    # make a copy of options for reference. 
    init_net = copy.deepcopy(pf.get_net_greeks())
    refops = copy.deepcopy(pf.OTC_options) 

    profit, pf, exercised, tobeadded = handle_exercise(pf)
    print('pf.OTC_options: ', pf.OTC_options)
    # exercise would be TRUE if any option was exercised. 
    assert exercised == any([op.exercise() for op in refops])
    # check that tau gets set to 0 
    assert [op.tau for op in pf.OTC_options] == [0, 0]

    # check that net greeks DO NOT go flat after exercise. 
    try:
        assert pf.get_net_greeks() == init_net
    except AssertionError:
        print('[Failure] test_simulation.test_handle_exercise')
        print('actual: ', pf.get_net_greeks())
        print('expected: ', init_net)

    # removing expired 
    pf.remove_expired() 

    # now check that greeks go flat. 
    assert pf.get_net_greeks() == {}


def test_handle_exercise_composite():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    # testing exercise for COMPOSITE portfolios. 
    pf = copy.deepcopy(pf_comp)
    ref_op_cc = next(iter(pf.OTC['CC']['Z7'][0]))
    ref_op_qc = next(iter(pf.OTC['QC']['Z7'][0]))
    cc_ttm = ref_op_cc.tau  
    qc_ttm = ref_op_qc.tau 
    # check to ensure that tau > tol does not trigger exercise. 
    diff = cc_ttm - 2/365 
    pf.timestep(diff)
    # check to ensure timestep happened.
    assert np.isclose(ref_op_cc.tau, 2/365)

    # first exercise.  
    profit, pf, exercised, tobeadded = handle_exercise(pf)
    assert not exercised 

    # now check that it does handle exercises when within tol. 
    pf.timestep(1/365)
    assert np.isclose(ref_op_cc.tau, 1/365)
    # make a copy of options for reference. 
    init_net = copy.deepcopy(pf.get_net_greeks())
    refops = copy.deepcopy(pf.OTC_options) 

    profit, pf, exercised, tobeadded = handle_exercise(pf)
    # print('pf.OTC_options: ', pf.OTC_options)
    # exercise would be TRUE if any option was exercised. 
    assert exercised == any([op.exercise() for op in refops])
    # check that tau gets set to 0 
    assert [op.tau for op in pf.OTC['CC']['Z7'][0]] == [0, 0]

    # check that net greeks DO NOT go flat after exercise. 
    try:
        assert pf.get_net_greeks() == init_net
    except AssertionError:
        print('##### [Failure] test_simulation.test_handle_exercise #####')
        print('actual: ', pf.get_net_greeks())
        print('expected: ', init_net)
        print('##########################################################')
    # removing expired 
    pf.remove_expired() 

    prerefresh_OTC = pf.OTC 
    prerefresh_net = pf.get_net_greeks() 

    # now check that greeks go flat. 
    assert 'CC' not in pf.OTC 
    assert 'CC' not in pf.get_net_greeks() 

    # check that the family containing CC is no longer valid,
    # i.e. contain check fails.
    try:
        assert pf.get_family_containing(ref_op_cc) is None 
    except AssertionError:
        print('########################################################')
        print('[Failure] test_simulation.test_handle_exercise_composite')
        print('actual: ', pf.get_family_containing(ref_op_cc))
        print('expected: ', 'None')
        print('#########################################################')

    # explicitly check the family to make sure CC ops have been removed. 
    ccfam = [x for x in pf.families if x.name == 'cc_comp'][0]
    assert ccfam.empty() 
    assert 'CC' not in ccfam.OTC 
    assert 'CC' not in ccfam.get_net_greeks()

    # check to make sure that the qc family is intact. 
    qcfam = [x for x in pf.families if x.name == 'qc_comp'][0]
    assert not qcfam.empty() 
    assert 'QC' in qcfam.OTC 
    assert 'QC' in qcfam.get_net_greeks() 
    # timestep back to the original 
    qcfam.timestep(-diff-1/365)

    assert qcfam.get_net_greeks() == pfqc.get_net_greeks() 

    # final check: refresh shouldn't mess with anything.
    pf.refresh() 
    assert pf.OTC == prerefresh_OTC
    assert pf.get_net_greeks() == prerefresh_net



def test_check_roll_status():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    
    # first: check the roll status of a simple portfolio.
    init_cc_price = ccops[0].underlying.get_price()
    init_qc_price = qcops[0].underlying.get_price()

    # should return True, since pf_simple initialized atm. 
    assert check_roll_status(pf_simple)
    # save initial greeks 
    init_greeks = pf_simple.get_net_greeks().copy()
    # update the price to something nuts, forcing deltas to go crazy. 
    for op in ccops:
        op.underlying.update_price(5000)
    pf_simple.refresh() 
    assert pf_simple.get_net_greeks() != init_greeks 
    assert not check_roll_status(pf_simple)

    # reset ccops. 
    for op in ccops:
        op.underlying.update_price(init_cc_price)
    
    pfcc.refresh() 
    pfqc.refresh()
    pf_comp.refresh() 

    # checking roll status of a complex portfolio. 
    assert check_roll_status(pf_comp)

    # update ccops but not qcops.     
    for op in ccops:
        op.underlying.update_price(5000)
    pfcc.refresh() 
    pfqc.refresh()
    pf_comp.refresh()

    assert not check_roll_status(pfcc)
    assert check_roll_status(pfqc)
    assert not check_roll_status(pf_comp)
     

    # reset cc prices, perturb QC prices. 
    for op in ccops:
        op.underlying.update_price(init_cc_price)
    pfcc.refresh() 
    pfqc.refresh()
    pf_comp.refresh()

    assert check_roll_status(pfcc)
    assert check_roll_status(pfqc)
    assert check_roll_status(pf_comp)

    for op in qcops:
        op.underlying.update_price(5000)
    pfcc.refresh() 
    pfqc.refresh()
    pf_comp.refresh()

    assert check_roll_status(pfcc)
    assert not check_roll_status(pfqc)
    assert not check_roll_status(pf_comp)
    # assert not check_roll_status()


def test_delta_roll():
    pass 


def test_hedge_delta_roll_simple():
    pass


def test_hedge_delta_roll_comp():
    pass


def test_roll_over():
    # contract roll works, this just makes sure that composites are correctly specified. 
    pass


def test_contract_roll():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf = pf_simple 
    ref_op = ccops[0]
    date = pd.to_datetime(vdf.value_date.min())
    flag = 'OTC'

    pf, cost, newop, op, iden = contract_roll(pf, ref_op, vdf, pdf, date, flag)

    try:
        assert 'H8' in pf.OTC['CC']
        assert 'H8' in pf.get_net_greeks()['CC']  
    except AssertionError:
        print('OTC: ', pf.OTC)
        print('net: ', pf.get_net_greeks())


    assert newop.get_month() == 'H8'
    assert op == ref_op 
    assert newop.lots == ref_op.lots 
    assert newop.shorted == ref_op.shorted 
