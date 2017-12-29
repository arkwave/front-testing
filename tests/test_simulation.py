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
    handle_exercise, contract_roll, roll_over, delta_roll, hedge_delta_roll
from scripts.util import create_straddle, combine_portfolios, create_vanilla_option, create_composites, create_underlying
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
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=False)
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
    # print('========== TEST: GENERATE PORTFOLIO ===========')
    pf1 = generate_portfolio('long')
    for op in pf1.get_all_options():
        assert not op.shorted
    pf2 = generate_portfolio('short')
    for op in pf2.get_all_options():
        assert op.shorted
    net1, net2 = pf1.net_greeks['C']['K7'], pf2.net_greeks['C']['K7']
    gamma1, gamma2 = net1[1], net2[1]
    vega1, vega2 = net1[3], net2[3]
    assert gamma1 == -gamma2
    assert vega1 == -vega2
    # print('================================================')


def test_shorted_greeks():
    # print('========== TEST: SHORTED GREEKS ===========')
    ft1 = Future('K7', 300, 'C')
    op1 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', False, 'K7', ordering=1)
    op2 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', True, 'K7', ordering=1)
    g1 = np.array((op1.greeks()))
    g2 = -np.array((op2.greeks()))
    # # print(g1, g2)
    assert np.isclose(g1.all(), g2.all())
    # print('================================================')


def test_feed_data_updates():
    # print('========== TEST: FEED DATA UPDATES ===========')
    pf = generate_portfolio('long')
    init_val = pf.compute_value()
    all_underlying = pf.get_underlying()
    for ft in all_underlying:
        ft.update_price(ft.price + 10)

    new_val = pf.compute_value()
    # try:
    assert new_val != init_val
    # except AssertionError:
    #     # print('new: ', new_val)
    #     # print('init: ', init_val)

    pf2 = generate_portfolio('long')
    init2 = pf2.compute_value()
    all_ft = pf2.get_all_futures()
    for ft in all_ft:
        ft.update_price(60)

    new2 = pf2.compute_value()
    # try:
    assert new2 != init2
    # except AssertionError:
    #     # print('new2: ', new2)
    #     # print('init2: ', init2)
    # print('================================================')


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
                   'theta': [['bound', (9000, 11000), 1, 'straddle',
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
    # print('========== TEST: HANDLE EXERCISE SIMPLE ===========')
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
    # print('pf.OTC_options: ', pf.OTC_options)
    # exercise would be TRUE if any option was exercised.
    assert exercised == any([op.exercise() for op in refops])
    # check that tau gets set to 0
    assert [op.tau for op in pf.OTC_options] == [0, 0]

    # check that net greeks DO NOT go flat after exercise.
    # try:
    assert pf.get_net_greeks() == init_net
    # except AssertionError:
    #     # print('[Failure] test_simulation.test_handle_exercise')
    #     # print('actual: ', pf.get_net_greeks())
    #     # print('expected: ', init_net)

    # removing expired
    pf.remove_expired()

    # now check that greeks go flat.
    assert pf.get_net_greeks() == {}
    # print('================================================')


def test_handle_exercise_composite():
    # print('========== TEST: HANDLE EXERCISE COMPOSITE ===========')
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
    # # print('pf.OTC_options: ', pf.OTC_options)
    # exercise would be TRUE if any option was exercised.
    assert exercised == any([op.exercise() for op in refops])
    # check that tau gets set to 0
    assert [op.tau for op in pf.OTC['CC']['Z7'][0]] == [0, 0]

    # check that net greeks DO NOT go flat after exercise.
    # try:
    assert pf.get_net_greeks() == init_net
    # except AssertionError:
    #     # print('##### [Failure] test_simulation.test_handle_exercise #####')
    #     # print('actual: ', pf.get_net_greeks())
    #     # print('expected: ', init_net)
    #     # print('##########################################################')
    # removing expired
    pf.remove_expired()

    prerefresh_OTC = pf.OTC
    prerefresh_net = pf.get_net_greeks()

    # now check that greeks go flat.
    assert 'CC' not in pf.OTC
    assert 'CC' not in pf.get_net_greeks()

    # check that the family containing CC is no longer valid,
    # i.e. contain check fails.
    # try:
    assert pf.get_family_containing(ref_op_cc) is None
    # except AssertionError:
    #     # print('########################################################')
    #     # print('[Failure] test_simulation.test_handle_exercise_composite')
    #     # print('actual: ', pf.get_family_containing(ref_op_cc))
    #     # print('expected: ', 'None')
    #     # print('#########################################################')

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

    # final check: refresh shouldn't mess with anything. < doesn't make sense,
    # since ccfam is now empty while QC fam is not.
    # pf.refresh()
    # assert pf.OTC == prerefresh_OTC
    # try:
    #     assert pf.get_net_greeks() == prerefresh_net
    # except AssertionError as e:
    #     raise AssertionError(
    #         'old, new: ', pf.get_net_greeks(), prerefresh_net) from e
    # print('================================================')


def test_check_roll_status():
    # print('========== TEST: CHECK ROLL STATUS ===========')
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
    # print('================================================')


def test_contract_roll():
    # print('========== TEST: CONTRACT ROLL BASE ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf = pf_simple
    ref_op = ccops[0]
    date = pd.to_datetime(vdf.value_date.min())
    flag = 'OTC'

    pf, cost, newop, op, iden = contract_roll(pf, ref_op, vdf, pdf, date, flag)

    # try:
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.get_net_greeks()['CC']
    # except AssertionError:
    #     # print('OTC: ', pf.OTC)
    #     # print('net: ', pf.get_net_greeks())

    assert newop.get_month() == 'H8'
    assert op == ref_op
    assert newop.lots == ref_op.lots
    assert newop.shorted == ref_op.shorted
    # print('================================================')


def test_roll_over_no_product_simple():
    # print('========== TEST: CONTRACT ROLL NO PRODUCT SIMPLE ===========')
    # contract roll works, this just makes sure that composites are correctly
    # specified.
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365

    # simple portfolio: checking composites in the same family.
    pf = copy.deepcopy(pf_simple)
    pf.ttm_tol = ttm_tol + 2
    pf.roll = True
    # Z7 -> H8
    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # print('pf: ', pf)

    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.get_net_greeks()['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.get_net_greeks()['CC']

    op1 = pf.OTC_options[0]
    op2 = pf.OTC_options[1]
    # checking for composites.
    assert op1 in op2.partners
    assert op2 in op1.partners
    # print('================================================')


def test_rollover_no_product_simple_split():
    # print('========== TEST: CONTRACT ROLL NO PRODUCT SPLIT ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365
    # simple portfolio: checking composites in different families

    pf1 = Portfolio(None, name='pf1')

    pf2 = Portfolio(None, name='pf2')
    pf1.add_security([ccops[0]], 'OTC')
    pf1.ttm_tol = ttm_tol + 1
    pf1.roll = True
    # print('pf1: ', pf1)
    pf2.add_security([ccops[1]], 'OTC')
    pf = combine_portfolios([pf1, pf2], name='pf', refresh=True)
    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # print('pf: ', pf)

    # try:
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.get_net_greeks()['CC']
    assert 'H8' in pf1.OTC['CC']
    assert 'H8' in pf1.get_net_greeks()['CC']
    assert 'H8' in pf2.OTC['CC']
    assert 'H8' in pf2.get_net_greeks()['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.get_net_greeks()['CC']
    assert 'Z7' not in pf1.OTC['CC']
    assert 'Z7' not in pf1.get_net_greeks()['CC']
    assert 'Z7' not in pf2.OTC['CC']
    assert 'Z7' not in pf2.get_net_greeks()['CC']

    op1 = pf.OTC_options[0]
    op2 = pf.OTC_options[1]

    # checking for composites.
    assert op1 in op2.partners
    assert op2 in op1.partners
    # print('================================================')


def test_rollover_no_product_comp():
    # print('========== TEST: CONTRACT ROLL NO PRODUCT COMP ===========')
    # comp portfolio: checking composites in same families
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365

    pf = copy.deepcopy(pf_comp)
    pf1 = [x for x in pf.families if x.name == 'cc_comp'][0]
    pf1.ttm_tol = ttm_tol + 2
    pf1.roll = True
    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.get_net_greeks()['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.get_net_greeks()['CC']

    pf1 = [x for x in pf.families if x.name == 'cc_comp'][0]

    # try:
    assert 'H8' in pf1.OTC['CC']
    assert 'H8' in pf1.get_net_greeks()['CC']
    assert 'Z7' not in pf1.OTC['CC']
    assert 'Z7' not in pf1.get_net_greeks()['CC']
    # except AssertionError:
    #     # print('pfcc.OTC: ', pf1.OTC['CC'])
    #     # print('pfcc.net: ', pf1.get_net_greeks()['CC'])

    oplst = list(pf.OTC['CC']['H8'][0])

    op1 = oplst[0]
    op2 = oplst[1]
    # checking for composites.
    assert op1 in op2.partners
    assert op2 in op1.partners

    op3 = pf1.OTC_options[0]
    op4 = pf1.OTC_options[1]
    # checking for composites.
    assert op3 in op4.partners
    assert op4 in op3.partners
    # print('================================================')


def test_rollover_no_product_split():
    # print('========== TEST: CONTRACT ROLL NO PRODUCT SPLIT FAMILY ===========')
    # comp portfolio: checking composites in different families
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365
    pf1 = Portfolio(None, name='pf1')
    pf2 = Portfolio(None, name='pf2')

    # setting roll = true and rolling params
    pf1.roll = True
    pf1.ttm_tol = ttm_tol + 2
    pf2.roll = True
    pf2.ttm_tol = ttm_tol + 2

    pf1.add_security([ccops[0], qcops[0]], 'OTC')
    pf2.add_security([ccops[1], qcops[1]], 'OTC')

    pf = combine_portfolios([pf1, pf2], name='pf', refresh=True)
    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # check that greeks are appropriately updated in overall pf
    assert 'H8' in pf.OTC['CC']
    assert 'H8' in pf.get_net_greeks()['CC']
    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.get_net_greeks()['CC']
    # check that greeks are appropriately updated in pf1
    assert 'H8' in pf1.OTC['CC']
    assert 'H8' in pf1.get_net_greeks()['CC']
    assert 'Z7' not in pf1.OTC['CC']
    assert 'Z7' not in pf1.get_net_greeks()['CC']
    # check that greeks are appropriately updated in pf2
    assert 'H8' in pf2.OTC['CC']
    assert 'H8' in pf2.get_net_greeks()['CC']
    assert 'Z7' not in pf2.OTC['CC']
    assert 'Z7' not in pf2.get_net_greeks()['CC']

    # checking composites are maintained
    op1 = [x for x in pf1.OTC_options if x.get_product() == 'CC'][0]
    op2 = [x for x in pf2.OTC_options if x.get_product() == 'CC'][0]
    op3 = [x for x in pf1.OTC_options if x.get_product() == 'QC'][0]
    op4 = [x for x in pf2.OTC_options if x.get_product() == 'QC'][0]

    # check composites preserved
    assert op1 in op2.partners
    assert op2 in op1.partners

    # check that unaffected options are unaffected.
    assert op3 in op4.partners
    assert op4 in op3.partners
    # print('================================================')


def test_roll_over_product():
    # print('========== TEST: CONTRACT ROLL PRODUCT ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    # simple case
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365

    pf = copy.deepcopy(pf_simple)
    pf.ttm_tol = ttm_tol + 1
    pf.roll = True

    pf_simple, cost, _ = roll_over(pf, r_vdf, r_pdf,
                                   date)

    assert 'Z7' not in pf.OTC['CC']
    assert 'Z7' not in pf.get_net_greeks()['CC']

    # check composites
    op1, op2 = [x for x in pf_simple.OTC_options if x.get_product() == 'CC']
    # op3, op4 = [x for x in pf_simple.OTC_options if x.get_product() == 'QC']

    assert op1 in op2.partners
    assert op2 in op1.partners

    # comp case
    pf2 = copy.deepcopy(pf_comp)
    ops = pf2.get_all_options()
    ops = create_composites(ops)

    cc_pf = [fam for fam in pf2.get_families()
             if fam.get_unique_products() == set(['CC'])][0]
    qc_pf = [fam for fam in pf2.get_families()
             if fam.get_unique_products() == set(['QC'])][0]

    cc_pf.roll = True
    cc_pf.ttm_tol = ttm_tol + 1

    pf2, cost, _ = roll_over(pf2, r_vdf, r_pdf,
                             date)

    assert 'Z7' not in pf2.OTC['CC']
    assert 'Z7' not in pf2.OTC['QC']
    assert 'Z7' not in pf2.get_net_greeks()['CC']
    assert 'Z7' not in pf2.get_net_greeks()['QC']

    assert 'Z7' not in cc_pf.OTC['CC']
    assert 'Z7' not in cc_pf.get_net_greeks()['CC']

    assert 'Z7' not in qc_pf.OTC['QC']
    assert 'Z7' not in qc_pf.get_net_greeks()['QC']

    op1, op2 = [x for x in pf2.OTC_options if x.get_product() == 'CC']
    op3, op4 = [x for x in pf2.OTC_options if x.get_product() == 'QC']
    assert op1 in op2.partners
    assert op2 in op1.partners
    assert op3 in op4.partners
    assert op4 in op3.partners
    # print('================================================')


def test_delta_roll():
    # print('========== TEST: DELTA ROLL BASE ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    op = pf_simple.OTC_options[0]
    newop, op, cost = delta_roll(pf_simple, op, 25, r_vdf, r_pdf, 'OTC')
    newop_delta = abs(newop.delta/newop.lots)
    assert abs(newop_delta - 0.25) < 0.02
    # print('================================================')


def test_hedge_delta_roll_simple():
    # print('========== TEST: DELTA ROLL SIMPLE ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    # resetting the hedge params
    cc_hedges_s = {'delta': [['static', 'zero', 1],
                             ['roll', 25, 1, (-10, 10)]]}
    pf_simple.hedge_params = cc_hedges_s
    pf_simple.refresh()
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    op = pf_simple.OTC_options[0]
    pf, cost = hedge_delta_roll(pf_simple, r_vdf, r_pdf)
    for op in pf.OTC_options:
        newop_delta = abs(op.delta/op.lots)
        assert abs(newop_delta - 0.25) < 0.02
    op1, op2 = pf.OTC_options
    assert op1 in op2.partners
    assert op2 in op1.partners

    # slightly more interesting case: one is within bounds, one is not.
    ccop1 = create_vanilla_option(
        r_vdf, r_pdf, 'CC  Z7.Z7', 'call', False, delta=40)
    ccop2 = create_vanilla_option(
        r_vdf, r_pdf, 'CC  Z7.Z7', 'put', True, delta=30)

    cops = create_composites([ccop1, ccop2])

    pf = Portfolio(cc_hedges_s)

    pf.add_security(cops, 'OTC')

    pf, cost = hedge_delta_roll(pf, r_vdf, r_pdf)

    for op in pf.OTC_options:
        newop_delta = abs(op.delta/op.lots)
        assert abs(newop_delta - 0.25) < 0.02

    op1, op2 = pf.OTC_options
    assert op1 in op2.partners
    assert op2 in op1.partners
    # print('================================================')


def test_hedge_delta_roll_comp():
    # print('========== TEST: DELTA ROLL COMPOSITE ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    cc_hedges_c = {'delta': [['roll', 25, 1, (-10, 10)]],
                   'theta': [['bound', (9000, 11000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    qc_hedges = {'delta': [['roll', 50, 1, (-15, 15)]],
                 'theta': [['bound', (9000, 11000), 1, 'straddle',
                            'strike', 'atm', 'uid']]}

    # reassign hedge params
    pfcc.hedge_params = cc_hedges_c
    pfqc.hedge_params = qc_hedges

    pfcc.refresh()
    pfqc.refresh()
    pf_comp.refresh()

    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    pf, cost = hedge_delta_roll(pf_comp, r_vdf, r_pdf)

    ccops1 = [x for x in pf.OTC_options if x.get_product() == 'CC']
    ccops2 = pfcc.OTC_options
    for op in ccops1 + list(ccops2):
        delta = abs(op.delta/op.lots)
        assert abs(delta - 0.25) < 0.02
    op1, op2 = ccops1

    assert op1 in op2.partners
    assert op2 in op1.partners

    # qc options are not rolled since they have different roll conditions
    for op in pfqc.OTC_options:
        delta = abs(op.delta/op.lots)
        assert abs(delta - 0.5) < 0.1
    # print('================================================')


def test_roll_over_hedge_options():
    # print('========== TEST: ROLL OVER HEDGES ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    pf = copy.deepcopy(pf_comp)
    create_composites(pf.get_all_options())
    cc_pf = [fam for fam in pf.get_families() if
             fam.get_unique_products() == set(['CC'])][0]
    ccops2 = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                             True, 'atm', greek='theta', greekval=5000)
    cc_pf.add_security(ccops2, 'hedge')

    pf.refresh()

    assert len(pf.get_all_options()) == 6
    # roll over, and check that hedge options are rolled as well.
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365

    # create composites to roll together.

    # print('cc_pf: ', cc_pf)

    cc_pf.roll = True
    cc_pf.ttm_tol = ttm_tol + 1

    # print('pf: ', pf)

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)
    pf.refresh()

    assert len(pf.OTC_options) == 4
    assert len(pf.get_all_options()) == 6
    assert len(pf.hedge_options) == 2

    try:
        assert 'Z7' not in pf.net_greeks['CC']
        assert 'Z7' not in pf.net_greeks['QC']
        assert 'Z7' not in pf.OTC['CC']
        assert 'Z7' not in pf.OTC['QC']
        assert 'Z7' not in pf.hedges['CC']
    except (KeyError) as e:
        raise AssertionError('DEBUG: ', [str(x)
                                         for x in pf.get_all_options()]) from e

    # print('================== test end ==========================')


def test_roll_composite_partners():
    # print('========== TEST: ROLL COMPOSITE PARTNERS ===========')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (9000, 11000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}
    qc_hedges = {'delta': [['roll', 50, 1, (-15, 15)]],
                 'theta': [['bound', (9000, 11000), 1, 'straddle',
                            'strike', 'atm', 'uid']]}
    gen_hedges = OrderedDict({'delta': [['static', 'zero', 1]]})

    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]
    ttm_tol = ccops[0].tau * 365

    cops2 = copy.deepcopy(ccops)
    qops2 = copy.deepcopy(qcops)

    ops = create_composites(cops2 + qops2)

    pf_c = Portfolio(cc_hedges_c, roll=True, ttm_tol=ttm_tol+1)
    pf_c.add_security(cops2, 'OTC')

    pf_q = Portfolio(qc_hedges)
    pf_q.add_security(qops2, 'OTC')

    pf = combine_portfolios([pf_q, pf_c], hedges=gen_hedges)

    pf.name = 'all'
    pf_c.name = 'cc_pf'
    pf_q.name = 'qc_pf'

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # print('pf after roll: ', pf)
    # print('pf_c after roll: ', pf_c)
    # print('pf_q after roll: ', pf_q)

    assert 'Z7' not in pf_c.net_greeks['CC']
    assert 'Z7' not in pf_q.net_greeks['QC']
    assert 'Z7' not in pf_c.OTC['CC']
    assert 'Z7' not in pf_q.OTC['QC']

    assert 'H8' in pf_c.net_greeks['CC']
    assert 'H8' in pf_q.net_greeks['QC']
    assert 'H8' in pf_c.OTC['CC']
    assert 'H8' in pf_q.OTC['QC']
    # print('================== test end ==========================')


def test_roll_product_hedges():
    # Case: QC Z7.Z7 hedged with QC Z7.Z7, and CC Z7.Z7 hedged with CC. CC and QC
    # OTC options are partners. Rolling CC OTC should trigger QC OTC and QC/CC
    # Hedge to roll.
    # print('================ TEST: ROLL OVER WITH PRODUCT & HEDGES ================')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    pf = copy.deepcopy(pf_comp)
    _ = create_composites(pf.get_all_options())
    pfcc1 = [fam for fam in pf.get_families() if fam.name == 'cc_comp'][0]
    pfqc1 = [fam for fam in pf.get_families() if fam.name == 'qc_comp'][0]

    for op in pfcc1.get_all_options():
        assert len(op.partners) == 3

    for op in pfqc1.get_all_options():
        assert len(op.partners) == 3

    # creating and adding hedges.
    ccops2 = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                             True, 'atm', greek='theta', greekval=5000)
    qcops2 = create_straddle('QC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                             True, 'atm', greek='theta', greekval=5000)

    pfcc1.add_security(ccops2, 'hedge')
    pfqc1.add_security(qcops2, 'hedge')

    # setting roll params for cc portfolio.
    pfcc1.roll = True
    pfcc1.ttm_tol = (ccops[0].tau * 365) + 1
    pf.refresh()

    # roll inputs
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    # print('pf: ', pf)
    # print('pfcc1: ', pfcc1)
    # print('pfqc1: ', pfqc1)

    assert 'Z7' not in pfcc1.net_greeks['CC']
    assert 'Z7' not in pfcc1.OTC['CC']
    assert 'Z7' not in pfcc1.hedges['CC']

    assert 'Z7' not in pfqc1.net_greeks['QC']
    assert 'Z7' not in pfqc1.OTC['QC']
    assert 'Z7' not in pfqc1.hedges['QC']

    assert 'H8' in pfcc1.net_greeks['CC']
    assert 'H8' in pfqc1.net_greeks['QC']
    assert 'H8' in pfcc1.OTC['CC']
    assert 'H8' in pfqc1.OTC['QC']
    assert 'H8' in pfcc1.hedges['CC']
    assert 'H8' in pfqc1.hedges['QC']


def test_roll_product_hedges2():
    # Case: QC Z7.Z7 hedged with QC Z7.Z7, and CC Z7.Z7 unhedged. CC and QC
    # OTC options are partners. Rolling CC should trigger both QC OTC and QC
    # Hedge to roll.
    # print('================ TEST: ROLL OVER WITH PRODUCT & HEDGES - 2 ================')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    pf = copy.deepcopy(pf_comp)
    create_composites(pf.get_all_options())
    pfcc1 = [fam for fam in pf.get_families() if fam.name == 'cc_comp'][0]
    pfqc1 = [fam for fam in pf.get_families() if fam.name == 'qc_comp'][0]

    for op in pfcc1.get_all_options():
        assert len(op.partners) == 3

    for op in pfqc1.get_all_options():
        assert len(op.partners) == 3

    # add QC hedges.
    qcops2 = create_straddle('QC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                             True, 'atm', greek='theta', greekval=5000)

    pfqc1.add_security(qcops2, 'hedge')

    # setting roll params for cc portfolio.
    pfcc1.roll = True
    pfcc1.ttm_tol = (ccops[0].tau * 365) + 1
    pf.refresh()

    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    assert 'Z7' not in pfcc1.net_greeks['CC']
    assert 'Z7' not in pfcc1.OTC['CC']

    assert 'Z7' not in pfqc1.net_greeks['QC']
    assert 'Z7' not in pfqc1.OTC['QC']
    assert 'Z7' not in pfqc1.hedges['QC']

    assert 'H8' in pfcc1.net_greeks['CC']
    assert 'H8' in pfqc1.net_greeks['QC']
    assert 'H8' in pfcc1.OTC['CC']
    assert 'H8' in pfqc1.OTC['QC']
    assert 'H8' in pfqc1.hedges['QC']


def test_roll_over_full():
    # print('================= TEST: CLOSING OUT DELTAS =================')
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # modifying portfolio
    pf = copy.deepcopy(pf_comp)
    create_composites(pf.get_all_options())
    cc_pf = [fam for fam in pf.get_families() if
             fam.get_unique_products() == set(['CC'])][0]

    ccops2 = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                             True, 'atm', greek='theta', greekval=5000)
    cc_pf.add_security(ccops2, 'hedge')
    cc_pf.roll = True
    cc_pf.ttm_tol = (ccops[0].tau * 365) + 1
    pf.refresh()

    # initializing the futures to zero out deltas.
    cc_delta = int(pf.net_greeks['CC']['Z7'][0])
    cc_shorted = False if cc_delta < 0 else True
    qc_delta = int(pf.net_greeks['QC']['Z7'][0])
    qc_shorted = False if qc_delta < 0 else True

    ccft, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=cc_delta, shorted=cc_shorted)
    qcft, _ = create_underlying(
        'QC', 'Z7', r_pdf, date, lots=qc_delta, shorted=qc_shorted)

    pf.add_security([ccft, qcft], 'hedge')
    pf.refresh()
    pf, cost, _ = roll_over(pf, r_vdf, r_pdf, date)

    pf.refresh()
    # print('pf: ', pf)

    # QC gets closed out altogether.
    assert 'QC' not in pf.hedges

    cc_hedges = pf.hedges['CC']
    assert 'Z7' not in cc_hedges

    # assert len(cc_hedges['H8'][1]) == 1
    # assert len(qc_hedges['H8'][1]) == 1

    # print('===============================================================')
