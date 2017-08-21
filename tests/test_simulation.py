"""
File Name      : test_simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 4/4/2017
Python version : 3.5
Description    : File contains tests for methods in simulation.py

"""

# Imports
from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
from scripts.prep_data import read_data, generate_hedges
import scripts.global_vars as gv
from scripts.simulation import hedge, hedge_delta_roll, check_roll_status, gen_hedge_inputs
import numpy as np
import pandas as pd
import time
# import numpy as np

t = time.clock()
vdf, pdf, edf, priceD = read_data(gv.test_vol_data,
                                  gv.test_price_data,
                                  gv.test_exp_data,
                                  gv.test_start_date, test=True)
elapsed = time.clock() - t
print('[test_simulation] data read-in elapsed: ', elapsed)


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
    pf = Portfolio()
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


def test_gen_inputs():
    hedges = {'vega': [['bound', (-10, 10), 1]],
              'delta': [['static', 'zero', 3]],
              'gamma': [['bound', (-10, 10), 1]],
              'theta': [['bound', (-10, 10), 1]]}

    # hedges = {'delta': 'zero', 'gamma': (-10, 10), 'vega': (-10, 10)}
    pf = generate_portfolio('long')
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    ordering = pf.compute_ordering(product, month)
    ginputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'gamma')

    # basic tests
    assert len(ginputs) == 9
    price, k, cvol, pvol, tau, underlying, greek, bound, order = ginputs
    assert bound == hedges['gamma'][0][1]
    assert greek == pf.get_net_greeks()['C']['K7'][1]

    vinputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'vega')

    price, k, cvol, pvol, tau, underlying, greek, bound, order = vinputs
    assert greek == pf.get_net_greeks()['C']['K7'][3]


def test_hedge_gamma_long():
    hedges = {'gamma': [['bound', (-3000, 3000), 1]],
              'vega':  [['bound', (-3000, 3000), 1]],
              'delta': [['static', 'zero', 1]],
              'theta': [['bound', (-1000, 1000), 1]]}

    pf = generate_portfolio('long')
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    ordering = pf.compute_ordering(product, month)
    inputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'gamma')
    greek = inputs[6]
    # print('#########################')
    # print('init long gamma: ', greek)

    # gamma hedging from above.
    net = pf.get_net_greeks()
    init_gamma = net['C']['K7'][1]
    # assert init_gamma not in range(*hedges['gamma'][1])
    assert init_gamma == greek
    pf, cost = hedge(pf, inputs, product, month, 'gamma')
    # print(inputs, product, month)
    # print('gamma long hedging expenditure: ', expenditure)
    end_gamma = pf.net_greeks['C']['K7'][1]
    # print('end long gamma: ', end_gamma)
    # print('#########################')
    try:
        assert end_gamma < 10 and end_gamma > -10

    except AssertionError:
        print('gamma long hedging failed: ', end_gamma, hedges['gamma'])
        print('#########################')


def test_hedge_gamma_short():
    # gamma hedging from below
    hedges = {'gamma': [['bound', (-3000, 3000), 1]],
              'vega':  [['bound', (-3000, 3000), 1]],
              'delta': [['static', 'zero', 1]],
              'theta': [['bound', (-1000, 1000), 1]]}
    pf = generate_portfolio('short')
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    ordering = pf.compute_ordering(product, month)
    inputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'gamma')
    greek = inputs[6]

    # print('init short gamma: ', greek)

    net = pf.get_net_greeks()
    init_gamma = net['C']['K7'][1]
    # assert init_gamma not in range(*hedges['gamma'][1])
    assert init_gamma == greek
    pf2, cost = hedge(pf, inputs, product, month, 'gamma')
    # print('gamma short hedging expenditure: ', expenditure)
    end_gamma = pf2.net_greeks['C']['K7'][1]
    # print('end short gamma: ', end_gamma)
    # print('#########################')
    try:
        assert end_gamma < 10 and end_gamma > -10

    except AssertionError:
        print('gamma short hedging failed: ', end_gamma, hedges['gamma'])
        print('#########################')


def test_hedge_vega_short():
    hedges = {'gamma': [['bound', (-3000, 3000), 1]],
              'vega':  [['bound', (-3000, 3000), 1]],
              'delta': [['static', 'zero', 1]],
              'theta': [['bound', (-1000, 1000), 1]]}
    pf = generate_portfolio('short')
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    ordering = pf.compute_ordering(product, month)
    inputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'vega')
    # print('short vega input: ', inputs)
    # print('short vega underlying: ', str(inputs[5]))
    greek = inputs[6]
    # print('init short vega: ', greek)

    net = pf.get_net_greeks()
    init_vega = net['C']['K7'][3]
    # assert init_vega not in range(*hedges['vega'][1])
    assert init_vega == greek
    pf2, cost = hedge(pf, inputs, product, month, 'vega')
    # print('vega hedging expenditure: ', expenditure)
    end_vega = pf2.net_greeks['C']['K7'][3]
    # print('end short vega: ', end_vega)
    # print('#########################')
    try:
        assert end_vega < 300 and end_vega > -300

    except AssertionError:
        print('vega short hedging failed: ', end_vega, hedges['vega'])
        print('#########################')


def test_hedge_vega_long():
    hedges = {'gamma': [['bound', (-3000, 3000), 1]],
              'vega':  [['bound', (-3000, 3000), 1]],
              'delta': [['static', 'zero', 1]],
              'theta': [['bound', (-1000, 1000), 1]]}
    pf = generate_portfolio('long')
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    ordering = pf.compute_ordering(product, month)
    inputs = gen_hedge_inputs(
        hedges, vdf1, pdf1, month, pf, product, ordering, 'vega')
    # print('long vega inputs: ', inputs)
    greek = inputs[6]
    # print('init long vega: ', greek)

    net = pf.get_net_greeks()
    init_vega = net['C']['K7'][3]
    # assert init_vega not in range(*hedges['vega'][1])
    assert init_vega == greek
    pf2, cost = hedge(pf, inputs, product, month, 'vega')
    # print('vega hedging expenditure: ', expenditure)
    end_vega = pf2.net_greeks['C']['K7'][3]

    try:
        assert end_vega < 300 and end_vega > -300
        # print('end long vega: ', end_vega)
        # print('#########################')
    except AssertionError:
        print('vega long hedging failed: ', end_vega, hedges['vega'])
        print('#########################')


def test_delta_hedging_long():
    pf = generate_portfolio('long')
    cond = 'zero'
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    assert len(pf.hedge_futures) == 0
    # print('init_pf_long', pf)
    ordering = pf.compute_ordering(product, month)
    pf, hedge, fees = hedge_delta(
        cond, vdf1, pdf1, pf, month, product, ordering)
    # print('hedge_long: ', hedge)
    # print('end_pf_long: ', pf)
    assert hedge.shorted == True
    assert len(pf.hedge_futures) == 1


def test_delta_hedging_short():
    pf = generate_portfolio('short')
    # print(pf.net_greeks)
    cond = 'zero'
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    product = 'C'
    month = 'K7'
    assert len(pf.hedge_futures) == 0
    # print('init_pf_short: ', pf)
    ordering = pf.compute_ordering(product, month)
    pf, hedge, fees = hedge_delta(
        cond, vdf1, pdf1, pf, month, product, ordering)
    # print('hedge_short: ', hedge)
    # print('end_pf_short: ', pf)
    try:
        assert len(pf.hedge_futures) == 1
    except AssertionError:
        print('num_futures: ', len(pf.hedge_futures))
    try:
        assert hedge.shorted == False
    except AssertionError:
        print('hedge: ', hedge)
        print('hedge type: ', hedge.shorted)


def test_check_roll_status():
    pass


def test_hedge_delta_roll():
    pass


def test_roll_over():
    pass


def test_contract_roll():
    pass
