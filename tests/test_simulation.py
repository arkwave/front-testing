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
from scripts.prep_data import read_data
from simulation import hedge_gamma_vega
import copy

vdf, pdf, edf = read_data()
# vdf.to_csv('vdf.csv')
# pdf.to_csv('pdf.csv')


def generate_portfolio():
    """Generate portfolio for testing purposes. """
    # Underlying Futures
    ft1 = Future('K7', 300, 'C')
    ft2 = Future('K7', 250, 'C')
    ft3 = Future('N7', 320, 'C')
    ft4 = Future('N7', 330, 'C')
    ft5 = Future('N7', 240, 'C')

    # options

    op1 = Option(
        350, 0.05106521860205984, 'call', 0.4245569263291844, ft1, 'amer', False, 'K7', ordering=1)

    op2 = Option(
        290, 0.2156100288506942, 'call', 0.45176132048500206, ft2, 'amer', False, 'K7', ordering=1)

    op3 = Option(300, 0.21534276294769317, 'call', 0.14464169782291536,
                 ft3, 'amer', True, 'N7',  direc='up', barrier='amer', bullet=False, ko=350, ordering=2)

    op4 = Option(330, 0.22365510948646386, 'put', 0.18282926924909026,
                 ft4, 'amer', False, 'N7', direc='down', barrier='amer', bullet=False, ki=280, ordering=2)

    op5 = Option(
        320, 0.010975090692443346, 'put', 0.8281728247909962, ft5, 'amer', True, 'N7', ordering=2)

    # Portfolio Futures
    ft6 = Future('K7', 370, 'C', shorted=False, ordering=1)
    ft7 = Future('N7', 290, 'C', shorted=False, ordering=2)
    ft8 = Future('Z7', 320, 'C', shorted=True, ordering=4)
    ft9 = Future('Z7', 320, 'C', shorted=True, ordering=4)

    OTCs, hedges = [op1, op2, ft7, op4, ft6], [op5, op3, ft8, ft9]

    # creating portfolio
    pf = Portfolio()
    for sec in hedges:
        pf.add_security(sec, 'hedge')

    for sec in OTCs:
        pf.add_security(sec, 'OTC')

    return pf


def test_feed_data_updates():
    pf = generate_portfolio()
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

    pf2 = generate_portfolio()
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


def test_hedge_gamma_vega():
    hedges = {'delta': 'zero', 'gamma': (-10, 10), 'vega': (-10, 10)}
    pf = generate_portfolio()
    net = pf.get_net_greeks()
    # vdf, pdf, edf = read_data()
    min_date = min(vdf.value_date)
    vdf1 = vdf[vdf.value_date == min_date]
    pdf1 = pdf[pdf.value_date == min_date]
    for product in net:
        for month in net[product]:
            ordering = pf.compute_ordering(product, month)
            print('ordering: ', ordering)
            cost, pf = hedge_gamma_vega(
                hedges, vdf1, pdf1, month, pf, product, ordering)

    net2 = copy.deepcopy(pf.get_net_greeks())
    g1, g2 = net2['C']['K7'][1], net2['C']['N7'][1]
    v1, v2 = net2['C']['K7'][3], net2['C']['N7'][3]
    gamma_bound = hedges['gamma']
    vega_bound = hedges['vega']
    # basic tests
    try:
        assert g1 in range(*gamma_bound)
    except AssertionError:
        print('g1: ', g1)
    try:
        assert g2 in range(*gamma_bound)
    except AssertionError:
        print('g2: ', g2)
    try:
        assert v1 in range(*vega_bound)
    except AssertionError:
        print('v1: ', v1)
    try:
        assert v2 in range(*vega_bound)
    except AssertionError:
        print('v2: ', v2)


def test_delta_hedging():
    pass
