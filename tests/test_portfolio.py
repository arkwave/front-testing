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
    for sec in longs:
        pf.add_security(sec, 'long')
    for sec in shorts:
        pf.add_security(sec, 'short')

    return pf


def test_basic_functionality():
    pf = generate_portfolio()
    long_pos = pf.get_securities_monthly('long')
    short_pos = pf.get_securities_monthly('short')
    net = pf.net_greeks()
    assert len(long_pos) == 3
    assert len(short_pos) == 3
    assert len(pf.long_options) == 3
    assert len(pf.short_options) == 2
    assert len(pf.long_futures) == 2
    assert len(pf.short_futures) == 2
    assert len(net) == 3


def test_compute_net_greeks():
    pass


def test_add_security():
    pass


def test_remove_security():
    pass


def test_remove_expired():
    pass


def test_update_sec_by_month():
    pass


def test_update_greeks_by_month():
    pass


def test_compute_value():
    pass


def test_exercise_option():
    pass


def test_timestep():
    pass
