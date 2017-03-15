"""
File Name      : test_options.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : File contains tests for Options class methods in classes.py

"""
from scripts.classes import Option, Future

ft = Future('march', 30, 'C')
strike = 30
tau = 30/365


def test_check_active_ko():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                'up', barrier='euro', bullet=False, ko=50)
    assert op.check_active() == True

    op = Option(strike, tau, 'call', vol, ft, payoff,
                'up', barrier='euro', bullet=False, ko=20)
    assert op.check_active() == False

    ft.update_price(70)
    strike = 75
    op = Option(strike, tau, 'call', vol, ft, payoff,
                'down', barrier='euro', bullet=False, ko=50)
    assert op.check_active() == True

    ft.update_price(49)
    assert op.check_active() == False


def test_check_active_ki():
    pass


def test_get_underlying():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                'up', barrier='euro', bullet=False, ko=50)
    assert op.get_underlying() == ft


def test_get_desc():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    assert op.get_desc() == 'option'


def test_init_greeks():
    pass


def test_update_greeks():
    pass


def test_greeks():
    pass


def test_update_tau():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    op.update_tau(0.1)
    assert op.tau == (30/365) - 0.1


def test_get_product():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    assert op.get_product() == 'C'


def test_exercise():
    ft = Future('march', 60, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff)
    assert op.exercise() == True
    op2 = Option(strike, tau, 'put', vol, ft, payoff)
    assert op2.exercise() == False
    ft.update_price(10)
    assert op.get_underlying().get_price() == 10
    assert op.get_price() == op.get_underlying().get_price()
    assert op.exercise() == False
    assert op2.exercise() == True


def test_moneyness():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    initial_val = op.get_price()
    assert op.moneyness == 0
    ft.update_price(50)
    assert op.moneyness == 1
    ft.update_price(20)
    assert op.moneyness == -1


def test_updates_passed():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    initial_val = op.get_price()
    d1, g1, t1, v1 = op.greeks()

    ft.update_price(50)

    curr_val = op.get_price()
    d2, g2, t2, v2 = op.greeks()
    assert [d1, g1, t1, v1] != [d2, g2, t2, v2]
    assert d2 > d1
    assert g2 > g1
    assert t2 < t1
    assert initial_val != curr_val
    assert initial_val < curr_val
