"""
File Name      : test_calc.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 21/3/2017
Python version : 3.5
Description    : File contains tests for the methods in calc.py

"""
from scripts.calc import _compute_iv
from scripts.classes import Option, Future
import numpy as np


def generate_vanop():
    ft = Future('Z7', 100, 'C')
    tau = 0.2
    op1 = Option(100, tau, 'call', .20, ft, 'amer', False, 'Z7')
    op2 = Option(120, tau, 'call', .20, ft, 'amer', False, 'Z7')
    op3 = Option(80, tau, 'call', .20, ft, 'amer', False, 'Z7')
    op4 = Option(100, tau, 'put', .20, ft, 'amer', False, 'Z7')
    op5 = Option(120, tau, 'put', .20, ft, 'amer', False, 'Z7')
    op6 = Option(80, tau, 'put', .20, ft, 'amer', False, 'Z7')
    return op1, op2, op3, op4, op5, op6


def generate_barrop_euro():
    ft = Future('march', 300, 'C')
    tau = 327/365
    # european barriers
    # call up in
    op1 = Option(
        300, tau, 'call', .20, ft, 'amer', False, 'Z7', direc='up', barrier='euro', ki=320)
    # call up out
    op2 = Option(
        300, tau, 'call', .20, ft, 'amer', False, 'Z7', direc='up', barrier='euro', ko=320)
    # call down in
    op3 = Option(
        300, tau, 'call', .20, ft, 'amer', False, 'Z7', direc='down', barrier='euro', ki=280)
    # call down out
    op4 = Option(
        300, tau, 'call', .20, ft, 'amer', False, 'Z7', direc='down', barrier='euro', ko=280)
    # put up in
    op5 = Option(
        300, tau, 'put', .20, ft, 'amer', False, 'Z7', direc='up', barrier='euro', ki=320)
    # put up out
    op6 = Option(
        300, tau, 'put', .20, ft, 'amer', False, 'Z7', direc='up', barrier='euro', ko=320)
    # put down in
    op7 = Option(
        300, tau, 'put', .20, ft, 'amer', False, 'Z7', direc='down', barrier='euro', ki=280)
    # put down out
    op8 = Option(
        300, tau, 'put', .20, ft, 'amer', False, 'Z7', direc='down', barrier='euro', ko=280)
    # american barriers
    return op1, op2, op3, op4, op5, op6, op7, op8


def generate_barrop_amer():
    ft = Future('march', 100, 'C')
    # european barriers
    # call up in
    op1 = Option(
        100, 0.2, 'call', .20, ft, 'amer', False, 'Z7', direc='up', barrier='amer', ki=120)
    # call up out
    op2 = Option(
        100, 0.2, 'call', .20, ft, 'amer', False, 'Z7', direc='up', barrier='amer', ko=120)
    # call down in
    op3 = Option(
        100, 0.2, 'call', .20, ft, 'amer', False, 'Z7', direc='down', barrier='amer', ki=80)
    # call down out
    op4 = Option(
        100, 0.2, 'call', .20, ft, 'amer', False, 'Z7', direc='down', barrier='amer', ko=80)
    # put up in
    op5 = Option(
        100, 0.2, 'put', .20, ft, 'amer', False, 'Z7', direc='up', barrier='amer', ki=120)
    # put up out
    op6 = Option(
        100, 0.2, 'put', .20, ft, 'amer', False, 'Z7', direc='up', barrier='amer', ko=120)
    # put down in
    op7 = Option(
        100, 0.2, 'put', .20, ft, 'amer', False, 'Z7', direc='down', barrier='amer', ki=80)
    # put down out
    op8 = Option(
        100, 0.2, 'put', .20, ft, 'amer', False, 'Z7', direc='down', barrier='amer', ko=80)
    # american barriers
    return op1, op2, op3, op4, op5, op6, op7, op8


def test_bsm_euro():
    trueval_atm_call = 3.567
    trueval_itm_call = 20.016
    trueval_otm_call = 0.075
    trueval_atm_put = 3.567
    trueval_itm_put = 20.075
    trueval_otm_put = 0.016

    # testing calls
    op1, op2, op3, op4, op5, op6 = generate_vanop()
    assert np.isclose(op1.get_price(), trueval_atm_call, atol=1e-3)
    assert np.isclose(op2.get_price(), trueval_otm_call, atol=1e-3)
    assert np.isclose(op3.get_price(), trueval_itm_call, atol=1e-3)
    # testing puts
    assert np.isclose(op4.get_price(), trueval_atm_put, atol=1e-3)
    assert np.isclose(op5.get_price(), trueval_itm_put, atol=1e-3)
    assert np.isclose(op6.get_price(), trueval_otm_put, atol=1e-3)


def test_compute_iv():
    op1, op2, op3, op4, op5, op6 = generate_vanop()
    oplist = [op1, op2, op3, op4, op5, op6]
    ivlist = []
    truelist = [0.2243, 0.5787, 0, 0.2243, 0, 0.7092]
    for op in oplist:
        char = op.char
        s = op.underlying.get_price()
        k = op.K
        c = 4
        tau = op.tau
        r = 0
        payoff = op.payoff
        iv = _compute_iv(char, s, k, c, tau, r, payoff)
        ivlist.append(iv)
    try:
        assert np.allclose(ivlist, truelist, atol=1e-4)
    except AssertionError:
        print(ivlist)
        print(truelist)


# def test_iv_pathological():
#     result = _compute_iv('call', 100, 80, 0.0162787346047, 0.2, 0, 'euro')
#     print('pathological: ', result)


def test_barrier_amer():
    op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_amer()
    cdo = 3.57
    pdo = 3.29
    cdi = 0
    pdi = 0.28
    cuo = 2.81
    puo = 3.57
    cui = 0.76
    pui = 0
    plist = [op1.get_price(), op2.get_price(), op3.get_price(), op4.get_price(),
             op5.get_price(), op6.get_price(), op7.get_price(), op8.get_price()]
    actuals = [cui, cuo, cdi, cdo, pui, puo, pdi, pdo]
    try:
        assert np.allclose(plist, actuals, atol=1e-2)
    except AssertionError:
        print('barrier_amer_prices: ', plist)
        print('barrier_amer_actuals: ', actuals)


def test_barrier_euro():
    op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_euro()
    pass


def test_euro_vanilla_greeks():
    trueval_atm_call = [0.51784, 0.04456, -0.02442, 0.17823]
    trueval_itm_call = [0.99445, 0.00177, -0.00097, 0.0071]
    trueval_otm_call = [0.02309, 0.00611, -0.00335, 0.02445]
    trueval_atm_put = [-0.48216, 0.04456, -0.02442, 0.17823]
    trueval_itm_put = [-0.97691, 0.00611, -0.00335, 0.02445]
    trueval_otm_put = [-0.00555, 0.00177, -0.00097, 0.0071]
    tst = [trueval_atm_call, trueval_otm_call, trueval_itm_call,
           trueval_atm_put, trueval_itm_put, trueval_otm_put]
    op1, op2, op3, op4, op5, op6 = generate_vanop()
    oplist = [op1, op2, op3, op4, op5, op6]
    for i in range(len(oplist)):
        delta, gamma, theta, vega = oplist[i].greeks()
        glist = [delta, gamma, theta, vega]
        comp = tst[i]
        try:
            assert np.allclose(glist, comp, atol=1e-3)
        except AssertionError:
            print(glist)
            print(comp)


def test_euro_barrier_amer_greeks():
    op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_amer()

    cdi = [0, 0, 0, 0]
    pdi = [-0.0867, 0.0246, -0.0135, 0.0982]
    cdo = [0.5178, 0.0445, -0.0244, 0.1780]
    pdo = [-0.3955, 0.0198, -0.0110, 0.0797]
    cuo = [0.3111, -0.0019, 0.0010, -0.0070]
    puo = [-0.4822, 0.0445, -0.0244, 0.1780]
    cui = [0.2067, 0.0464, -0.0254, 0.1850]
    pui = [0, 0, 0, 0]

    op1greeks = list(op1.greeks())

    op2greeks = list(op2.greeks())
    op3greeks = list(op3.greeks())
    op4greeks = list(op4.greeks())
    op5greeks = list(op5.greeks())
    op6greeks = list(op6.greeks())
    op7greeks = list(op7.greeks())
    op8greeks = list(op8.greeks())
    try:
        assert np.allclose(op1greeks, cui, atol=1e-3)
    except AssertionError:
        print('predicted 1 : ', op1greeks)
        print('actual 1: ', cui)

    try:
        assert np.allclose(op2greeks, cuo, atol=1e-3)
    except AssertionError:
        print('predicted 2 : ', op2greeks)
        print('actual 2 : ', cuo)

    try:
        assert np.allclose(op3greeks, cdi, atol=1e-3)
    except AssertionError:
        print('predicted 3 : ', op3greeks)
        print('actual 3 : ', cdi)

    assert np.allclose(op4greeks, cdo, atol=1e-3)
    assert np.allclose(op5greeks, pui, atol=1e-3)
    assert np.allclose(op6greeks, puo, atol=1e-3)
    assert np.allclose(op7greeks, pdi, atol=1e-3)
    assert np.allclose(op8greeks, pdo, atol=1e-3)


def test_euro_barrier_euro_greeks():
    op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_euro()
    pass


# # NIU: Not in Use.
# def test_amer_barrier_euro_greeks():
#     pass

# # NIU: Not in Use.
# def test_amer_barrier_amer_greeks():
#     pass

# def test_num_vega():
#     pass

# # NIU
# def test_amer_vanilla_greeks():
#     pass

# # NIU: Not in Use.
# def test_CRRBinomial():
#     pass
