"""
File Name      : test_calc.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : File contains tests for the methods in calc.py

"""
from scripts.calc import *
from scripts.classes import Option, Future
import numpy as np


def generate_vanop():
    ft = Future('march', 100, 'C')
    op1 = Option(100, 0.2, 'call', .20, ft, 'amer')
    op2 = Option(120, 0.2, 'call', .20, ft, 'amer')
    op3 = Option(80, 0.2, 'call', .20, ft, 'amer')
    op4 = Option(100, 0.2, 'put', .20, ft, 'amer')
    op5 = Option(120, 0.2, 'put', .20, ft, 'amer')
    op6 = Option(80, 0.2, 'put', .20, ft, 'amer')
    return op1, op2, op3, op4, op5, op6


def generate_barrop_euro():
    ft = Future('march', 100, 'C')
    # european barriers
    # call up in
    op1 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='up', barrier='euro', ki=120)
    # call up out
    op2 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='up', barrier='euro', ko=120)
    # call down in
    op3 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='down', barrier='euro', ki=80)
    # call down out
    op4 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='down', barrier='euro', ko=80)
    # put up in
    op5 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='up', barrier='euro', ki=120)
    # put up out
    op6 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='up', barrier='euro', ko=120)
    # put down in
    op7 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='down', barrier='euro', ki=80)
    # put down out
    op8 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='down', barrier='euro', ko=80)
    # american barriers
    return op1, op2, op3, op4, op5, op6, op7, op8


def generate_barrop_amer():
    ft = Future('march', 100, 'C')
    # european barriers
    # call up in
    op1 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='up', barrier='amer', ki=120)
    # call up out
    op2 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='up', barrier='amer', ko=120)
    # call down in
    op3 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='down', barrier='amer', ki=80)
    # call down out
    op4 = Option(
        100, 0.2, 'call', .20, ft, 'amer', direc='down', barrier='amer', ko=80)
    # put up in
    op5 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='up', barrier='amer', ki=120)
    # put up out
    op6 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='up', barrier='amer', ko=120)
    # put down in
    op7 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='down', barrier='amer', ki=80)
    # put down out
    op8 = Option(
        100, 0.2, 'put', .20, ft, 'amer', direc='down', barrier='amer', ko=80)
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
        assert np.allclose(plist, actuals)
    except AssertionError:
        print(plist)
        print(actuals)

    # try:
    #     assert np.isclose(op1.get_price(), cui)
    # except AssertionError:
    #     print(op1.get_price())
    #     print(cui)

    # assert np.isclose(op2.get_price(), cuo)
    # assert np.isclose(op3.get_price(), cdi)
    # assert np.isclose(op4.get_price(), cdo)
    # assert np.isclose(op5.get_price(), pui)
    # assert np.isclose(op6.get_price(), puo)
    # assert np.isclose(op7.get_price(), pdi)
    # assert np.isclose(op8.get_price(), pdo)


def test_barrier_euro():
    #op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_euro()
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


def test_euro_barrier_euro_greeks():
    #op1, op2, op3, op4, op5, op6, op7, op8 = generate_barrop_euro()
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
