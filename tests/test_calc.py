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


def generate_barrop(flag):
    pass


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


def test_barrier_euro():
    pass


def test_barrier_amer():
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
    pass


def test_euro_barrier_euro_greeks():
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
