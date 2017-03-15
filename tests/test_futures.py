"""
File Name      : test_futures.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : File contains tests for the Futures class methods in classes.py

"""
from scripts.classes import Future


def test_get_price():
    ft = Future('march', 30, 'C')
    assert ft.get_price() == 30

    ft.update_price(50)
    assert ft.get_price() == 50


def test_get_desc():
    ft = Future('march', 30, 'C')
    assert ft.get_desc() == 'future'


def test_update_price():
    ft = Future('march', 30, 'C')
    assert ft.get_price() == 30
    ft.update_price(50)
    assert ft.get_price() == 50


def test_get_product():
    ft = Future('march', 30, 'C')
    assert ft.get_product() == 'C'


def test_get_month():
    ft = Future('march', 30, 'C')
    assert ft.get_month() == 'march'
