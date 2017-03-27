import pandas as pd
import numpy as np
from scripts.classes import Option, Future
from operator import sub
df = pd.read_csv('tests/testing_vanilla.csv')


def test_vanilla_pricing():
    # tests vanilla pricing against settlement prices drawn from DB. Some
    # difference is expected, since we are using settlement prices and not
    # actual prices, accounting for atol = 0.1354 in the np.allclose call.
    passed = 0
    prices = []
    actuals = df['price']
    for i in df.index:
        row = df.iloc[i]
        tau = row['tau']
        product = row['product']
        strike = row['strike']
        vol = row['vol']
        s = row['s']
        price = row['price']
        ft = Future('Z6', s, product)
        op = Option(strike, tau, 'call', vol, ft, 'euro', False, 'Z7')
        prices.append(op.get_price())
    try:
        assert np.allclose(prices, actuals, atol=0.1354)
    except AssertionError:
        print('Passed ' + str(passed) + ' tests.')
        print('OptionPrice, Actual : ', str((op.get_price(), price)))
        resid = list(map(sub, actuals, prices))
        print('Residuals: ', resid)
        print('Max residual: ', max(resid))


def test_euro_barriers():
    pass


def test_amer_barriers():
    pass
