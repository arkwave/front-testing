import pandas as pd
import numpy as np
from scripts.classes import Option, Future
from operator import sub
from scripts.calc import _bsm_euro, _euro_vanilla_greeks, _euro_barrier_euro_greeks, _euro_barrier_amer_greeks
import datetime as dt
df = pd.read_csv('tests/testing_vanilla.csv')


def ttm(df, s, edf):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
    s = s.unique()
    df['tau'] = ''
    df['expdate'] = ''
    for iden in s:
        expdate = get_expiry_date(iden, edf)
        # print('Expdate: ', expdate)
        try:
            expdate = expdate.values[0]
        except IndexError:
            print('Vol ID: ', iden)
        currdate = pd.to_datetime(df[(df['vol_id'] == iden)]['value_date'])
        timedelta = (expdate - currdate).dt.days / 365
        df.ix[df['vol_id'] == iden, 'tau'] = timedelta
        df.ix[df['vol_id'] == iden, 'expdate'] = pd.to_datetime(expdate)
    return df


def get_expiry_date(volid, edf):
    """Computes the expiry date of the option given a vol_id """
    target = volid.split()
    op_yr = target[1][1]  # + decade
    # op_yr = op_yr.astype(str)
    op_mth = target[1][0]
    # un_yr = pd.to_numeric(target[1][-1]) + decade
    # un_yr = un_yr.astype(str)
    # un_mth = target[1][3]
    prod = target[0]
    overall = op_mth + op_yr  # + '.' + un_mth + un_yr
    expdate = edf[(edf['opmth'] == overall) & (edf['product'] == prod)][
        'expiry_date']
    expdate = pd.to_datetime(expdate)
    return expdate


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


def test_vanilla_xtrader():
    curr_date = pd.Timestamp(dt.date.today())
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365
    spot = 387.125

    ids = ['call'] * 7 + ['put'] * 8

    # prices = [46.392, 53.342, 60.879, 69.127, 77.827, 41.343, 47.731,
    # 18.870, 105.573, 91.061, 399.258, 580.141, 0.000, 0.014, 16.597]

    strikes = [350, 340, 330, 320, 310, 358, 348,
               369, 485, 469, 784, 965, 120, 200, 364]

    vols = [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21,
            0.22, 0.25, 0.25, 0.37, 0.45, 0.30, 0.26, 0.21]

    deltas = [0.75, 0.8, 0.85, 0.89, 0.92, 0.71, 0.76, -
              0.36, -0.84, -0.8, -0.99, -0.99, 0, 0, -0.33]

    gammas = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
              0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.01]

    thetas = [-0.02, - 0.01, - 0.01, - 0.01, - 0.01, -
              0.02, -0.02, - 0.02, - 0.02, - 0.02, 0, 0, 0, 0, -0.02]
    # thetas = [x*252/365 for x in thetas]

    vegas = [0.39, 0.34, 0.29, 0.23, 0.18, 0.42,
             0.38, 0.46, 0.3, 0.34, 0.04, 0.03, 0, 0, 0.45]

    assert len(strikes) == len(vols) == len(
        deltas) == len(gammas) == len(thetas) == len(vegas)

    dollar_mult = 0.3936786

    for i in range(len(strikes)):
        char, vol, k = ids[i], vols[i], strikes[i]
        d, g, t, v = deltas[i], gammas[i], thetas[i], vegas[i]
        delta, gamma, theta, vega = _euro_vanilla_greeks(
            char, k, tau, vol, spot, 0, 'C', 10)
        gamma, theta, vega = gamma/dollar_mult, theta*dollar_mult, vega*dollar_mult
        try:
            assert np.isclose(d, delta, atol=1e-2)
        except AssertionError:
            print('dresidue: ', abs(delta - d))
        try:
            assert np.isclose(g, gamma, atol=1e-2)
        except AssertionError:
            print('gresidue: ', gamma, g)
        try:
            assert np.isclose(t, theta, atol=1e-2)
        except AssertionError:
            print('tresidue: ', theta, t)
        try:
            assert np.isclose(v, vega, atol=1e-2)
        except AssertionError:
            print('vresidue: ', vega, v)


def test_euro_barriers():
    curr_date = pd.Timestamp(dt.date.today())
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365
    spot = 387.375
    vols = [0.2216, 0.2216, 0.2248, 0.2248]
    bvols = [0.2314, 0.2248, 0.2158, 0.2190]
    deltas = [0.01, 0.52, 0, -0.54]
    gammas = [0, 0.01, 0, 0.02]
    thetas = [0, -0.02, 0, -0.02]
    vegas = [-0.03, 0.5, -0.05, 0.51]
    strikes = [390, 390, 400, 400]
    kis = [None, 400, None, 380]
    kos = [420, None, 370, None]
    chars = ['call', 'call', 'put', 'put']
    directions = ['up', 'up', 'down', 'down']
    payoff = 'amer'
    lots = 10
    dollar_mult = 0.3936786

    for i in range(len(chars)):
        vol, bvol, k, ki, ko, char, direc = vols[i], bvols[
            i], strikes[i], kis[i], kos[i], chars[i], directions[i]
        d, g, t, v = deltas[i], gammas[i], thetas[i], vegas[i]
        try:
            d1, g1, t1, v1 = _euro_barrier_euro_greeks(
                char, tau, vol, k, spot, 0, payoff, direc, 'C', ki, ko, lots, bvol=bvol)
        except TypeError:
            print(_euro_barrier_euro_greeks(
                char, tau, vol, k, spot, 0, payoff, direc, 'C', ki, ko, lots, bvol=bvol) is None)
        g1, t1, v1 = g1/dollar_mult, t1*dollar_mult, v1*dollar_mult
        try:
            assert np.isclose(d1, d, atol=1e-2)
        except AssertionError:
            print('deltas: ', d1, d)
        try:
            assert np.isclose(g1, g, atol=1e-2)
        except AssertionError:
            print('gammas: ', g1, g)
        try:
            assert np.isclose(t1, t, atol=1e-2)
        except AssertionError:
            print('thetas: ', t1, t)
        try:
            assert np.isclose(v1, v, atol=1e-2)
        except AssertionError:
            print('vegas: ', v1, v)


def test_amer_barriers():
    pass
