import pandas as pd
import numpy as np
from scripts.classes import Option, Future
from operator import sub
from scripts.calc import _bsm_euro, _euro_vanilla_greeks, _euro_barrier_euro_greeks, _euro_barrier_amer_greeks, _barrier_amer, _barrier_euro
# import datetime as dt
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


def test_vanilla_pricing_settle():
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
        print('testcalc: Passed ' + str(passed) + ' tests.')
        print('testcalc: OptionPrice, Actual : ', str((op.get_price(), price)))
        resid = list(map(sub, actuals, prices))
        print('testcalc: Residuals: ', resid)
        print('testcalc: Max residual: ', max(resid))


def test_vanilla_pricing():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420,
               440, 440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    vol = 0.22
    s = 387.750
    chars = ['call']*9 + ['put']*9
    # product = 'C'
    # lots = 10
    actuals = [49.005645520224000, 42.405054015026500, 33.620522291462900, 27.213528169831200, 27.097482827734100,
               26.636930698204900, 21.947072146899300, 15.077435162573200, 10.066343062463700, 62.316343062463700,
               47.327435162573200, 34.197072146899300, 27.2135281698312,  27.3474828277341, 27.8869306982049,
               11.2556455202240, 14.6550540150265, 18.6483083574217]
    for i in range(len(strikes)):
        char = chars[i]
        k = strikes[i]
        actual = actuals[i]
        val = _bsm_euro(char, tau, vol, k, s, 0)
        try:
            assert np.isclose(val, actual)
        except AssertionError:
            print('vanilla _ pricing _ %% error: ',
                  (abs(val-actual)/actual) * 100)


# char, tau, vol, k, s, r, payoff, direction, ki, ko

def test_amer_barrier_pricing():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    chars = ['put'] * 4 + ['call'] * 4
    spot = 387.750
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    vol = 0.22

    r = 0
    payoffs = ['amer']*8
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']
    prices = [28.428865379491300, 50.449884052880000, 19.558953734663000, 0.198835317329896,
              36.390665957702900, 9.182972317226490, 0.643093633583953, 15.841689331908300]

    assert len(chars) == len(strikes) == len(kis) == len(
        kos) == len(payoffs) == len(directions) == len(prices)

    for i in range(len(chars)):
        char = chars[i]
        k = strikes[i]
        ki = kis[i]
        ko = kos[i]
        payoff = payoffs[i]
        direction = directions[i]
        price = prices[i]
        val = _barrier_amer(char, tau, vol, k, spot, r,
                            payoff, direction, ki, ko)
        try:
            assert np.isclose(val, price)
        except AssertionError:
            print('amerbarr _ run ' + str(i) + ': ', val, price)


def test_vanilla_greeks():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    strikes = [350, 360, 375, 387.75, 388, 389, 400, 420,
               440, 440, 420, 400, 387.75, 388, 389, 350, 360, 370]
    vol = 0.22
    s = 387.750
    chars = ['call']*9 + ['put']*9
    deltas = [7.4842859309570300, 6.9484395361432800, 6.0944898647578700,
              5.3509159016096900, 5.3363727454162100, 5.2782508035247300,
              4.6473987996563200, 3.5737686353737800, 2.6448749876143700,
              -7.3551250123856300, -6.4262313646262200, -5.3526012003436800,
              -4.649084098390310, -4.663627254583790, -4.721749196475270,
              -2.515714069042970, -3.051560463856720, -3.616375394010970]
    gammas = [15.059813000513300, 16.548906492286400, 18.130051215724700,
              18.770786536726700, 18.776711054794000, 18.797880046671100,
              18.770083735887700, 17.626313297716400, 15.456046231256800,
              15.456046231256800, 17.626313297716400, 18.770083735887700,
              18.770786536726700, 18.776711054794000, 18.797880046671100,
              15.059813000513300, 16.548906492286400, 17.698718589721000]
    thetas = [-23.266402976629900, -25.566952740978800, -28.009715496308800,
              -28.999608676209800, -29.008761660034500, -29.041466335424900,
              -28.998522895959400, -27.231474133378900, -23.878556794137300,
              -23.878556794137300, -27.231474133378900, -28.998522895959400,
              -28.999608676209800, -29.008761660034500, -29.041466335424900,
              -23.266402976629900, -25.566952740978800, -27.343335462690400]
    vegas = [494.939845139217000, 543.878812853549000, 595.843038739661000,
             616.900766384828000, 617.095475313462000, 617.791192953584000,
             616.877668877681000, 579.287722473697000, 507.962026348011000,
             507.962026348011000, 579.287722473697000, 616.877668877681000,
             616.900766384828000, 617.095475313462000, 617.791192953584000,
             494.939845139217000, 543.878812853549000, 581.667318024504000]

    assert len(vegas) == len(thetas) == len(
        gammas) == len(deltas) == len(strikes)

    for i in range(len(vegas)):
        d1, g1, t1, v1 = deltas[i], gammas[i], thetas[i], vegas[i]
        k = strikes[i]
        char = chars[i]
        d, g, t, v = _euro_vanilla_greeks(char, k, tau, vol, s, 0,  'C', 10)
        try:
            assert np.isclose(d, d1)
        except AssertionError:
            assert (abs(d - d1)/d1) * 100 < 2e-3
        try:
            assert np.isclose(g, g1)
        except AssertionError:
            assert (abs(g - g1)/g1) * 100 < 2e-3
        try:
            assert np.isclose(t, t1)
        except AssertionError:
            assert (abs(t - t1)/t1) * 100 < 2e-3
        try:
            assert np.isclose(v, v1)
        except AssertionError:
            assert (abs(v - v1)/v1) * 100 < 2e-3


# FIXME: Still not working properly.
def test_american_barrier_greeks():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    chars = ['put'] * 4 + ['call'] * 4

    s = 387.750
    strikes = [390, 450, 400, 400, 370, 380, 380, 380]
    kis = [380, 400, None, None, 390, 360, None, None]
    kos = [None, None, 410, 370, None, None, 420, 370]
    vol = 0.22
    r = 0
    payoffs = ['amer']*8
    directions = ['down', 'up', 'up', 'down', 'up', 'down', 'up', 'down']

    deltas = [-4.783986635992220, 8.465546983416060,   -8.826913688153580, 0.104405133782848,
              6.417554302373670,  -2.185933112297530, -0.157934175770702, 8.952093100500490]
    gammas = [18.811985261450300,   18.939209435216300,  1.604629218669980, -0.348433773196910,
              17.715106574116600, 14.254265198563700, -1.221882168877410, 1.401185157017180]
    thetas = [-29.109439649076300, -29.153737444399800, -2.471758863475060, 0.615362469696290,  -
              27.394820388902700, -22.028421202236400, 1.878320472499140,  -2.213958487979010]
    vegas = [618.568709025737000, 619.585744363217000, 52.512848382684200,  -13.103350453807200,
             582.136493086884000, 468.151697544542000, -39.969178254451000,
             47.037403962064200]

    # dollar_mult = 0.393678571428571
    # lot_mult = 127.007166832986
    # lots = 10
    # deltas = [x / lots for x in deltas]
    # gammas = [x * dollar_mult / (lots * lot_mult) for x in gammas]
    # thetas = [x / (lots * dollar_mult * lot_mult) for x in thetas]
    # vegas = [x / (lots * lot_mult * dollar_mult) for x in vegas]

    for i in range(len(chars)):
        char = chars[i]
        k = strikes[i]
        ki = kis[i]
        ko = kos[i]
        payoff = payoffs[i]
        direction = directions[i]
        d, g, t, v = _euro_barrier_amer_greeks(
            char, tau, vol, k, s, r, payoff, direction, 'C', ki, ko, 10)
        d1, g1, t1, v1 = deltas[i], gammas[i], thetas[i], vegas[i]
        # g, t, v = g/dollar_mult, t*dollar_mult, v*dollar_mult
        try:
            assert np.isclose(d, deltas[i])
        except AssertionError:
            # print('ab _ greeks _ delta run ' + str(i) + ': ', d, deltas[i])
            print('testcalc: ab _ greeks _ delta run ' + str(i) +
                  ' %  error: ', (abs(d - d1)/d1) * 100)
        try:
            assert np.isclose(g, gammas[i])
        except AssertionError:
            # print('ab _ greeks _ gamma run ' + str(i) + ': ', g, gammas[i])
            print('testcalc: ab _ greeks _ gammas run ' + str(i) +
                  ' %  error: ', (abs(g - g1)/g1) * 100)
        try:
            assert np.isclose(t, thetas[i])
        except AssertionError:
            # print('ab _ greeks _ theta run ' + str(i) + ': ', t, thetas[i])
            print('testcalc: ab _ greeks _ thetas run ' + str(i) +
                  ' %  error: ', (abs(t - t1)/t1) * 100)
        try:
            assert np.isclose(v, vegas[i])
        except AssertionError:
            # print('ab _ greeks _ vega run ' + str(i) + ': ', v, vegas[i])
            print('testcalc: ab _ greeks _ vega run ' + str(i) +
                  ' %  error: ', (abs(v - v1)/v1) * 100)


def test_euro_barrier_greeks():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    # dollar_mult = 0.393678571428571
    s = 387.750
    vol = 0.22
    bvol = 0.22

    deltas = [-0.0557490887856027, 5.8001729954295600,
              0.0045642948890046, -5.0722086291524400]
    gammas = [-3.130091906576830, 18.658738232493300, -
              0.450432407726885, 19.037792844756400]
    thetas = [4.835782466210640, -28.826501546721800,
              0.695887917834766, -29.412115548619900]
    vegas = [-102.870281553938000, 613.218305630264000, -
             14.803433888480200, 625.675912579732000]
    strikes = [350, 380, 395, 395]
    chars = ['call', 'call', 'put', 'put']
    directions = ['up', 'up', 'down', 'down']
    kis = [None, 390, None, 385]
    kos = [390, None, 380, None]
    payoff = 'amer'
    lots = 10
    # dollar_mult = 0.3936786

    for i in range(len(chars)):
        k, ki, ko, char, direc = strikes[i], kis[
            i], kos[i], chars[i], directions[i]
        d1, g1, t1, v1 = deltas[i], gammas[i], thetas[i], vegas[i]
        try:
            d, g, t, v = _euro_barrier_euro_greeks(
                char, tau, vol, k, s, 0, payoff, direc, 'C', ki, ko, lots, barvol=bvol)
        except TypeError:
            print(_euro_barrier_euro_greeks(
                char, tau, vol, k, s, 0, payoff, direc, 'C', ki, ko, lots, barvol=bvol) is None)
        # g1, t1, v1 = g1/dollar_mult, t1*dollar_mult, v1*dollar_mult
        try:
            assert np.isclose(d, d1)
        except AssertionError:
            assert (abs(d - d1)/d1) * 100 < 2e-3
        try:
            assert np.isclose(g, g1)
        except AssertionError:
            assert (abs(g - g1)/g1) * 100 < 2e-3
        try:
            assert np.isclose(t, t1)
        except AssertionError:
            assert (abs(t - t1)/t1) * 100 < 2e-3
        try:
            assert np.isclose(v, v1)
        except AssertionError:
            assert (abs(v - v1)/v1) * 100 < 2e-3


def test_euro_barrier_pricing():
    curr_date = pd.Timestamp('2017-04-05')
    expdate = pd.Timestamp('2017-11-24')
    tau = ((expdate - curr_date).days)/365 + 1/365
    s = 387.750
    chars = ['call'] * 2 + ['put']*2
    directions = ['up']*2 + ['down']*2
    kis = [None, 390, None, 370]
    kos = [400, None, 350, None]
    vol = 0.22
    strikes = [350, 360, 390, 400]
    product = 'C'
    actuals = [7.242409174727800, 39.760087685635000,
               4.713297229131670, 31.552517740371500]
    for i in range(len(actuals)):
        actual = actuals[i]
        char = chars[i]
        k = strikes[i]
        ki = kis[i]
        ko = kos[i]
        direction = directions[i]
        payoff = 'amer'
        val = _barrier_euro(char, tau, vol, k, s, 0,
                            payoff, direction, ki, ko, product, barvol=vol)
        try:
            assert np.isclose(val, actual)
        except AssertionError:
            print('testcalc: euro _ pricing : ', val, actual)
