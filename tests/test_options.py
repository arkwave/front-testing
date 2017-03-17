"""
File Name      : test_options.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 16/3/2017
Python version : 3.5
Description    : File contains tests for Options class methods in classes.py

"""
from scripts.classes import Option, Future


def test_check_active_ko_american():
    # testing Option.check_active on American Knock-Out options.
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    direc = 'up'
    barrier = 'amer'

    # testing up and out
    # test 1: up and out at 50, spot at 30.
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ko=50)
    assert op.check_active() == True
    assert op.knockedout == False

    # test 2: up and out at 20, spot at 30
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ko=20)
    assert op.check_active() == False
    assert op.knockedout == True

    # devil's advocate: up and out at 20, spot falls from 30 to 10. Option is
    # still knocked out.
    ft.update_price(10)
    assert op.check_active() == False
    assert op.knockedout == True

    # testing down and out.
    # test 3: down and out at 50, spot at 70.
    ft2 = Future('march', 70, 'C')
    strike2 = 75
    direc2 = 'down'
    barrier2 = 'amer'
    op2 = Option(strike2, tau, 'call', vol, ft2, payoff,
                 direc=direc2, barrier=barrier2, bullet=False, ko=50)
    assert op2.check_active() == True
    assert op2.knockedout == False

    # test 4: down and out at 50, spot at 49.
    ft2.update_price(49)
    assert op2.check_active() == False
    assert op2.knockedout == True


def test_check_active_ki_american():
    # testing check_active on american barrier knock-in options.
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    direc = 'down'
    barrier = 'amer'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ki=20)

    # testing down and in
    # test 1: down and in at 20, spot at 30.
    assert op.check_active() == True
    assert op.knockedin == False

    # test 2: spot hits barrier, option knocks in.
    ft.update_price(20)
    assert op.check_active() == True
    assert op.knockedin == True

    # test 3: price rises after knocked in.
    ft.update_price(30)
    assert op.check_active() == True
    assert op.knockedin == True

    # testing up and in
    # test 4: up and in at 50, spot at 30.
    ft2 = Future('march', 30, 'C')
    direc2 = 'up'
    op2 = Option(strike, tau, 'call', vol, ft2, payoff,
                 direc=direc2, barrier=barrier, ki=50)
    assert op2.check_active() == True
    assert op2.knockedin == False

    # test 5: spot hits barrier,option knocks in.
    ft2.update_price(50)
    assert op2.check_active() == True
    assert op2.knockedin == True

    # test 6: spot falls after knock in. option is still active.
    ft2.update_price(30)
    assert op2.check_active() == True
    assert op2.knockedin == True


def test_check_active_ko_euro():
    # testing Options.check_active on European barrier knock-out options.
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 0.01
    vol = 0.2
    payoff = 'euro'
    direc = 'up'
    barrier = 'euro'

    # testing up and out
    # test 1: up and out at 50, spot at 30.
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ko=50)
    # op.update_tau(0.01)
    assert op.check_active() == True
    assert op.knockedout == False

    op.update_tau(0.01)

    assert op.check_active() == False
    assert op.knockedout == False

    # test 2: up and out at 20, spot at 30
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ko=20)
    assert op.check_active() == False
    assert op.knockedout == True

    # devil's advocate: up and out at 20, spot falls from 30 to 10. Option is
    # still knocked out.
    ft.update_price(10)
    assert op.check_active() == False
    assert op.knockedout == True

    # testing down and out.
    # test 3: down and out at 50, spot at 70.
    ft2 = Future('march', 70, 'C')
    strike2 = 75
    direc2 = 'down'
    barrier2 = 'amer'
    op2 = Option(strike2, tau, 'call', vol, ft2, payoff,
                 direc=direc2, barrier=barrier2, bullet=False, ko=50)
    assert op2.check_active() == True
    assert op2.knockedout == False

    # test 4: down and out at 50, spot at 49.
    ft2.update_price(49)
    assert op2.check_active() == False
    assert op2.knockedout == True


def test_check_active_ki_euro():
    # testing Options.check_active on European barrier knock-in options.
    # testing check_active on american barrier knock-in options.
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 0.01
    vol = 0.2
    payoff = 'euro'
    direc = 'down'
    barrier = 'amer'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc=direc, barrier=barrier, bullet=False, ki=20)

    # testing down and in
    # test 1: down and in at 20, spot at 30.
    assert op.check_active() == True
    assert op.knockedin == False

    # test 2: spot hits barrier, option knocks in.
    ft.update_price(20)
    assert op.check_active() == True
    assert op.knockedin == True

    # test 3: price rises after knocked in.
    ft.update_price(30)
    assert op.check_active() == True
    assert op.knockedin == True

    op.update_tau(0.01)
    assert op.tau == 0
    assert op.check_active() == False
    assert op.knockedin == True

    # testing up and in
    # test 4: up and in at 50, spot at 30.
    ft2 = Future('march', 30, 'C')
    direc2 = 'up'
    op2 = Option(strike, tau, 'call', vol, ft2, payoff,
                 direc=direc2, barrier=barrier, ki=50)
    assert op2.check_active() == True
    assert op2.knockedin == False

    # test 5: spot hits barrier,option knocks in.
    ft2.update_price(50)
    assert op2.check_active() == True
    assert op2.knockedin == True

    # test 6: spot falls after knock in. option is still active.
    ft2.update_price(30)
    assert op2.check_active() == True
    assert op2.knockedin == True


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
    assert op.exercise() == False
    assert op2.exercise() == True


def test_moneyness_american():
    # tests moneyness function for american barrier calls and puts.

    # call options
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='amer', bullet=False, ko=50)
    # at the money
    assert op.moneyness() == 0
    # in the moneu
    ft.update_price(35)
    assert op.moneyness() == 1
    # out of the money
    ft.update_price(20)
    assert op.moneyness() == -1
    # hit barrier; knocked out.
    ft.update_price(50)
    assert op.check_active() == False
    assert op.knockedout == True
    assert op.moneyness() == None
    # futher changes should not affect moneyness since option has knocked out.
    ft.update_price(20)
    assert op.moneyness() == None
    ft.update_price(35)
    assert op.moneyness() == None

    # put options
    strike2 = 20
    ft2 = Future('march', 20, 'C')
    op2 = Option(strike2, tau, 'put', vol, ft2, payoff,
                 direc='up', barrier='amer', bullet=False, ko=50)
    assert op2.moneyness() == 0
    ft2.update_price(30)
    assert op2.moneyness() == -1
    ft2.update_price(10)
    assert op2.moneyness() == 1
    ft2.update_price(50)
    assert op2.moneyness() == None


def test_moneyness_euro():
    # tests Options.moneyness() for european barrier options.

    # call options
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='euro', bullet=False, ko=50)
    # at the money
    assert op.moneyness() == 0
    # in the money
    ft.update_price(35)
    assert op.moneyness() == 1
    # out of the money
    ft.update_price(20)
    assert op.moneyness() == -1
    # hit barrier; knocked out.
    ft.update_price(50)
    assert op.check_active() == True
    assert op.knockedout == True
    assert op.moneyness() == None
    # futher changes should not affect moneyness since option has knocked out.
    ft.update_price(20)
    assert op.moneyness() == None
    ft.update_price(35)
    assert op.moneyness() == None

    # put options
    strike2 = 20
    ft2 = Future('march', 20, 'C')
    op2 = Option(strike2, tau, 'put', vol, ft2, payoff,
                 direc='up', barrier='amer', bullet=False, ko=50)
    assert op2.moneyness() == 0
    ft2.update_price(30)
    assert op2.moneyness() == -1
    ft2.update_price(10)
    assert op2.moneyness() == 1
    ft2.update_price(50)
    assert op2.moneyness() == None


def test_updates_passed():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='amer', bullet=False, ko=50)
    initial_val = op.get_price()
    d1, g1, t1, v1 = op.greeks()

    ft.update_price(40)
    curr_val = op.get_price()
    d2, g2, t2, v2 = op.greeks()
    # check greeks have updated.
    assert [d1, g1, t1, v1] != [d2, g2, t2, v2]
    # postive movement in spot == positive delta.
    assert d2 > d1
    # check that current value has increased.
    assert initial_val != curr_val
    assert initial_val < curr_val
    # breaks knockout barrier; option is no longer active.
    ft.update_price(50)
    assert op.check_active() == False
    assert op.knockedout == True


def test_zero_options():
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 0.01
    vol = 0.2
    payoff = 'euro'
    op = Option(strike, tau, 'call', vol, ft, payoff,
                direc='up', barrier='amer', bullet=False, ko=50)
    delta, gamma, theta, vega = op.greeks()
    initial_val = op.get_price()
    assert initial_val > 0
    assert [x != 0 for x in [delta, gamma, theta, vega]]

    op.update_tau(0.01)
    assert op.check_active() == False
    final = op.get_price()
    delta, gamma, theta, vega = op.greeks()
    assert [x == 0 for x in [delta, gamma, theta, vega]]
    assert final == 0


def test_barrier_options():
    # check if knock in barriers = regular vanilla after knock in
    ft = Future('march', 30, 'C')
    strike = 30
    tau = 30/365
    vol = 0.5
    payoff = 'euro'
    vanop = Option(strike, tau, 'call', vol, ft, payoff)
    barOp = Option(strike, tau, 'call', vol, ft, payoff,
                   direc='up', barrier='amer', bullet=False, ki=40)
    d1, g1, t1, v1 = vanop.greeks()
    p1 = vanop.get_price()
    d2, g2, t2, v2 = barOp.greeks()
    assert barOp.knockedin == False
    assert barOp.check_active() == True
    p2 = barOp.get_price()

    # before knock-in, price of vanilla and barrier should differ.
    assert p1 != p2

    # price rises, knocks in barrier.
    ft.update_price(55)
    # after knockin, greeks and price should be identical.
    d1, g1, t1, v1 = vanop.greeks()
    p1 = vanop.get_price()
    d2, g2, t2, v2 = barOp.greeks()
    p2 = barOp.get_price()
    assert (p1 == p2)
    l1 = [d1, g1, t1, v1]
    l2 = [d2, g2, t2, v2]
    assert l1 == l2


# TODO: Figure out why this holds only for ITM options. Should hold for all.
def test_barrier_options2():
    ft2 = Future('march', 35, 'C')
    strike2 = 35
    tau2 = 30/365
    vol2 = 0.2
    payoff2 = 'euro'
    vanop2 = Option(strike2, tau2, 'call', vol2, ft2, payoff2)
    barOp2 = Option(strike2, tau2, 'call', vol2, ft2, payoff2,
                    direc='up', barrier='amer', bullet=False, ko=36)
    d1, g1, t1, v1 = vanop2.greeks()
    p3 = vanop2.get_price()
    d2, g2, t2, v2 = barOp2.greeks()
    p4 = barOp2.get_price()
    try:
        assert p4 < p3
    except AssertionError:
        print('vanilla: ', p3)
        print("barrier: ", p4)


# TODO if necessary:
# test init greeks
# test update greeks (pretty sure this works fine since it is being called by all the others.
# test greeks (redundant)
