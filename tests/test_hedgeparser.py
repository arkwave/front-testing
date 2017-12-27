# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-30 21:19:46
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-27 21:24:48

from collections import OrderedDict
from scripts.util import create_straddle, combine_portfolios, assign_hedge_objects
from scripts.portfolio import Portfolio
from scripts.fetch_data import grab_data
from scripts.hedge_mods import TrailingStop
import copy
import numpy as np
import pandas as pd


############## variables ###########
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=False)
vdf.value_date = pd.to_datetime(vdf.value_date)
pdf.value_date = pd.to_datetime(pdf.value_date)

r_vdf = vdf[vdf.value_date == min(vdf.value_date)]
r_pdf = pdf[pdf.value_date == min(pdf.value_date)]
####################################


def comp_portfolio(refresh=True):
    # creating the options.
    ccops = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                            False, 'atm', greek='theta', greekval=10000)
    qcops = create_straddle('QC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                            True, 'atm', greek='theta', greekval=10000)
    # create the hedges.
    gen_hedges = OrderedDict({'delta': [['static', 0, 1]]})
    cc_hedges_s = {'delta': [['static', 0, 1],
                             ['roll', 50, 1, (-10, 10)]]}

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (9000, 11000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    qc_hedges = {'delta': [['roll', 50, 1, (-15, 15)]],
                 'theta': [['bound', (9000, 11000), 1, 'straddle',
                            'strike', 'atm', 'uid']]}
    cc_hedges_s = OrderedDict(cc_hedges_s)
    cc_hedges_c = OrderedDict(cc_hedges_c)
    qc_hedges = OrderedDict(qc_hedges)

    # create one simple and one complex portfolio.
    pf_simple = Portfolio(cc_hedges_s, name='cc_simple')
    pf_simple.add_security(ccops, 'OTC')

    pfcc = Portfolio(cc_hedges_c, name='cc_comp')
    pfcc.add_security(ccops, 'OTC')
    pfqc = Portfolio(qc_hedges, name='qc_comp')
    pfqc.add_security(qcops, 'OTC')

    pf_comp = combine_portfolios(
        [pfcc, pfqc], hedges=gen_hedges, refresh=refresh, name='full')

    return pf_simple, pf_comp, ccops, qcops, pfcc, pfqc


def test_basic():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}

    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1]]})
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    tstop = pf.get_hedger().get_intraday_conds()
    assert tstop is None

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be]]})
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    hp = pf.get_hedger().get_hedgeparser()

    assert hp.get_hedger_ratio() == 1
    assert hp.parse_hedges('eod') == {'QC  Z7': 0, 'CC  Z7': 0}
    assert hp.parse_hedges('intraday') == {'QC  Z7': 0, 'CC  Z7': 0}

    vals = {'CC  Z7': 30, 'QC  Z7': 30}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals]]})
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    hp = pf.get_hedger().get_hedgeparser()

    assert hp.get_hedger_ratio() == 1
    assert hp.parse_hedges('eod') == {'QC  Z7': 0, 'CC  Z7': 0}
    assert hp.parse_hedges('intraday') == {'QC  Z7': 0, 'CC  Z7': 0}

    gen_hedges_2 = OrderedDict({'delta': [['static', 0, 1],
                                          ['intraday', 'breakeven', be, 0.7,
                                           intraday_params]]})
    pf_comp.hedge_params = gen_hedges_2
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    hp = pf.get_hedger().get_hedgeparser()
    try:
        assert hp.get_hedger_ratio() == 0.7
    except AssertionError as e:
        raise AssertionError("expected 0.7, got '%s' " %
                             str(hp.get_hedger_ratio()))

    assert hp.parse_hedges('eod') == {'QC  Z7': 0, 'CC  Z7': 0}
    actual = hp.parse_hedges('intraday')
    for uid in actual:
        assert np.isclose(actual[uid], 0.3)


def test_gen_prices_basic():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})

    print('stop values: ', tstop.get_stop_values())

    assert hp.gen_prices(1560, 1565, 10, 'QC  Z7', copy.deepcopy(tstop)) == []

    assert hp.gen_prices(1560, 1580, 10, 'QC  Z7',
                         copy.deepcopy(tstop)) == [1570, 1580]

    try:
        prices = hp.gen_prices(1560, 1595, 10, 'QC  Z7',
                               copy.deepcopy(tstop))
        assert prices == [1570, 1580, 1590]
    except AssertionError as e:
        raise AssertionError(prices) from e

    try:
        prices = hp.gen_prices(1560, 1595, 12, 'QC  Z7',
                               copy.deepcopy(tstop))
        assert prices == [1572, 1584]
    except AssertionError as e:
        raise AssertionError(prices) from e

    assert hp.gen_prices(1560, 1570, 10, 'QC  Z7',
                         copy.deepcopy(tstop)) == [1570]

    assert hp.gen_prices(1560, 1550, 10, 'QC  Z7',
                         copy.deepcopy(tstop)) == [1550]
    try:
        val = hp.gen_prices(1560, 1630, 10, 'QC  Z7', copy.deepcopy(tstop))
        assert val == [1570, 1580, 1590]
    except AssertionError as e:
        raise AssertionError(val) from e

    assert hp.gen_prices(1560, 1540, 10, 'QC  Z7',
                         copy.deepcopy(tstop)) == [1550, 1540]

    try:
        val = hp.gen_prices(1560, 1520, 10, 'QC  Z7',
                            copy.deepcopy(tstop))
        assert val == [1550, 1540, 1530]
    except AssertionError as e:
        raise AssertionError(val) from e


def test_gen_prices_basic_2():
    vals = {'CC  Z7': 12, 'QC  Z7': 12}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (20, 'price'),
                                             'CC  Z7': (20, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})

    print('stop values: ', tstop.get_stop_values())

    assert hp.gen_prices(1560, 1565, 12, 'QC  Z7', copy.deepcopy(tstop)) == []

    try:
        prices = hp.gen_prices(1560, 1580, 12, 'QC  Z7',
                               copy.deepcopy(tstop))
        assert prices == [1572]
    except AssertionError as e:
        raise AssertionError(prices) from e

    assert hp.gen_prices(1560, 1590, 12, 'QC  Z7',
                         copy.deepcopy(tstop)) == [1572]

    # assert hp.gen_prices(1560, 1550, 10, 'QC  Z7',
    #                      copy.deepcopy(tstop)) == [1550]
    try:
        val = hp.gen_prices(1560, 1630, 12, 'QC  Z7', copy.deepcopy(tstop))
        assert val == [1572]
    except AssertionError as e:
        raise AssertionError(val) from e

    try:
        prices = hp.gen_prices(1560, 1539, 12, 'QC  Z7',
                               copy.deepcopy(tstop))
        assert prices == [1548]
    except AssertionError as e:
        print(hp)
        raise AssertionError(prices) from e

    try:
        t = copy.deepcopy(tstop)
        val = hp.gen_prices(1560, 1520, 12, 'QC  Z7', t)
        assert val == [1548]
        assert t.get_active('QC  Z7')
    except AssertionError as e:
        raise AssertionError(val) from e


def test_gen_prices_seq_sellstop():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})
    print('stop values: ', tstop.get_stop_values())

    assert hp.gen_prices(1560, 1565, 10, 'QC  Z7', tstop) == []
    assert hp.gen_prices(1560, 1570, 10, 'QC  Z7', tstop) == [1570]
    assert hp.gen_prices(1570, 1580, 10, 'QC  Z7', tstop) == [1580]
    assert tstop.get_thresholds() == {'QC  Z7': (
        1530, 1590), 'CC  Z7': (1956, 2016)}
    try:
        val = hp.gen_prices(1580, 1610, 10, 'QC  Z7', tstop)
        assert val == [1590]
    except AssertionError as e:
        raise AssertionError(val)

    assert hp.gen_prices(1610, 1620, 10, 'QC  Z7', tstop) == []
    assert hp.gen_prices(1620, 1630, 10, 'QC  Z7', tstop) == []

    # dip below to hit trailing stop.
    try:
        val = hp.gen_prices(1630, 1625, 10, 'QC  Z7', tstop)
        assert val == [1625]
    except AssertionError as e:
        raise AssertionError(val)

    # check to ensure that the trailngstop is updated appropriately.
    assert tstop.get_stop_values() == {'CC  Z7': None, 'QC  Z7': None}
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}
    assert tstop.get_anchor_points() == {'CC  Z7': 1986, 'QC  Z7': 1625}
    assert tstop.get_locks() == {'CC  Z7': False, 'QC  Z7': False}


def test_gen_prices_seq_buystop():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})
    print('stop values: ', tstop.get_stop_values())

    assert tstop.get_thresholds() == {'CC  Z7': (
        1956, 2016), 'QC  Z7': (1530, 1590)}
    assert hp.gen_prices(1560, 1555, 10, 'QC  Z7', tstop) == []
    assert tstop.get_stop_values('QC  Z7') is None
    assert hp.gen_prices(1560, 1550, 10, 'QC  Z7', tstop) == [1550]
    assert tstop.get_stop_values('QC  Z7') is None
    assert hp.gen_prices(1550, 1540, 10, 'QC  Z7', tstop) == [1540]
    assert tstop.get_stop_values('QC  Z7') is None
    try:
        val = hp.gen_prices(1540, 1520, 10, 'QC  Z7', tstop)
        assert val == [1530]
        assert tstop.get_stop_values('QC  Z7') is not None
        assert tstop.get_stop_values('QC  Z7') == 1525
    except AssertionError as e:
        raise AssertionError(val)

    assert hp.gen_prices(1510, 1500, 10, 'QC  Z7', tstop) == []
    assert hp.gen_prices(1500, 1490, 10, 'QC  Z7', tstop) == []

    assert tstop.get_current_level('QC  Z7') == 1490

    # jump up to hit trailing stop.
    try:
        val = hp.gen_prices(1490, 1495, 10, 'QC  Z7', tstop)
        assert val == [1495]
    except AssertionError as e:
        raise AssertionError(val)

    # check to ensure that the trailngstop is updated appropriately.
    assert tstop.get_stop_values() == {'CC  Z7': None, 'QC  Z7': None}
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}
    assert tstop.get_anchor_points() == {'CC  Z7': 1986, 'QC  Z7': 1495}
    assert tstop.get_locks() == {'CC  Z7': False, 'QC  Z7': False}


def test_gen_prices_seq_buystop_2():
    vals = {'CC  Z7': 12, 'QC  Z7': 12}
    interval = vals['QC  Z7']

    intraday_params = {'tstop': {'trigger': {'QC  Z7': (29, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()

    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})
    print('stop values: ', tstop.get_stop_values())

    print('################### INITIAL ##########################')
    assert tstop.get_thresholds() == {'CC  Z7': (
        1956, 2016), 'QC  Z7': (1531, 1589)}

    assert hp.gen_prices(1560, 1555, interval, 'QC  Z7', tstop) == []
    assert hp.gen_prices(1560, 1548, interval, 'QC  Z7', tstop) == [1548]

    print('#######################################################')

    print('############## BELOW THRESHOLD ###############')
    # becomes active by going below 1531 (lower threshold)
    try:
        prices = hp.gen_prices(1548, 1530, interval, 'QC  Z7', tstop)
        assert prices == [1536]
        assert tstop.get_active(uid='QC  Z7')
        assert tstop.get_current_level(uid='QC  Z7') == 1530
    except AssertionError as e:
        print(hp)
        raise AssertionError(prices) from e
    print('##############################################')

    # stays active. should return empty. comp = 1536 since this is
    # the value passed in to the call in granularize.
    print('############## ACTIVE EMPTY ###############')
    try:
        val = hp.gen_prices(1536, 1520, interval, 'QC  Z7', tstop)
        assert val == []
        assert tstop.get_current_level(uid='QC  Z7') == 1520
    except AssertionError as e:
        raise AssertionError(val)
    # check that the trailingstop value is accurate.
    try:
        assert tstop.get_stop_values(uid='QC  Z7') == 1525
    except AssertionError as e:
        raise AssertionError(tstop.get_stop_values(uid='QC  Z7'))
    print('###########################################')

    print('############## TSTOP HIT TEST ################')
    # price point that causes trailing stop to get hit.
    try:
        val = hp.gen_prices(1536, 1536, interval, 'QC  Z7', tstop)
        assert val == [1525]
        assert not tstop.get_active('QC  Z7')
        assert tstop.get_current_level('QC  Z7') == 1536
        assert tstop.get_anchor_points('QC  Z7') == 1525
        assert tstop.get_thresholds('QC  Z7') == (1496, 1554)
    except AssertionError as e:
        raise AssertionError(val) from e
    print('###############################################')

    # assert hp.gen_prices(1525, 1520, interval, 'QC  Z7', tstop) == []
    # assert hp.gen_prices(1525, 1490, interval, 'QC  Z7', tstop) == [1513, 1501]
    # assert tstop.get_current_level('QC  Z7') == 1490
    # assert tstop.get_active('QC  Z7')
    # assert tstop.get_locks()['QC  Z7']
    # assert tstop.get_stop_values('QC  Z7') == 1495

    # # jump up to hit trailing stop.
    # try:
    #     val = hp.gen_prices(1490, 1496, interval, 'QC  Z7', tstop)
    #     assert val == [1495]
    # except AssertionError as e:
    #     raise AssertionError(val)

    # # # check to ensure that the trailngstop is updated appropriately.
    # assert tstop.get_stop_values() == {'CC  Z7': None, 'QC  Z7': None}
    # assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}
    # assert tstop.get_anchor_points() == {'CC  Z7': 1986, 'QC  Z7': 1495}
    # assert tstop.get_locks() == {'CC  Z7': False, 'QC  Z7': False}


def test_relevant_price_move():
    be = {'CC': {'U7': 1, 'Z7': 1},
          'QC': {'U7': 1.5, 'Z7': 1}}

    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})

    print('TrailingStop params: ', tstop)

    # NOTE: in this case, the hedge interval for CC is consistently higher than
    # the trigger multiple. So this test really just tests handling of QC.

    # case 2, single breakeven move
    qc_interval = hedger.get_hedge_interval(uid='QC  Z7')
    prices = hp.relevant_price_move('QC  Z7', 1587, comparison=1560)
    try:
        assert prices == [1560 + qc_interval]
    except AssertionError as e:
        raise AssertionError(prices) from e

    # check trailngstop properties.
    try:
        assert tstop.get_current_level() == {
            'QC  Z7': 1587, 'CC  Z7': 1986}
        assert tstop.get_locks() == {'QC  Z7': False, 'CC  Z7': False}
        assert tstop.get_thresholds() == {'QC  Z7': (
            1530, 1590), 'CC  Z7': (1956, 2016)}
    except AssertionError as e:
        print(tstop)
        raise AssertionError from e

    # case 2, multiple breakeven move that blows through threshold.
    print('************** Second Move ***************')
    comp = 1586.9944395970483
    final = 1640.9944395970483
    assert tstop.get_thresholds() == {'QC  Z7': (
        1530, 1590), 'CC  Z7': (1956, 2016)}
    prices = hp.relevant_price_move(
        'QC  Z7', final, comparison=comp)
    try:
        assert prices == []
    except AssertionError as e:
        raise AssertionError(prices) from e

    # check trailingstop properties.
    assert tstop.get_active() == {"CC  Z7": False, 'QC  Z7': True}
    assert tstop.get_locks() == {"CC  Z7": False, 'QC  Z7': True}
    assert tstop.get_anchor_points() == {"CC  Z7": 1986, 'QC  Z7': 1560}
    try:
        assert np.isclose(tstop.get_stop_values(
            'QC  Z7'), 1640.9833187870001 - 5)
    except AssertionError as e:
        raise AssertionError(tstop.get_stop_values(
            'QC  Z7') - 1640.9833187870001 + 5) from e
    try:
        assert np.isclose(tstop.get_current_level(
            'QC  Z7'), 1640.9833187870001)
    except AssertionError as e:
        raise AssertionError(tstop.get_current_level(
            'QC  Z7') - 1640.9833187870001) from e
    print('==========================================')

    print('tstop: ', tstop)

    # case 3: similar move on the downside.
    comp = tstop.get_current_level('QC  Z7')
    new = 1600
    stopval = tstop.get_stop_values('QC  Z7')
    print('stopval: ', stopval)
    prices_1 = hp.relevant_price_move(
        'QC  Z7', new, comparison=tstop.get_current_level('QC  Z7'))

    # should be exactly 2 points: the stop value on the way down, and 1 be
    # from there.
    try:
        assert len(prices_1) == 2
    except AssertionError as e:
        raise AssertionError(prices_1) from e
    assert np.isclose(prices_1[0], stopval)
    assert np.isclose(prices_1[1], stopval-qc_interval)

    # sanity check: running the same query twice more should not matter.
    prices_2 = hp.relevant_price_move(
        'QC  Z7', new, comparison=tstop.get_current_level('QC  Z7'))
    prices_3 = hp.relevant_price_move(
        'QC  Z7', new, comparison=tstop.get_current_level('QC  Z7'))
    try:
        assert not prices_2
        assert not prices_3
    except AssertionError as e:
        raise AssertionError(prices_2, prices_3)

    # check trailingstop parameters.
    curr = tstop.get_current_level('QC  Z7')
    assert np.isclose(curr, 1600)
    print('curr: ', curr)
    print('thresholds: ', tstop.get_thresholds()['QC  Z7'])
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}


def test_relevant_price_move_pathological():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})
    print('stop values: ', tstop.get_stop_values())

    # first things first: activate by breaching upside barrier.
    prices = hp.relevant_price_move('QC  Z7', 1595, 1560)
    assert prices == [1570, 1580, 1590]
    print('tstop: ', tstop)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1590, 'CC  Z7': None}
    assert tstop.get_current_level() == {'QC  Z7': 1595, 'CC  Z7': 1986}

    # now, move from 1595 --> 1510. prices should be 1590, 1580, 1570, 1560.
    prices = hp.relevant_price_move('QC  Z7', 1510, 1595)

    assert tstop.get_current_level() == {'CC  Z7': 1986, 'QC  Z7': 1510}
    try:
        assert tstop.get_thresholds(uid='QC  Z7') == (1560, 1620)
    except AssertionError as e:
        print(tstop.get_thresholds(uid='QC  Z7'))
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1515, 'CC  Z7': None}
    assert prices == [1590, 1580, 1570, 1560]

    # similar move back up to 1595. prices should be 1515, 1525, 1535, 1545
    assert tstop.get_current_level() == {'QC  Z7': 1510, 'CC  Z7': 1986}
    prices = hp.relevant_price_move('QC  Z7', 1595, 1510)
    assert tstop.get_thresholds(uid='QC  Z7') == (1485, 1545)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1590, 'CC  Z7': None}
    assert prices == [1515, 1525, 1535, 1545]


def test_relevant_price_move_be_pathological():
    # vals = {'CC  Z7': 10, 'QC  Z7': 10}

    be = {'CC': {'Z7': 1}, 'QC': {'Z7': 1}}

    intraday_params = {'tstop': {'trigger': {'QC  Z7': (1, 'breakeven'),
                                             'CC  Z7': (1, 'breakeven')},
                                 'value': {'QC  Z7': (0.2, 'breakeven'),
                                           'CC  Z7': (0.2, 'breakeven')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    # print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
    #                             for uid in pf.get_unique_uids()})
    # print('stop values: ', tstop.get_stop_values())

    # first things first: activate by breaching upside barrier.
    prices = hp.relevant_price_move('QC  Z7', 1615, 1560)
    assert prices == [1587]
    print('tstop: ', tstop)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert np.isclose(tstop.get_stop_values('QC  Z7'), 1609.6)
    assert tstop.get_current_level() == {'QC  Z7': 1615, 'CC  Z7': 1986}

    # now, move from 1615 --> 1510. prices should be 1609.6, 1582.6
    prices = hp.relevant_price_move('QC  Z7', 1510, 1615)

    assert tstop.get_current_level() == {'CC  Z7': 1986, 'QC  Z7': 1510}
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert np.isclose(tstop.get_stop_values('QC  Z7'), 1515.4)
    assert prices == [1609.6, 1582.6]
    try:
        assert tstop.get_thresholds(uid='QC  Z7') == (1582.6, 1636.6)
    except AssertionError as e:
        print(tstop.get_thresholds(uid='QC  Z7'))

    # similar move back up to 1595. prices should be 1515.4, 1542.4.
    assert tstop.get_current_level() == {'QC  Z7': 1510, 'CC  Z7': 1986}
    prices = hp.relevant_price_move('QC  Z7', 1595, 1510)
    assert tstop.get_thresholds(uid='QC  Z7') == (1488.4, 1542.4)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert np.isclose(tstop.get_stop_values('QC  Z7'), 1589.6)
    assert prices == [1515.4, 1542.4]


def test_irrelevant_price_moves():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    tstop = hp.get_mod_obj()
    assert tstop is not None
    hedger = pf.get_hedger()
    # basic checks.
    assert isinstance(hp.get_mod_obj(), TrailingStop)
    assert hp.get_hedger_ratio() == 1

    print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
                                for uid in pf.get_unique_uids()})
    print('stop values: ', tstop.get_stop_values())

    prices = hp.relevant_price_move('QC  Z7', 1565, 1560)
    assert not prices
    print('tstop: ', tstop)
    assert tstop.get_active() == {'QC  Z7': False, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': None, 'CC  Z7': None}
    assert tstop.get_current_level() == {'QC  Z7': 1565, 'CC  Z7': 1986}

    # sequence of multiple irrelevant moves. all of these should return empty.
    p1 = hp.relevant_price_move('QC  Z7', 1564, 1565)
    assert not p1
    p2 = hp.relevant_price_move('QC  Z7', 1565, 1564)
    assert not p2
    p3 = hp.relevant_price_move('QC  Z7', 1563, 1565)
    assert not p3
    p4 = hp.relevant_price_move('QC  Z7', 1565, 1563)
    assert not p4
    p5 = hp.relevant_price_move('QC  Z7', 1563, 1565)
    assert not p5
    p6 = hp.relevant_price_move('QC  Z7', 1561, 1563)
    assert not p6

    assert tstop.get_active() == {'QC  Z7': False, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': None, 'CC  Z7': None}
    assert tstop.get_current_level() == {'QC  Z7': 1561, 'CC  Z7': 1986}
    assert tstop.get_thresholds() == {'QC  Z7': (
        1530, 1590), 'CC  Z7': (1956, 2016)}

    # break through upside barrier and then apply irrelevant moves.
    prices = hp.relevant_price_move('QC  Z7', 1591, 1561)
    assert prices == [1571, 1581]
    assert tstop.get_active('QC  Z7')
    assert tstop.get_stop_values(uid='QC  Z7') == 1586

    p1 = hp.relevant_price_move('QC  Z7', 1592, 1591)
    p2 = hp.relevant_price_move('QC  Z7', 1593, 1592)
    p3 = hp.relevant_price_move('QC  Z7', 1592, 1593)
    p4 = hp.relevant_price_move('QC  Z7', 1589, 1592)
    p5 = hp.relevant_price_move('QC  Z7', 1589, 1589)
    p6 = hp.relevant_price_move('QC  Z7', 1590, 1589)

    try:
        assert not p1
        assert not p2
        assert not p3
        assert not p4
        assert not p5
        assert not p6
    except AssertionError as e:
        print('p1: ', p1)
        print('p2: ', p2)
        print('p3: ', p3)
        print('p4: ', p4)
        print('p5: ', p5)
        print('p6: ', p6)
        raise AssertionError from e

    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1588, 'CC  Z7': None}
    assert tstop.get_current_level() == {'QC  Z7': 1590, 'CC  Z7': 1986}
    assert tstop.get_thresholds() == {'QC  Z7': (
        1530, 1590), 'CC  Z7': (1956, 2016)}

    # return to inactive by hitting trailing stop.
    print('***************** start ******************')

    prices = hp.relevant_price_move('QC  Z7', 1588, 1590)
    try:
        assert prices == [1588]
    except AssertionError as e:
        raise AssertionError(prices) from e
    print('**********************************************')

    # assert tstop.get_active() == {'QC  Z7': False, 'CC  Z7': False}
    # assert tstop.get_stop_values() == {'QC  Z7': None, 'CC  Z7': None}
    # assert tstop.get_current_level() == {'QC  Z7': 1588, 'CC  Z7': 1986}
    # assert tstop.get_anchor_points() == {'QC  Z7': 1588, 'CC  Z7': 1986}

    # # break through the downside barrier and then apply irrelevant moves.
    # prices = hp.relevant_price_move('QC  Z7', 1528, 1588)
    # assert prices == [1578, 1568, 1558]
    # assert tstop.get_stop_values(uid='QC  Z7') == 1533

    # p1 = hp.relevant_price_move('QC  Z7', 1529, 1528)
    # p2 = hp.relevant_price_move('QC  Z7', 1530, 1529)
    # p3 = hp.relevant_price_move('QC  Z7', 1531, 1530)
    # p4 = hp.relevant_price_move('QC  Z7', 1529, 1531)
    # p5 = hp.relevant_price_move('QC  Z7', 1528, 1529)
    # p6 = hp.relevant_price_move('QC  Z7', 1527, 1528)

    # assert tstop.get_stop_values(uid='QC  Z7') == 1532

    # try:
    #     assert not p1
    #     assert not p2
    #     assert not p3
    #     assert not p4
    #     assert not p5
    #     assert not p6
    # except AssertionError as e:
    #     print('p1: ', p1)
    #     print('p2: ', p2)
    #     print('p3: ', p3)
    #     print('p4: ', p4)
    #     print('p5: ', p5)
    #     print('p6: ', p6)
    #     raise AssertionError from e


def test_duplicates():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (30, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (5, 'price')}}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 1,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    hp = pf.get_hedgeparser(dup=True)
    hp2 = pf.get_hedgeparser()

    assert hp is not hp2
    assert hp.parent is not pf
