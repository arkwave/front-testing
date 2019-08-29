# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-30 21:19:46
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-29 14:02:35

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
    ccops = create_straddle('CC  Z7.Z7', vdf, pdf,
                            False, 'atm', greek='theta', greekval=10000)
    qcops = create_straddle('QC  Z7.Z7', vdf, pdf,
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


# def test_relevant_price_move():
#     be = {'CC': {'U7': 1, 'Z7': 1},
#           'QC': {'U7': 1.5, 'Z7': 1}}

#     intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
#                                              'CC  Z7': (30, 'price')},
#                                  'value': {'QC  Z7': (5, 'price'),
#                                            'CC  Z7': (5, 'price')}}}

#     gen_hedges = OrderedDict({'delta': [['static', 0, 1],
#                                         ['intraday', 'breakeven', be, 1,
#                                          intraday_params]]})

#     pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
#     pf_comp.hedge_params = gen_hedges
#     pf = copy.deepcopy(pf_comp)
#     pf = assign_hedge_objects(pf)

#     hp = pf.get_hedgeparser(dup=True)

#     print('hp: ', hp)

#     tstop = hp.get_mod_obj()
#     assert tstop is not None
#     hedger = pf.get_hedger()
#     # basic checks.
#     assert isinstance(hp.get_mod_obj(), TrailingStop)
#     assert hp.get_hedger_ratio() == 1

#     print('hedge intervals: ', {uid:  hedger.get_hedge_interval(uid)
#                                 for uid in pf.get_unique_uids()})

#     print('TrailingStop params: ', tstop)

#     # NOTE: in this case, the hedge interval for CC is consistently higher than
#     # the trigger multiple. So this test really just tests handling of QC.
#     # case 2, single breakeven move
#     r1, m1, p1 = hp.relevant_price_move('QC  Z7', 1587, comparison=1560)
#     assert r1
#     assert m1 == 1
#     try:
#         assert p1 == [1560, 1560+hedger.get_hedge_interval('QC  Z7')]
#     except AssertionError as e:
#         raise AssertionError('p1: ', p1)

#     # check trailngstop properties.
#     try:
#         assert tstop.get_current_level() == {
#             'QC  Z7': 1560+hedger.get_hedge_interval('QC  Z7'), 'CC  Z7': 1986}
#         assert tstop.get_locks() == {'QC  Z7': False, 'CC  Z7': False}
#         assert tstop.get_thresholds() == {'QC  Z7': (
#             1530, 1590), 'CC  Z7': (1956, 2016)}
#     except AssertionError as e:
#         print(tstop)
#         raise AssertionError from e
#     # case 2, multiple breakeven move
#     print('************** Second Move ***************')
#     r2, m2, p2 = hp.relevant_price_move(
#         'QC  Z7', 1640.9944395970483, comparison=1586.9944395970483)
#     assert r2
#     assert m2 == 2
#     assert p2 == [1586.9944395970483, 1586.9944395970483+hedger.get_hedge_interval(
#         'QC  Z7'), 1586.9944395970483 + 2*hedger.get_hedge_interval('QC  Z7')]
#     print('==========================================')
