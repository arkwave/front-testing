# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-29 20:02:36
# @Last Modified by:   arkwave
# @Last Modified time: 2017-11-29 20:06:59

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


# FIXME
def test_trailingstop_processing():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    # print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    # print('pf_be: ', pf.breakeven())

    tstop = pf.get_hedger().get_intraday_conds()
    print('tstop: ', tstop)

    assert tstop.get_active() == {'QC  Z7': False, 'CC  Z7': False}
    assert not tstop.get_active(uid='QC  Z7')
    assert not tstop.get_active(uid='CC  Z7')
    # both should be equal to current prices present in the portfolio
    assert tstop.get_entry_level() == pf.uid_price_dict()
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_maximals() == pf.uid_price_dict()
    assert tstop.get_stop_values() == {'CC  Z7': None, 'QC  Z7': None}


def test_trailingstop_updates():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    # print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    # print('pf_be: ', pf.breakeven())

    tstop = pf.get_hedger().get_intraday_conds()
    print('-'*50)
    print('tstop: ', tstop)
    print('-'*50)
    assert tstop is not None
    assert isinstance(tstop, TrailingStop)
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}

    # update the prices.
    newprices = {'QC  Z7': 1610, 'CC  Z7': 1990}
    tstop.update_current_level(newprices)
    # QC Z7 should be active, CC Z7 inactive.
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_maximals() == newprices
    assert tstop.get_stop_values() == {'QC  Z7': 1605, 'CC  Z7': None}
    print('-'*50)
    print('first update: ', tstop)
    print('-'*50)
    # both should now be active.
    newprices = {'QC  Z7': 1650, 'CC  Z7': 2100}
    tstop.update_current_level(newprices)
    assert tstop.get_maximals() == newprices
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1645, 'CC  Z7': 2090}
    print('-'*50)
    print('second update: ', tstop)
    print('-'*50)
    # check if updates are passed.
    newprices = {'QC  Z7': 1660, 'CC  Z7': 2095}
    tstop.update_current_level(newprices)
    assert tstop.get_maximals() == {'QC  Z7': 1660, 'CC  Z7': 2100}
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1655, 'CC  Z7': 2090}
    print('-'*50)
    print('third update: ', tstop)
    print('-'*50)


def test_trailingstop_hit_sellstop():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    # print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    # print('pf_be: ', pf.breakeven())

    print('='*50)
    print('pf: ', pf)
    print('='*50)

    tstop = pf.get_hedger().get_intraday_conds()
    # print('tstop: ', tstop)
    assert tstop is not None
    assert isinstance(tstop, TrailingStop)
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}

    # update the prices.
    newprices = {'QC  Z7': 1610, 'CC  Z7': 1990}
    tstop.update_current_level(newprices)
    # print('first update: ', tstop)
    # QC Z7 should be active, CC Z7 inactive.
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_maximals() == newprices
    assert tstop.get_stop_values() == {'QC  Z7': 1605, 'CC  Z7': None}

    print('-'*50)
    print('first update: ', tstop)
    print('-'*50)

    # both should now be active.
    newprices = {'QC  Z7': 1650, 'CC  Z7': 2100}
    tstop.update_current_level(newprices)
    assert tstop.get_maximals() == newprices
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1645, 'CC  Z7': 2090}
    print('-'*50)
    print('second update: ', tstop)
    print('-'*50)
    # check if updates are passed.
    newprices = {'QC  Z7': 1660, 'CC  Z7': 2095}
    tstop.update_current_level(newprices)
    assert tstop.get_maximals() == {'QC  Z7': 1660, 'CC  Z7': 2100}
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1655, 'CC  Z7': 2090}
    print('-'*50)
    print('third update: ', tstop)
    print('-'*50)
    # QC hits trailing stop.
    newprices = {'QC  Z7': 1655, 'CC  Z7': 2095}
    tstop.update_current_level(newprices)
    assert tstop.trailing_stop_hit('QC  Z7')

    # CC hits trailing stop.
    newprices = {'QC  Z7': 1655, 'CC  Z7': 2090}
    tstop.update_current_level(newprices)
    assert tstop.trailing_stop_hit('CC  Z7')
    print('tstop after hit: ', tstop)


def test_trailingstop_update_locks():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    print('='*50)
    print('pf: ', pf)
    print('='*50)

    tstop = pf.get_hedger().get_intraday_conds()
    assert tstop is not None
    assert isinstance(tstop, TrailingStop)
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}
    assert tstop.get_locks() == {'QC  Z7': False, 'CC  Z7': False}

    # update the prices to move QC past threshold.
    newprices = {'QC  Z7': 1610, 'CC  Z7': 1990}
    tstop.update_current_level(newprices)
    # print('first update: ', tstop)
    # QC Z7 should be active, CC Z7 inactive.
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_locks() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1605, 'CC  Z7': None}

    # update the prices to move QC back within threshold. check that limit is
    # breached.
    newprices = {'QC  Z7': 1570, 'CC  Z7': 1990}
    tstop.update_current_level(newprices)
    assert tstop.trailing_stop_hit('QC  Z7')
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}


def test_trailingstop_hit_buystop():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    # print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    # print('pf_be: ', pf.breakeven())

    print('='*50)
    print('pf: ', pf)
    print('='*50)

    tstop = pf.get_hedger().get_intraday_conds()
    # print('tstop: ', tstop)
    assert tstop is not None
    assert isinstance(tstop, TrailingStop)
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}

    # update the prices.
    newprices = {'QC  Z7': 1520, 'CC  Z7': 1990}
    tstop.update_current_level(newprices)
    # print('first update: ', tstop)
    # QC Z7 should be active, CC Z7 inactive.
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    # assert tstop.get_maximals() == newprices
    assert tstop.get_stop_values() == {'QC  Z7': 1525, 'CC  Z7': None}

    print('-'*50)
    print('first update: ', tstop)
    print('-'*50)

    # both should now be active.
    newprices = {'QC  Z7': 1510, 'CC  Z7': 1930}
    tstop.update_current_level(newprices)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1515, 'CC  Z7': 1940}
    print('-'*50)
    print('second update: ', tstop)
    print('-'*50)

    # check if updates are passed.
    newprices = {'QC  Z7': 1514, 'CC  Z7': 1935}
    tstop.update_current_level(newprices)
    # assert tstop.get_maximals() == {'QC  Z7': 1660, 'CC  Z7': 2100}
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1515, 'CC  Z7': 1940}
    print('-'*50)
    print('third update: ', tstop)
    print('-'*50)

    # QC hits trailing stop.
    newprices = {'QC  Z7': 1515, 'CC  Z7': 1935}
    tstop.update_current_level(newprices)
    print('-'*50)
    print('fourth update: ', tstop)
    print('-'*50)
    assert tstop.trailing_stop_hit('QC  Z7')

    # CC hits trailing stop.
    newprices = {'QC  Z7': 1515, 'CC  Z7': 1940}
    tstop.update_current_level(newprices)
    print('-'*50)
    print('fifth update: ', tstop)
    print('-'*50)
    assert tstop.trailing_stop_hit('CC  Z7')
    print('tstop after hit: ', tstop)


def test_trailingstop_run_deltas_sellstop():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'trigger': {'QC  Z7': (30, 'price'),
                                             'CC  Z7': (50, 'price')},
                                 'value': {'QC  Z7': (5, 'price'),
                                           'CC  Z7': (10, 'price')}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7,
                                         intraday_params]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    # print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    # print('pf_be: ', pf.breakeven())

    print('='*50)
    print('pf: ', pf)
    print('='*50)

    tstop = pf.get_hedger().get_intraday_conds()
    print('='*50)
    print('tstop: ', tstop)
    print('='*50)

    assert tstop is not None
    assert isinstance(tstop, TrailingStop)
    assert tstop.get_current_level() == pf.uid_price_dict()
    assert tstop.get_active() == {'CC  Z7': False, 'QC  Z7': False}

    # update the prices.
    newprices = {'QC  Z7': 1520, 'CC  Z7': 1990}
    # QC Z7 should be active, CC Z7 inactive.
    assert tstop.run_deltas('QC  Z7', newprices)

    print('-'*50)
    print('first update: ', tstop)
    print('-'*50)
    assert not tstop.run_deltas('CC  Z7', newprices)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': False}
    assert tstop.get_stop_values() == {'QC  Z7': 1525, 'CC  Z7': None}

    # both should now be active.
    newprices = {'QC  Z7': 1510, 'CC  Z7': 1930}
    assert tstop.run_deltas('QC  Z7', newprices)
    assert tstop.run_deltas('CC  Z7', newprices)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1515, 'CC  Z7': 1940}
    print('-'*50)
    print('second update: ', tstop)
    print('-'*50)

    # check if updates are passed.
    newprices = {'QC  Z7': 1514, 'CC  Z7': 1935}
    assert tstop.run_deltas('QC  Z7', newprices)
    assert tstop.run_deltas('CC  Z7', newprices)
    assert tstop.get_active() == {'QC  Z7': True, 'CC  Z7': True}
    assert tstop.get_stop_values() == {'QC  Z7': 1515, 'CC  Z7': 1940}
    print('-'*50)
    print('third update: ', tstop)
    print('-'*50)

    # QC hits trailing stop.
    newprices = {'QC  Z7': 1515, 'CC  Z7': 1935}
    assert not tstop.run_deltas('QC  Z7', newprices)
    # after run_deltas call, anchor point should be reset.
    assert not tstop.trailing_stop_hit('QC  Z7')
    print('-'*50)
    print('fourth update: ', tstop)
    print('-'*50)

    # CC hits trailing stop.
    newprices = {'QC  Z7': 1515, 'CC  Z7': 1940}
    assert not tstop.run_deltas('CC  Z7', newprices)
    print('-'*50)
    print('fifth update: ', tstop)
    print('-'*50)
    assert not tstop.trailing_stop_hit('CC  Z7')
    print('tstop after hit: ', tstop)


def test_trailingstop_run_deltas_buystop():
    pass


def test_trailingstop_gapped_moves_buystop():
    pass


def test_trailingstop_gapped_moves_sellstop():
    pass
