# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-08-11 19:24:36
# @Last Modified by:   arkwave
# @Last Modified time: 2017-11-27 21:46:20

from collections import OrderedDict
from scripts.util import create_straddle, combine_portfolios, assign_hedge_objects
from scripts.portfolio import Portfolio
from scripts.fetch_data import grab_data
from scripts.hedge import Hedge
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


def test_process_hedges():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio()
    # uid representation: no ttm
    cc_engine = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    dic = cc_engine.params
    assert 'theta' in dic
    assert 'delta' in dic
    dic = cc_engine.params['theta']
    assert dic['kind'] == 'straddle'
    assert dic['spectype'] == 'strike'
    assert dic['spec'] == 'atm'
    assert cc_engine.desc == 'uid'

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    # uid representation: ttm specified.
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 0.5, 'years', 'straddle', 'strike', 'atm', 'uid']]}
    pfcc.hedge_params = cc_hedges_c
    engine2 = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)

    dic2 = engine2.params
    assert 'theta' in dic2
    assert 'delta' in dic2
    dic2 = engine2.params['theta']
    assert dic2['kind'] == 'straddle'
    assert dic2['spectype'] == 'strike'
    assert dic2['spec'] == 'atm'
    assert dic2['tau_val'] == 0.5
    assert engine2.desc == 'uid'

    # exp representation with ttm specified.
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 0.5, 'years',
                              'straddle', 'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pfcc.hedge_params = cc_hedges_c
    engine3 = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    dic3 = engine3.params
    assert 'theta' in dic3
    assert 'delta' in dic3
    dic3 = engine3.params['theta']
    assert dic3['kind'] == 'straddle'
    assert dic3['spectype'] == 'strike'
    assert dic3['spec'] == 'atm'
    assert dic3['tau_val'] == 0.5
    assert engine3.desc == 'exp'
    assert engine3.buckets == [0, 20, 40, 60, 80]

    # exp represetion with no ttm specified
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}

    pfcc.hedge_params = cc_hedges_c
    engine4 = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    dic4 = engine4.params
    assert 'theta' in dic4
    assert 'delta' in dic4
    dic4 = engine4.params['theta']
    assert dic4['kind'] == 'straddle'
    assert dic4['spectype'] == 'strike'
    assert dic4['spec'] == 'atm'
    assert engine4.desc == 'exp'
    assert engine4.buckets == [0, 20, 40, 60, 80]

    # other tau descs
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 100, 'days',
                              'straddle', 'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pfcc.hedge_params = cc_hedges_c
    engine5 = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    dic5 = engine5.params
    assert 'theta' in dic5
    assert 'delta' in dic5
    dic5 = engine5.params['theta']
    assert dic5['kind'] == 'straddle'
    assert dic5['spectype'] == 'strike'
    assert dic5['spec'] == 'atm'
    assert dic5['tau_val'] == 100/365
    assert dic5['tau_desc'] == 'days'
    assert engine5.desc == 'exp'
    assert engine5.buckets == [0, 20, 40, 60, 80]

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 0.6, 'ratio',
                              'straddle', 'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pfcc.hedge_params = cc_hedges_c
    engine5 = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    dic5 = engine5.params
    assert 'theta' in dic5
    assert 'delta' in dic5
    dic5 = engine5.params['theta']
    assert dic5['kind'] == 'straddle'
    assert dic5['spectype'] == 'strike'
    assert dic5['spec'] == 'atm'
    assert dic5['tau_val'] == 0.6
    assert dic5['tau_desc'] == 'ratio'
    assert engine5.desc == 'exp'
    assert engine5.buckets == [0, 20, 40, 60, 80]


# NOTE: this test needs to be expanded to check for short-dated options.
# at the moment, results are trivial.
def test_calibrate():
    pass


def test_hedge_delta():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio()

    # uid representation: no ttm
    cc_engine = Hedge(pf_simple, pf_simple.hedge_params, r_vdf, r_pdf)

    print('cc_engine: ', cc_engine)

    # check that deltas are zeroed out.
    cc_engine.hedge_delta()
    assert abs(pf_simple.net_greeks['CC']['Z7'][0]) < 1
    assert cc_engine.satisfied()

    # check composites now.
    engine = Hedge(pf_comp, pf_comp.hedge_params, r_vdf, r_pdf)
    engine.hedge_delta()

    assert abs(pf_comp.net_greeks['CC']['Z7'][0]) < 1
    assert abs(pf_comp.net_greeks['QC']['Z7'][0]) < 1
    assert engine.satisfied()

    # all hedges should be satisfied.

    # simple to a fixed positive value.
    cc_hedges_s = {'delta': [['static', 5, 1],
                             ['roll', 50, 1, (-10, 10)]]}
    pf_simple.hedge_params = cc_hedges_s
    engine = Hedge(pf_simple, pf_simple.hedge_params, r_vdf, r_pdf)
    engine.hedge_delta()

    assert pf_simple.net_greeks['CC']['Z7'][0] < 6
    assert pf_simple.net_greeks['CC']['Z7'][0] > 4
    assert engine.satisfied()

    # comp to a fixed positive value.
    gen_hedges = OrderedDict({'delta': [['static', 5, 1]]})
    pf_comp.hedge_params = gen_hedges
    engine = Hedge(pf_comp, pf_comp.hedge_params, r_vdf, r_pdf)
    engine.hedge_delta()

    assert pf_comp.net_greeks['CC']['Z7'][0] < 6
    assert pf_comp.net_greeks['CC']['Z7'][0] > 4
    assert pf_comp.net_greeks['QC']['Z7'][0] < 6
    assert pf_comp.net_greeks['QC']['Z7'][0] > 4
    assert engine.satisfied()

    # trying negative delta values.

    # simple to a fixed negative value.
    cc_hedges_s = {'delta': [['static', -5, 1],
                             ['roll', 50, 1, (-10, 10)]]}
    pf_simple.hedge_params = cc_hedges_s
    engine = Hedge(pf_simple, pf_simple.hedge_params, r_vdf, r_pdf)
    engine.hedge_delta()

    assert pf_simple.net_greeks['CC']['Z7'][0] > -6
    assert pf_simple.net_greeks['CC']['Z7'][0] < -4
    assert engine.satisfied()

    # comp to a fixed negative value.
    gen_hedges = OrderedDict({'delta': [['static', -5, 1]]})
    pf_comp.hedge_params = gen_hedges
    engine = Hedge(pf_comp, pf_comp.hedge_params, r_vdf, r_pdf)
    engine.hedge_delta()

    assert pf_comp.net_greeks['CC']['Z7'][0] < -4
    assert pf_comp.net_greeks['CC']['Z7'][0] > -6
    assert pf_comp.net_greeks['QC']['Z7'][0] < -4
    assert pf_comp.net_greeks['QC']['Z7'][0] > -6
    assert engine.satisfied()


# TODO: add in as required.
def test_pos_type():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio()
    # uid representation: no ttm
    cc_engine = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)

    # straddle testing
    # short straddles to reduce gamma.
    assert cc_engine.pos_type('straddle', -10000, 'gamma')
    # short straddles to reduce vega
    assert cc_engine.pos_type('straddle', -10000, 'vega')
    # long straddles to increase vega
    assert not cc_engine.pos_type('straddle', 10000, 'vega')
    # long straddles to increase gamma
    assert not cc_engine.pos_type('straddle', 10000, 'gamma')
    # long straddles to short theta.
    assert not cc_engine.pos_type('straddle', -10000, 'theta')
    # short straddles to long theta
    assert cc_engine.pos_type('straddle', 10000, 'theta')

    # callop testing
    # short straddles to reduce gamma.
    assert cc_engine.pos_type('call', -10000, 'gamma')
    # short calls to reduce vega
    assert cc_engine.pos_type('call', -10000, 'vega')
    # long calls to increase vega
    assert not cc_engine.pos_type('call', 10000, 'vega')
    # long calls to increase gamma
    assert not cc_engine.pos_type('call', 10000, 'gamma')
    # long calls to short theta.
    assert not cc_engine.pos_type('call', -10000, 'theta')
    # short calls to long theta
    assert cc_engine.pos_type('call', 10000, 'theta')

    # putop testing
    # short straddles to reduce gamma.
    assert cc_engine.pos_type('put', -10000, 'gamma')
    # short puts to reduce vega
    assert cc_engine.pos_type('put', -10000, 'vega')
    # long puts to increase vega
    assert not cc_engine.pos_type('put', 10000, 'vega')
    # long puts to increase gamma
    assert not cc_engine.pos_type('put', 10000, 'gamma')
    # long puts to short theta.
    assert not cc_engine.pos_type('put', -10000, 'theta')
    # short puts to long theta
    assert cc_engine.pos_type('put', 10000, 'theta')


def test_add_hedges():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio()

    # testing uid repr

    # simple case: there should be no hedging options added.
    engine = Hedge(pfcc, pfcc.hedge_params, r_vdf, r_pdf)
    data = engine.params['theta']
    ops = engine.add_hedges(data, False, 'CC  Z7.Z7', 'theta', 0, 'Z7')
    assert not ops

    # nontrivial case
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (4500, 5000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}
    pf = copy.deepcopy(pfcc)
    pf.hedge_params = cc_hedges_c
    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    assert not engine.satisfied()
    data = engine.params['theta']
    # shorted = True because want to bring from -10,000 -> -5000
    ops = engine.add_hedges(data, True, 'CC  Z7.Z7', 'theta', 5250, 'Z7')
    assert len(ops) == 2
    total_theta = sum([op.theta for op in ops])
    assert total_theta > 5235
    assert total_theta < 5265
    assert engine.satisfied()
    chars = set(['call', 'put'])
    actchars = set([op.char for op in ops])
    assert actchars == chars

    # testing exp repr
    pf = copy.deepcopy(pfcc)

    # trivial case: exp repr
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (9000, 11000), 1, 'straddle',
                              'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pf.hedge_params = cc_hedges_c
    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    data = engine.params['theta']
    ops = engine.add_hedges(data, False, 'CC  Z7.Z7', 'theta', 0, 'Z7')
    assert not ops

    # exp case: bring theta down to -5000 from -10000 --> short straddles.
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (4500, 5000), 1, 'straddle',
                              'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pf = copy.deepcopy(pfcc)
    pf.hedge_params = cc_hedges_c
    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    assert not engine.satisfied()
    data = engine.params['theta']
    ops = engine.add_hedges(data, True, 'CC  Z7.Z7', 'theta', 5250, 'Z7')
    assert len(ops) == 2
    assert engine.satisfied()

    # removing theta; going from 10000 -> 5000 theta.
    pf = copy.deepcopy(pfqc)
    qc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (4500, 5000), 1, 'straddle',
                              'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}
    pf.hedge_params = qc_hedges_c
    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    data = engine.params['theta']

    assert not engine.satisfied()

    ops = engine.add_hedges(data, False, 'QC  Z7.Z7', 'theta', 5250, 'Z7')
    assert len(ops) == 2
    assert engine.satisfied()


def test_hedge():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio()

    # simple case
    pf = copy.deepcopy(pfcc)

    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    assert engine.desc == 'uid'
    assert engine.satisfied()

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (4500, 5000), 1, 'straddle',
                              'strike', 'atm', (0, 20, 40, 60, 80), 'exp']]}

    pf = copy.deepcopy(pfcc)
    pf.hedge_params = cc_hedges_c

    print('____________________ HEDGE TEST 1 _____________________')
    engine = Hedge(pf, pf.hedge_params, r_vdf, r_pdf)
    tval = pf.net_greeks['CC']['Z7'][2]
    bounds = pf.hedge_params['theta'][0][1]
    print('bounds: ', bounds)
    assert not engine.satisfied()

    print('engine.mappings: ', engine.mappings['theta'])

    engine.hedge('theta', 'CC', 80, tval, (bounds[0] + bounds[1])/2)

    assert engine.satisfied()
    print('____________________ HEDGE TEST 1 END _____________________')


def test_intraday_hedge_processing_static():
    # from scripts.util import assign_hedge_objects
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 0.7]]})
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges

    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)

    engine = pf.get_hedger()

    print('engine: ', engine)

    assert 'intraday' in engine.params['delta']
    assert len(engine.params['delta']['intraday']) == 5
    assert not engine.params['delta']['intraday']['conditions']
    assert engine.params['delta']['intraday']['kind'] == 'static'
    assert engine.params['delta']['intraday']['modifier'] == vals
    assert engine.params['delta']['intraday']['ratio'] == 0.7


def test_intraday_hedgeing_static():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges

    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)

    print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)

    # case 1: for both products, moves are less than static value.
    engine = pf.get_hedger()

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']
    # price_changes = {'CC  Z7': 5, 'QC  Z7': 5}

    # print('pf.net_greeks: ', pf.net_greeks)
    init_greeks = pf.net_greeks.copy()
    for x in ccfts:
        x.update_price(x.get_price() + 5)
    for x in qcfts:
        x.update_price(x.get_price() + 5)
    engine.hedge_delta(intraday=True)
    assert pf.net_greeks == init_greeks
    # print('pf.net_greeks: ', pf.net_greeks)

    # case 2: CC move exceeds static value.
    # price_changes = {'CC  Z7': 20, 'QC  Z7': 5}
    init_greeks = pf.net_greeks.copy()
    print('pf.net_greeks pre_hedge: ', pf.net_greeks)
    for x in ccfts:
        x.update_price(x.get_price() + 6)
    engine.hedge_delta(intraday=True)
    # pf.refresh()
    print('pf.net_greeks post_hedge: ', pf.net_greeks)
    # assert that CC is hedged, QC is not.
    assert abs(pf.net_greeks['CC']['Z7'][0]) < 1
    assert pf.net_greeks['QC'] == init_greeks['QC']

    # case 3: QC move exceeds breakeven
    # price_changes = {'CC  Z7': 30, 'QC  Z7': 56}
    for x in qcfts:
        x.update_price(x.get_price() + 11)
    engine.hedge_delta(intraday=True)
    assert abs(pf.net_greeks['QC']['Z7'][0]) < 1


def test_intraday_hedging_static_ratio():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals, 0.7]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges

    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)

    print('pf: ', pf)
    print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)

    # case 1: for both products, moves are less than static value.
    engine = pf.get_hedger()

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']
    # price_changes = {'CC  Z7': 5, 'QC  Z7': 5}

    # print('pf.net_greeks: ', pf.net_greeks)
    init_greeks = pf.net_greeks.copy()
    for x in ccfts:
        x.update_price(x.get_price() + 5)
    for x in qcfts:
        x.update_price(x.get_price() + 5)
    engine.hedge_delta(intraday=True)
    assert pf.net_greeks == init_greeks
    # print('pf.net_greeks: ', pf.net_greeks)

    pf.refresh()

    # case 2: CC move exceeds static value.
    # price_changes = {'CC  Z7': 20, 'QC  Z7': 5}
    init_greeks = pf.net_greeks.copy()
    print('pf.net_greeks pre_hedge: ', pf.net_greeks)
    for x in ccfts:
        x.update_price(x.get_price() + 6)

    pf.refresh()

    init_delta = pf.net_greeks['CC']['Z7'][0]
    print('init delta: ', init_delta)
    engine.hedge_delta(intraday=True)

    pf.refresh()
    # assert that CC is hedged, QC is not.
    try:
        assert pf.net_greeks['QC'] == init_greeks['QC']
    except AssertionError as e:
        print('actual: ', pf.net_greeks['QC'])
        print('desired: ', init_greeks['QC'])
        raise AssertionError from e

    try:
        assert np.isclose(round(abs(pf.net_greeks['CC']['Z7'][0])), 17)
    except AssertionError as e:
        raise AssertionError(
            round(abs(pf.net_greeks['CC']['Z7'][0])), 17) from e

    # case 3: QC move exceeds breakeven
    # price_changes = {'CC  Z7': 30, 'QC  Z7': 56}
    for x in qcfts:
        x.update_price(x.get_price() + 11)
    pf.refresh()
    init_qc_delta = pf.net_greeks['QC']['Z7'][0]
    print('init_qc_delta: ', init_qc_delta)
    engine.hedge_delta(intraday=True)
    assert np.isclose(round(abs(pf.net_greeks['QC']['Z7'][0])), 49)


def test_intraday_hedge_processing_be():
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    intraday_params = {'tstop': {'type': 'breakeven',
                                 'value': {'QC': 1, 'CC': 1.5}}}

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

    engine = pf.get_hedger()
    print('engine: ', engine)
    assert 'intraday' in engine.params['delta']
    assert len(engine.params['delta']['intraday']) == 5
    assert engine.params['delta']['intraday']['kind'] == 'breakeven'
    assert engine.params['delta']['intraday']['modifier'] == be
    assert engine.params['delta']['intraday']['ratio'] == 0.7
    assert engine.params['delta']['intraday']['conditions'] == intraday_params


def test_intraday_hedging_be():
    # price_changes = {'CC  Z7': 20, 'QC  Z7': 20}
    be = {'CC': {'U7': 1, 'Z7': 1},
          'QC': {'U7': 1, 'Z7': 1}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)
    print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)
    print('pf_be: ', pf.breakeven())

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']

    # case 1: for both products, moves are less than breakeven * be_mult. no
    # change.
    engine = pf.get_hedger()
    # print('pf.net_greeks: ', pf.net_greeks)
    init_greeks = pf.net_greeks.copy()
    for x in ccfts:
        x.update_price(x.get_price() + 5)
    for x in qcfts:
        x.update_price(x.get_price() + 5)
    engine.hedge_delta(intraday=True)
    assert pf.net_greeks == init_greeks
    # print('pf.net_greeks: ', pf.net_greeks)

    # case 2: CC move exceeds breakeven * mult.
    # price_changes = {'CC  Z7': 28, 'QC  Z7': 15}
    init_greeks = pf.net_greeks.copy()
    print('pf.net_greeks pre_hedge: ', pf.net_greeks)
    for x in ccfts:
        x.update_price(x.get_price() + 33)
    for x in qcfts:
        x.update_price(x.get_price() + 20)
    engine.hedge_delta(intraday=True)
    print('pf.net_greeks post_hedge: ', pf.net_greeks)
    # assert that CC is hedged, QC is not.
    assert abs(pf.net_greeks['CC']['Z7'][0]) < 1
    assert pf.net_greeks['QC'] == init_greeks['QC']

    # case 3: QC move exceeds breakeven
    # price_changes = {'CC  Z7': 30, 'QC  Z7': 56}
    for x in ccfts:
        x.update_price(x.get_price() + 45)
    for x in qcfts:
        x.update_price(x.get_price() + 8)
    engine.hedge_delta(intraday=True)
    assert abs(pf.net_greeks['QC']['Z7'][0]) < 1


def test_intraday_hedging_be_ratio():
    # price_changes = {'CC  Z7': 20, 'QC  Z7': 20}
    be = {'CC': {'U7': 1, 'Z7': 1},
          'QC': {'U7': 1, 'Z7': 1}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be, 0.7]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']

    # case 1: for both products, moves are less than breakeven * be_mult. no
    # change.
    engine = pf.get_hedger()
    init_greeks = pf.net_greeks.copy()
    for x in ccfts:
        x.update_price(x.get_price() + 5)
    for x in qcfts:
        x.update_price(x.get_price() + 5)
    engine.hedge_delta(intraday=True)
    assert pf.net_greeks == init_greeks
    # refresh post price update.
    pf.refresh()

    # case 2: CC move exceeds breakeven * mult.
    for x in ccfts:
        x.update_price(x.get_price() + 33)
    for x in qcfts:
        x.update_price(x.get_price() + 20)
    pf.refresh()

    # save the greeks for later comparison.
    init_greeks = pf.net_greeks.copy()
    pre_hedge_cc_delta = round(abs(pf.net_greeks['CC']['Z7'][0]))

    # hedge and refresh portfolio.
    engine.hedge_delta(intraday=True)
    pf.refresh()
    init_cc_delta = round(abs(pf.net_greeks['CC']['Z7'][0]))

    # assert that CC is hedged, QC is not.
    try:
        assert round(init_cc_delta/pre_hedge_cc_delta, 1) == 0.3
    except AssertionError as e:
        raise AssertionError('desired value: ',
                             round(init_cc_delta/pre_hedge_cc_delta, 1))
    assert pf.net_greeks['QC'] == init_greeks['QC']

    # case 3: QC move exceeds breakeven
    # price_changes = {'CC  Z7': 30, 'QC  Z7': 56}
    for x in ccfts:
        x.update_price(x.get_price() + 45)
    for x in qcfts:
        x.update_price(x.get_price() + 8)
    pf.refresh()
    pre_hedge_qc_delta = round(abs(pf.net_greeks['QC']['Z7'][0]))
    engine.hedge_delta(intraday=True)
    pf.refresh()
    new_qc_delta = round(abs(pf.net_greeks['QC']['Z7'][0]))
    assert round(new_qc_delta/pre_hedge_qc_delta, 1) == 0.3


def test_breakeven():
    # checks that the breakeven dictionary does not change when prices/vols
    # are updated
    be = {'CC': {'U7': 1, 'Z7': 1.3},
          'QC': {'U7': 1.5, 'Z7': 2}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be]]})
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)
    init_be = pf.breakeven().copy()
    print('pf_be: ', init_be)

    hedge_engine = pf.get_hedger()
    assert init_be == hedge_engine.breakeven
    # update the prices/vols of the options in pf, and assert that
    # the be dict in the hedge object remains the same.

    for ft in pf.get_all_futures():
        ft.update_price(ft.get_price() + 2)

    for op in pf.get_all_options():
        op.update_greeks(vol=(op.vol + 0.2))
    pf.refresh()
    assert hedge_engine.breakeven == init_be


# TODO: fix this.
def test_trailing_stop_hit():
    pass


# TODO: fix this.
def test_is_relevant_price_move_tstop_price():
    be = {'CC': {'U7': 1, 'Z7': 1},
          'QC': {'U7': 1, 'Z7': 1}}
    intraday_params = {'tstop': {'type': 'price',
                                 'trigger': 30,
                                 'value': {'QC': 1, 'CC': 1.5}}}

    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be,
                                         intraday_params]]})
    # intraday_params =
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)
    print('pf: ', pf)

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']

    ccprice = ccfts[0].get_price()
    qcprice = qcfts[0].get_price()

    assert not pf.hedger.is_relevant_price_move('CC  Z7', ccprice + 30)[0]

    for x in ccfts:
        x.update_price(2000)

    assert pf.hedger.is_relevant_price_move('CC  Z7', ccprice + 38)[0]
    assert not pf.hedger.is_relevant_price_move('QC  Z7', qcprice + 20)[0]
    assert pf.hedger.is_relevant_price_move('QC  Z7', qcprice + 28)[0]


# TODO: fix this.
def test_is_relevant_price_move_tstop_be_mult():
    pass


def test_is_relevant_price_move_be():
    be = {'CC': {'U7': 1, 'Z7': 1},
          'QC': {'U7': 1, 'Z7': 1}}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'breakeven', be]]})
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges
    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)
    init_be = pf.breakeven()
    print('be: ', init_be)

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']

    ccprice = ccfts[0].get_price()
    qcprice = qcfts[0].get_price()

    assert not pf.hedger.is_relevant_price_move('CC  Z7', ccprice + 30)[0]
    assert pf.hedger.is_relevant_price_move('CC  Z7', ccprice + 38)[0]
    assert not pf.hedger.is_relevant_price_move('QC  Z7', qcprice + 20)[0]
    assert pf.hedger.is_relevant_price_move('QC  Z7', qcprice + 28)[0]


def test_is_relevant_price_move_static():
    vals = {'CC  Z7': 10, 'QC  Z7': 10}
    gen_hedges = OrderedDict({'delta': [['static', 0, 1],
                                        ['intraday', 'static', vals]]})

    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    pf_comp.hedge_params = gen_hedges

    # assign hedge objects and create copy
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf, vdf=r_vdf, pdf=r_pdf)

    print('pf.hedger.hedgpoints: ', pf.hedger.last_hedgepoints)

    # case 1: for both products, moves are less than static value.
    engine = pf.get_hedger()

    ccfts = [x for x in pf.get_all_futures() if x.get_uid() == 'CC  Z7']
    qcfts = [x for x in pf.get_all_futures() if x.get_uid() == 'QC  Z7']

    ccprice = ccfts[0].get_price()
    qcprice = qcfts[0].get_price()

    assert not engine.is_relevant_price_move('CC  Z7', ccprice + 5)[0]
    assert engine.is_relevant_price_move('CC  Z7', ccprice + 10)[0]
    assert not engine.is_relevant_price_move('QC  Z7', qcprice + 5)[0]
    assert engine.is_relevant_price_move('QC  Z7', qcprice + 10)[0]
