# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-30 21:19:46
# @Last Modified by:   arkwave
# @Last Modified time: 2017-11-30 22:22:26

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
