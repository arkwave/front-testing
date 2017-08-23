# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-08-11 19:24:36
# @Last Modified by:   arkwave
# @Last Modified time: 2017-08-23 22:27:28

from collections import OrderedDict
from scripts.util import create_straddle, combine_portfolios
from scripts.portfolio import Portfolio
from scripts.fetch_data import grab_data
from scripts.hedge import Hedge
import copy
import pandas as pd

############## variables ###########
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=True)
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
    gen_hedges = OrderedDict({'delta': [['static', 'zero', 1]]})
    cc_hedges_s = {'delta': [['static', 'zero', 1],
                             ['roll', 50, 1, (-10, 10)]]}

    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
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
    try:
        assert 'theta' in dic3
        assert 'delta' in dic3
    except AssertionError:
        print('dic3: ', dic3)
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
    try:
        assert 'theta' in dic5
        assert 'delta' in dic5
    except AssertionError:
        print('dic5: ', dic5)
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
    try:
        assert 'theta' in dic5
        assert 'delta' in dic5
    except AssertionError:
        print('dic5: ', dic5)
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


def test_get_bucket():
    pass


def test_uid_hedges_satisfied():
    pass


def test_exp_hedges_satisfied():
    pass


def test_hedge_delta():
    pass


def test_pos_type():
    pass


def test_add_hedges():
    pass


def test_apply():
    pass
