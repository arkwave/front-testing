# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-08-09 17:01:19
# @Last Modified by:   Ananth
# @Last Modified time: 2017-09-21 18:06:28

from scripts.util import combine_portfolios, create_straddle, create_vanilla_option, create_underlying, merge_dicts, merge_lists, transfer_dict, assign_hedge_objects
from scripts.fetch_data import grab_data
from scripts.portfolio import Portfolio
from collections import OrderedDict
import unittest as un
import pandas as pd
import copy

############## variables ###########
yr = 2017
start_date = '2017-07-01'
end_date = '2017-08-10'
pdts = ['QC', 'CC']

# grabbing data
vdf, pdf, edf = grab_data(pdts, start_date, end_date,
                          write_dump=False)

start_date = min(pdf.value_date)
end_date = max(pdf.value_date)
####################################


def test_create_vanilla_option():
    tc = un.TestCase
    # check that not passing in delta and strike raises error.
    with tc.assertRaises(create_vanilla_option, ValueError) as e1:
        op = create_vanilla_option(vdf, pdf, 'CC  Z7.Z7', 'call', False)

    assert isinstance(e1.exception, ValueError)

    # check that bad date value throws indexerror.
    # fakedate = 'lolol'
    # tc2 = un.TestCase
    # with tc2.assertRaises(create_vanilla_option, IndexError) as e2:
    #     op = create_vanilla_option(
    #         vdf, pdf, 'CC  Z7.Z7', 'call', False, delta=25, date=fakedate)
    # assert isinstance(e2.exception, ValueError)

    # check that bad volid throws indexerror
    fakevolid = 'CX KEK'
    tc3 = un.TestCase
    with tc3.assertRaises(create_vanilla_option, IndexError) as e3:
        op = create_vanilla_option(vdf, pdf, fakevolid, 'call', False, delta=25,
                                   date=start_date)
    assert isinstance(e3.exception, IndexError)


def test_create_straddle():
    pass


def test_create_strangle():
    pass


def test_create_butterfly():
    pass


def test_create_spread():
    pass


def test_create_skew():
    pass


def comp_portfolio(refresh=False):
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


def test_transfer_dict():
    # Test to ensure that modifying the output does not modify the input dict.
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)

    # check the edge cases.
    newstrad = create_straddle('CC  Z7.Z7', vdf, pdf, pd.to_datetime(start_date),
                               False, 'atm', greek='theta', greekval=5000)

    dic = pf_simple.OTC
    ret = transfer_dict(dic)
    ret.pop('CC')
    assert len(ret) == 0
    assert len(dic) == 1 and 'CC' in dic and 'Z7' in dic['CC']
    assert len(pf_simple.OTC) == 1 and 'CC' in dic and 'Z7' in dic['CC']
    assert pf_simple.OTC == dic

    ret = transfer_dict(dic)
    # modify the initial and see how it plays out.
    for op in newstrad:
        ret['CC']['Z7'][0].add(op)
        ret['CC']['Z7'][2] += op.delta
        ret['CC']['Z7'][3] += op.gamma
        ret['CC']['Z7'][4] += op.theta
        ret['CC']['Z7'][5] += op.vega
    assert len(ret['CC']['Z7'][0]) == 4
    assert len(dic['CC']['Z7'][0]) == 2


def test_merge_dicts_updates():
    # Tests to ensure that modifying the result of merge_dicts
    # does not modify the constituent dictionaries.

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initializing the portfolios and options
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    pf1 = Portfolio(cc_hedges_c, name='cc_1')
    pf2 = Portfolio(cc_hedges_c, name='cc_2')

    straddle1 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle2 = create_straddle('CC  H8.H8', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle3 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                False, 'atm', greek='theta', greekval=2500)

    pf1.add_security(straddle1, 'OTC')
    pf1.add_security(straddle3, 'hedge')

    pf2.add_security(straddle2, 'OTC')

    # refresh after adding securities.
    pf1.refresh()
    pf2.refresh()

    # merge the dictionaries.
    newhedge = {}
    newOTC = {}
    for x in [pf1, pf2]:
        # print('----- merging OTCs ------')
        newOTC = merge_dicts(x.OTC, newOTC)
        # print('-------------------------')
        # print('----- merging hedges ------')
        newhedge = merge_dicts(x.hedges, newhedge)
        # print('----------------------------')
    # check to ensure it is all right.
    assert 'Z7' in newOTC['CC']
    assert 'H8' in newOTC['CC']
    assert 'Z7' in newhedge['CC']
    assert 'H8' not in newhedge['CC']

    # print('pf2.hedges before ft: ', pf2.hedges)
    # print('pf1.hedges before ft: ', pf1.hedges)
    # print('newhedge before ft: ', newhedge)

    assert newhedge is not pf1.hedges

    # add in a future to the new hedge dictionary.
    cc_delta = int(pf2.net_greeks['CC']['H8'][0])
    cc_shorted = False if cc_delta < 0 else True

    ccft, _ = create_underlying(
        'CC', 'H8', r_pdf, date, lots=abs(cc_delta), shorted=cc_shorted)

    newhedge['CC']['H8'] = [set(), set([ccft]), cc_delta, 0, 0, 0]
    # print('newhedge: ', newhedge)
    # print('pf2.hedges after ft: ', pf2.hedges)
    # print('pf1.hedges after ft: ', pf1.hedges)

    # check to ensure underlying dictionaries are not modified by modifying
    # the result of merge_dicts.
    assert 'Z7' in pf1.hedges['CC']
    assert 'H8' not in pf1.hedges['CC']
    assert 'CC' not in pf2.hedges


def test_merge_dicts_with_hedges():
    # Tests to ensure that modifying the result of merge_dicts
    # does not modify the constituent dictionaries.

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initializing the portfolios and options
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    pf1 = Portfolio(cc_hedges_c, name='cc_1')
    pf2 = Portfolio(cc_hedges_c, name='cc_2')

    straddle1 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle2 = create_straddle('CC  H8.H8', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle3 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                False, 'atm', greek='theta', greekval=2500)

    pf1.add_security(straddle1, 'OTC')
    pf1.add_security(straddle3, 'hedge')

    pf2.add_security(straddle2, 'OTC')

    # refresh after adding securities.
    pf1.refresh()
    pf2.refresh()

    # merge the dictionaries.
    newhedge = {}
    newOTC = {}
    for x in [pf1, pf2]:
        # print('----- merging OTCs ------')
        newOTC = merge_dicts(x.OTC, newOTC)
        # print('-------------------------')
        # print('----- merging hedges ------')
        newhedge = merge_dicts(x.hedges, newhedge)
        # print('----------------------------')
    # check to ensure it is all right.
    assert 'Z7' in newOTC['CC']
    assert 'H8' in newOTC['CC']
    assert 'Z7' in newhedge['CC']
    assert 'H8' not in newhedge['CC']

    # print('pf2.hedges before ft: ', pf2.hedges)
    # print('pf1.hedges before ft: ', pf1.hedges)
    # print('newhedge before ft: ', newhedge)

    ccdelta2 = int(pf1.net_greeks['CC']['Z7'][0])
    shorted = False if ccdelta2 < 0 else True
    ccft2, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=abs(ccdelta2), shorted=shorted)

    newhedge['CC']['Z7'][1].add(ccft2)
    newhedge['CC']['Z7'][2] += ccft2.get_delta()

    # print('newhedge: ', newhedge)
    # print('pf2.hedges after ft: ', pf2.hedges)
    # print('pf1.hedges after ft: ', pf1.hedges)

    assert ccft2 not in pf1.hedges['CC']['Z7'][1]
    assert 'CC' not in pf2.hedges


def test_merge_dicts_OTC():
    # same as above, just add to OTC dictionary this time.
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initializing the portfolios and options
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    pf1 = Portfolio(cc_hedges_c, name='cc_1')
    pf2 = Portfolio(cc_hedges_c, name='cc_2')

    straddle1 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle2 = create_straddle('CC  H8.H8', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    straddle3 = create_straddle('CC  Z7.Z7', r_vdf, r_pdf, date,
                                False, 'atm', greek='theta', greekval=2500)

    pf1.add_security(straddle1, 'OTC')
    pf1.add_security(straddle3, 'hedge')

    pf2.add_security(straddle2, 'OTC')

    # refresh after adding securities.
    pf1.refresh()
    pf2.refresh()

    # merge the dictionaries.
    newhedge = {}
    newOTC = {}
    for x in [pf1, pf2]:
        # print('----- merging OTCs ------')
        newOTC = merge_dicts(x.OTC, newOTC)
        # print('-------------------------')
        # print('----- merging hedges ------')
        newhedge = merge_dicts(x.hedges, newhedge)
        # print('----------------------------')
    # check to ensure it is all right.
    assert 'Z7' in newOTC['CC']
    assert 'H8' in newOTC['CC']
    assert 'Z7' in newhedge['CC']
    assert 'H8' not in newhedge['CC']

    # print('pf2.OTC before ft: ', pf2.OTC)
    # print('pf1.OTC before ft: ', pf1.OTC)
    # print('newOTC before ft: ', newOTC)

    ccdelta2 = int(pf1.net_greeks['CC']['Z7'][0])
    shorted = False if ccdelta2 < 0 else True
    ccft2, _ = create_underlying(
        'CC', 'Z7', r_pdf, date, lots=abs(ccdelta2), shorted=shorted)

    newOTC['CC']['Z7'][1].add(ccft2)
    newOTC['CC']['Z7'][2] += ccft2.get_delta()

    # print('newhedge: ', newOTC)
    # print('pf2.hedges after ft: ', pf2.hedges)
    # print('pf1.OTC after ft: ', pf1.OTC)

    assert ccft2 not in pf1.OTC['CC']['Z7'][1]
    assert 'Z7' not in pf2.OTC['CC']


def test_merge_lists():

    # creation info
    date = pdf.value_date.min()
    r_vdf = vdf[vdf.value_date == date]
    r_pdf = pdf[pdf.value_date == date]

    # initializing the portfolios and options
    cc_hedges_c = {'delta': [['roll', 50, 1, (-10, 10)]],
                   'theta': [['bound', (-11000, -9000), 1, 'straddle',
                              'strike', 'atm', 'uid']]}

    # create one portfolio, add a CC H8.H8 straddle
    pf1 = Portfolio(cc_hedges_c, name='cc_1')
    straddle2 = create_straddle('CC  H8.H8', r_vdf, r_pdf, date,
                                True, 'atm', greek='theta', greekval=5000)
    pf1.add_security(straddle2, 'OTC')

    # create a second portfolio, add a CC  H8.H8 future that would zero out deltas of the
    # above straddle.
    pf2 = Portfolio(cc_hedges_c, name='cc_2')
    cc_delta = int(pf1.net_greeks['CC']['H8'][0])
    cc_shorted = False if cc_delta < 0 else True
    ccft, _ = create_underlying(
        'CC', 'H8', r_pdf, date, lots=abs(cc_delta), shorted=cc_shorted)
    pf2.add_security([ccft], 'OTC')

    # get the first list, sanity check
    l1 = pf1.OTC['CC']['H8']
    assert len(l1[0]) == 2
    assert len(l1[1]) == 0
    # second list, sanity check
    l2 = pf2.OTC['CC']['H8']
    assert len(l2[0]) == 0
    assert len(l2[1]) == 1
    assert l2[2] == ccft.get_delta()

    # merge the two lists.
    ret = merge_lists(l1, l2)
    # print(ret)
    # sanity check
    assert len(ret[0]) == 2
    assert len(ret[1]) == 1

    assert abs(ret[2]) < 0.5

    # modify
    newccft, _ = create_underlying(
        'CC', 'H8', r_pdf, date, lots=abs(cc_delta), shorted=cc_shorted)
    ret[1].add(newccft)
    ret[2] += newccft.get_delta()
    assert len(ret[1]) == 2
    assert len(ret[0]) == 2
    assert abs(ret[2]) < 71 and abs(ret[2]) > 69

    # check that modifying should not change constituents.
    assert len(l1[0]) == 2
    assert len(l1[1]) == 0
    # second list, sanity check
    assert len(l2[0]) == 0
    assert len(l2[1]) == 1
    assert l2[2] == ccft.get_delta()


def test_combine_portfolios_basic():

    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    pf1 = Portfolio(None)
    pf1.add_security([qc1, qc2], 'OTC')

    pf2 = Portfolio(None)
    pf2.add_security([cc1, cc2], 'OTC')

    pf3 = combine_portfolios([pf1, pf2], refresh=True)

    # OTC check
    otc = pf1.OTC.copy()
    otc.update(pf2.OTC.copy())
    for pdt in otc:
        for mth in otc[pdt]:
            assert pf3.OTC[pdt][mth] == otc[pdt][mth]

    for pdt in pf2.OTC:
        for mth in pf2.OTC[pdt]:
            assert pf3.OTC[pdt][mth] == pf2.OTC[pdt][mth]

    # hedge check
    hed = pf1.hedges.copy()
    hed.update(pf2.hedges.copy())
    for pdt in hed:
        for mth in hed[pdt]:
            assert pf3.hedges[pdt][mth] == hed[pdt][mth]
    for pdt in pf2.hedges:
        for mth in pf2.hedges[pdt]:
            assert pf3.hedges[pdt][mth] == pf2.hedges[pdt][mth]

    # net check
    net_tst = pf1.net_greeks.copy()
    net_tst.update(pf2.net_greeks)
    for pdt in net_tst:
        for mth in net_tst[pdt]:
            assert net_tst[pdt][mth] == pf3.net_greeks[pdt][mth]

    for pdt in pf2.net_greeks:
        for mth in pf2.net_greeks[pdt]:
            assert pf3.net_greeks[pdt][mth] == pf2.net_greeks[pdt][mth]

    # lists check
    otcops = pf1.OTC_options.copy()
    otcops.extend(pf2.OTC_options)
    assert otcops == pf3.OTC_options

    hops = pf1.hedge_options.copy()
    hops.extend(pf2.hedge_options)
    assert hops == pf3.hedge_options

    assert pf3.families == [pf1, pf2]
    assert isinstance(pf3, Portfolio)


def test_combine_portfolios_containment():

    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    pf1 = Portfolio(None)
    pf1.add_security([qc1, qc2], 'OTC')

    pf2 = Portfolio(None)
    pf2.add_security([cc1, cc2], 'OTC')

    pf3 = combine_portfolios([pf1, pf2], refresh=True)

    # create CC future
    cc_delta = pf3.net_greeks['CC']['U7'][0]
    cc_shorted = True if cc_delta > 0 else False
    ccft, _ = create_underlying('CC', 'U7', pdf, pd.to_datetime(
        start_date), shorted=cc_shorted, lots=abs(cc_delta))

    # create QC future
    qc_delta = pf3.net_greeks['QC']['U7'][0]
    qc_shorted = True if cc_delta > 0 else False
    qcft, _ = create_underlying('QC', 'U7', pdf, pd.to_datetime(
        start_date), shorted=qc_shorted, lots=abs(qc_delta))

    # add both futures to super-portfolio
    pf3.add_security([qcft, ccft], 'OTC')
    # print('pf2: ', pf2.OTC)
    # print('pf3.net_greeks post ft pre refresh: ', pf3.net_greeks)
    # print('------------ refresh call -------------')
    pf3.refresh()
    # print('------------ end refresh call -------------')
    # print('pf3.net_greeks post ft post refresh: ', pf3.net_greeks)

    # check that futures are not sent into the constituent families.
    try:
        assert ccft not in pf1.OTC_options
        assert len(pf1.OTC['QC']['U7'][1]) == 0
        assert len(pf2.OTC['CC']['U7'][1]) == 0
        # assert ccft not in pf1.OTC_options
    except AssertionError as e:
        # print('pf1: ', pf1.OTC)
        # print('pf2: ', pf2.OTC)
        raise AssertionError("pf1, pf2: ", pf1.OTC, pf2.OTC) from e

    # check that the net greeks have been updated post refresh.
    try:
        assert pf3.net_greeks['CC']['U7'][0] == 0
        assert pf3.net_greeks['QC']['U7'][0] == 0
    except AssertionError as e:
        raise AssertionError("pf1, pf2, pf3: ", pf1.net_greeks,
                             pf2.net_greeks, pf3.net_greeks)


def test_combine_portfolios_lists_vs_sets():

    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    pf1 = Portfolio(None)
    pf1.add_security([qc1, qc2], 'OTC')

    pf2 = Portfolio(None)
    pf2.add_security([cc1, cc2], 'OTC')

    pf3 = combine_portfolios([pf1, pf2], refresh=True)

    ccs = [x for x in pf3.OTC_options if x.get_product() == 'CC']
    qcs = [x for x in pf3.OTC_options if x.get_product() == 'QC']

    for x in ccs:
        assert x in pf3.OTC['CC']['U7'][0]

    for x in qcs:
        assert x in pf3.OTC['QC']['U7'][0]


def test_combine_portfolios_check_families():
    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    pf1 = Portfolio(None)
    pf1.add_security([qc1, qc2], 'OTC')

    pf2 = Portfolio(None)
    pf2.add_security([cc1, cc2], 'OTC')

    pf3 = combine_portfolios([pf1, pf2], refresh=True)
    assert pf3.families == [pf1, pf2]


def test_merge_dicts_list_case():
    d1 = {'Z7': [set(), set(), 0, 0, 0, 0]}
    d2 = {'H8': [set(), set(), 0, 0, 0, 0]}
    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    d1['Z7'][0].add(cc1)
    d1['Z7'][0].add(cc2)

    for x in [cc1, cc2]:
        d1['Z7'][2] += x.delta
        d1['Z7'][3] += x.gamma
        d1['Z7'][4] += x.theta
        d1['Z7'][5] += x.vega

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    d2['H8'][0].add(qc1)
    d2['H8'][0].add(qc2)

    for x in [qc1, qc2]:
        d2['H8'][2] += x.delta
        d2['H8'][3] += x.gamma
        d2['H8'][4] += x.theta
        d2['H8'][5] += x.vega

    # print('d1: ', d1)
    # print('d2: ', d2)

    # merge_dict list case.
    ret = merge_dicts(d1, d2)

    # print('ret: ', ret)

    # create a cc_future
    ccft, _ = create_underlying('CC', 'U7', pdf, pd.to_datetime(
        start_date), shorted=True, lots=100)

    # isolate the Z7 entry.
    ret_cc = ret['Z7']
    ret_qc = ret['H8']

    # print('ret_cc: ', ret_cc)
    # print('ret_qc: ', ret_qc)
    # add the future into ret. assert that the initial is unchanged
    ret_cc[1].add(ccft)
    assert len(ret_cc[1]) == 1
    try:
        assert len(d1['Z7'][1]) == 0
    except AssertionError:
        raise AssertionError('actual: ', [str(x) for x in d1['Z7'][1]])


def test_merge_dicts_edge_case():
    d1 = {'Z7': [set(), set(), 0, 0, 0, 0]}
    d2 = {'H8': [set(), set(), 0, 0, 0, 0]}
    cc1, cc2 = create_straddle('CC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), False, 'atm', greek='vega', greekval=20000)

    d1['Z7'][0].add(cc1)
    d1['Z7'][0].add(cc2)

    for x in [cc1, cc2]:
        d1['Z7'][2] += x.delta
        d1['Z7'][3] += x.gamma
        d1['Z7'][4] += x.theta
        d1['Z7'][5] += x.vega

    qc1, qc2 = create_straddle('QC  U7.U7', vdf, pdf, pd.to_datetime(
        start_date), True, 'atm', greek='vega', greekval=20000)

    d2['H8'][0].add(qc1)
    d2['H8'][0].add(qc2)

    for x in [qc1, qc2]:
        d2['H8'][2] += x.delta
        d2['H8'][3] += x.gamma
        d2['H8'][4] += x.theta
        d2['H8'][5] += x.vega

    # print('d1: ', d1)
    # print('d2: ', d2)

    # merge_dict list case.
    ret = merge_dicts(d1, d2)

    # print('ret: ', ret)

    # isolate the Z7 entry.
    ret_cc = ret['Z7']
    ret_qc = ret['H8']

    # print('ret_cc: ', ret_cc)
    # print('ret_qc: ', ret_qc)
    # remove first option from new, assert old is unchanged.
    first = next(iter(ret_cc[0]))
    # print('first: ', first)
    ret_cc[0].remove(first)

    # print('')

    assert len(ret_cc[0]) == 1
    try:
        assert len(d1['Z7'][0]) == 2
    except AssertionError:
        raise AssertionError('actual: ', [str(x) for x in d1['Z7'][0]])


def test_assign_hedge_objects():
    pf_simple, pf_comp, ccops, qcops, pfcc, pfqc = comp_portfolio(refresh=True)
    # test simple case.
    pf = copy.deepcopy(pf_simple)
    pf = assign_hedge_objects(pf)
    assert pf.hedger is not None
    assert pf.hedger.vdf is None
    assert pf.hedger.pdf is None

    # complex case
    pf = copy.deepcopy(pf_comp)
    pf = assign_hedge_objects(pf)
    assert pf.hedger is not None
    assert pf.hedger.vdf is None
    assert pf.hedger.pdf is None
    for fa in pf.get_families():
        assert fa.hedger is not None
        assert fa.hedger.pdf is None
        assert fa.hedger.vdf is None
