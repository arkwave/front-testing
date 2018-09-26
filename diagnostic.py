import numpy as np
import pandas as pd
from scripts.fetch_data import grab_data
from scripts.util import create_barrier_option, create_vanilla_option, create_underlying, hedge_all_deltas, create_straddle, assign_hedge_objects
from scripts.portfolio import Portfolio
from scripts.simulation import run_simulation
from collections import OrderedDict 
from scripts.prep_data import handle_dailies
from sqlalchemy import create_engine
import time
# seed = 7
# np.random.seed(seed)


pdts = ['KC']
# contract = ['Z8']
start = '2018-05-30'
end = '2018-06-08'
volid = 'KC  Z8.Z8'
hedge_vid = 'KC  N8.N8'
vdf, pdf, edf = grab_data(pdts, start, end)
up_bar = 145
down_bar = 95
strike = 120


# pdf['time'] = pd.to_datetime(pdf['time']).dt.time
# vdf['time'] = pd.to_datetime(vdf['time']).dt.time

ecuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'euro', 'up', bullet=False, ko=up_bar, 
                             lots=1)
ecui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'euro', 'up', bullet=False, ki=up_bar, 
                             lots=1)
epdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'euro', 'down', bullet=False, ko=down_bar, 
                             lots=1)
epdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'euro', 'down', bullet=False, ki=down_bar, 
                             lots=1)
cuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'amer', 'up', bullet=False, ko=up_bar, 
                             lots=1)
cui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'amer', 'up', bullet=False, ki=up_bar, 
                             lots=1)
cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'amer', 'down', bullet=False, ko=down_bar, 
                             lots=1)
cdi = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                             'amer', 'down', bullet=False, ki=down_bar, 
                             lots=1)
puo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'amer', 'up', bullet=False, ko=up_bar, 
                             lots=1)
pui = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'amer', 'up', bullet=False, ki=up_bar, 
                             lots=1)
pdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'amer', 'down', bullet=False, ko=down_bar, 
                             lots=1)
pdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                             'amer', 'down', bullet=False, ki=down_bar, 
                             lots=1)

dc = create_vanilla_option(vdf, pdf, volid, 'call', False, lots=1, strike='atm', bullet=False)

ops = [ecuo, ecui, epdo, epdi, cuo, cui, cdo, cdi, puo, pui, pdo, pdi, dc]


for op in ops:
    print('--------------------')
    print('Processing %s \n' % op)
    t = time.clock() 
    newval = op.compute_price()
    dt = time.clock() - t
    print('Pricing: ', dt)
    t2 = time.clock()
    x = op.update_greeks(vol=op.vol, bvol=op.bvol, bvol2=op.bvol2)
    dt2 = time.clock() - t2 
    print('Greeks: ', dt2)
    print('--------------------')


# print('============================')
# print('future price: ', ecuo.get_underlying().get_price())
# print('strike vol: ', ecuo.vol)
# print('Upper barrier vol: ', ecuo.bvol)
# print('Upper digital: ', ecuo.bvol2)
# print('Lower barrier vol: ', epdi.bvol)
# print('Lower digital: ', epdi.bvol2)
# print('num ops: ', len(ecuo.get_ttms()))
# print('============================')


# # # specify the hedging parameters
# # gen_hedges = OrderedDict({'delta': [['static', 0, 1]]})
# hedge_dict_1 = OrderedDict({'delta': [['static', 0, 1]],
#                             'theta':  [['bound', (-100, 100), 1, 'straddle', 
#                                         'strike', 'atm', 'agg']]})

# vid_dict = {'theta': {('KC', 'Z8'): hedge_vid}}
# pf = Portfolio(hedge_dict_1, name='test', roll=True, ttm_tol=5)

# # add security to be hedged. 
# pf.add_security([dc], 'OTC')

# # assign hedge objects. 
# pf = assign_hedge_objects(pf, vdf=vdf, pdf=pdf, book=False, auto_volid=False, vid_dict=vid_dict)

# print('auto_volid: ', pf.get_hedger().auto_detect_volids())

# # hedge the greeks passed into vid_dict 
# for flag in vid_dict:
#     pf.get_hedger().apply(flag)

# # hedge all deltas introduced by hedging other greeks
# pf = hedge_all_deltas(pf, pdf)
# assert pf.get_hedger().satisfied()
# print(pf.get_aggregated_greeks())

# # print('pf: ', pf)
# print('auto_volid: ', pf.get_hedger().auto_detect_volids())

# # # print('relevant prices: ', pdf[pdf.underlying_id.isin(pf.get_unique_uids())])
# results = run_simulation(vdf, pdf, pf, plot_results=False, slippage=2, 
#                          roll_hedges_only=True, same_month_exception=True)

