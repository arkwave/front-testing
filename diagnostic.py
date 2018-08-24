import numpy as np
import pandas as pd
from scripts.fetch_data import grab_data
from scripts.util import create_barrier_option, create_vanilla_option, create_underlying, hedge_all_deltas, create_straddle, assign_hedge_objects
from scripts.portfolio import Portfolio
from scripts.simulation import run_simulation
from collections import OrderedDict 
from scripts.prep_data import handle_dailies
from sqlalchemy import create_engine
# seed = 7
# np.random.seed(seed)


pdts = ['KC']
# contract = ['Z8']
start = '2018-05-30'
end = '2018-06-08'
volid = 'KC  Z8.Z8'
hedge_vid = 'KC  N8.N8'
vdf, pdf, edf = grab_data(pdts, start, end)
up_bar = 125
down_bar = 115
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

ops = [ecuo, ecui, epdo, epdi, cuo, cui, cdo, cdi, puo, pui, pdo, pdi]

print('============================')
print('future price: ', ecuo.get_underlying().get_price())
print('strike vol: ', ecuo.vol)
print('Upper barrier vol: ', ecuo.bvol)
print('Upper digital: ', ecuo.bvol2)
print('Lower barrier vol: ', epdi.bvol)
print('Lower digital: ', epdi.bvol2)
print('num ops: ', len(ecuo.get_ttms()))
print('============================')


# for op in ops:
#     print('-------------------------------------------------')
#     print(op)
#     print('market: ', op.get_price())
#     delta, gamma, theta, vega = op.greeks()
#     print('num_ops: ', len(op.get_ttms()))
#     print('delta: ', delta)
#     print('gamma: ', gamma)
#     print('theta: ', theta)
#     print('vega: ', vega)


# # specify the hedging parameters
# gen_hedges = OrderedDict({'delta': [['static', 0, 1]]})
hedge_dict_1 = OrderedDict({'delta': [['static', 0, 1]],
                            'vega':  [['bound', (3800, 4200), 1, 'straddle', 
                                       'strike', 'atm', 'uid']]})

pf = Portfolio(hedge_dict_1, name='test', roll=True, ttm_tol=5)
pf.add_security([ecuo], 'OTC')

# zeroing deltas
pf = hedge_all_deltas(pf, pdf)

pf = assign_hedge_objects(pf, vdf=vdf, pdf=pdf, book=False, auto_volid=True)

vega = pf.get_net_greeks()['KC']['Z8'][-1]
shorted = True if vega > 0 else False

# create the hedge options 
hedge_ops = create_straddle(hedge_vid, vdf, pdf, pd.to_datetime(start), 
                            shorted, strike='atm', greek='vega', greekval=round(vega))
pf.add_security(hedge_ops, 'hedge')

print(pf)

engine = pf.get_hedger()
# print('relevant prices: ', pdf[pdf.underlying_id.isin(pf.get_unique_uids())])
# results = run_simulation(vdf, pdf, pf, plot_results=False, slippage=2, 
#                          roll_hedges_only=True, same_month_exception=True)
