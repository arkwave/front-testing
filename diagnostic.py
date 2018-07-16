import numpy as np
import pandas as pd
from scripts.fetch_data import grab_data
from scripts.util import create_barrier_option, create_vanilla_option, create_underlying
from scripts.portfolio import Portfolio
from scripts.simulation import run_simulation
from collections import OrderedDict 
from scripts.prep_data import handle_dailies

seed = 7
np.random.seed(seed)


pdts = ['KC']
contract = ['Z8']
start = '2018-07-13'
end = '2018-07-16'
vdf, pdf, edf = grab_data(pdts, start, end)

ecuo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'euro', 'up', bullet=False, ko=130, 
                             lots=1)
ecui = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'euro', 'up', bullet=False, ki=130, 
                             lots=1)
epdo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'euro', 'down', bullet=False, ko=115, 
                             lots=1)
epdi = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'euro', 'down', bullet=False, ki=115, 
                             lots=1)
cuo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'amer', 'up', bullet=False, ko=130, 
                             lots=1)
cui = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'amer', 'up', bullet=False, ki=130, 
                             lots=1)
cdo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'amer', 'down', bullet=False, ko=115, 
                             lots=1)
cdi = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'call', 120, False, 
                             'amer', 'down', bullet=False, ki=115, 
                             lots=1)
puo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'amer', 'up', bullet=False, ko=130, 
                             lots=1)
pui = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'amer', 'up', bullet=False, ki=130, 
                             lots=1)
pdo = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'amer', 'down', bullet=False, ko=115, 
                             lots=1)
pdi = create_barrier_option(vdf, pdf, 'KC  Z8.Z8', 'put', 120, False, 
                             'amer', 'down', bullet=False, ki=115, 
                             lots=1)

ops = [ecuo, ecui, epdo, epdi, cuo, cui, cdo, cdi, puo, pui, pdo, pdi]


print('============================')
print('future price: ', ecuo.get_underlying().get_price())
print('strike vol: ', ecuo.vol)
print('Upper barrier vol: ', ecuo.bvol)
print('Upper digital: ', ecuo.bvol2)
print('Lower barrier vol: ', epdi.bvol)
print('Lower digital: ', epdi.bvol2)
print('============================')


# ops = [ecuo]

# print('initial ttms: ', np.array(ecuo.get_ttms()) * 365)
# ecuo.update_tau(1/365)
# print('final ttms: ', np.array(ecuo.get_ttms()) * 365)

for op in ops:
    # print('init ttms:')
    # print(np.array(op.get_ttms()) * 365)
    op.update_tau(1/365)
    op.update()
    # print('final ttms:')
    # print(np.array(op.get_ttms()) * 365)
    print('-'*30)
    print(op)
    print('num ops: ', len([x for x in op.get_ttms() if not np.isclose(x, 0)]))
    print('FV: ', op.get_price())
    print('delta: ', op.delta)
    print('gamma: ', op.gamma)
    print('theta: ', op.theta)
    print('vega: ', op.vega)
    print('----------------------')

# print('-'*30)
# print("t = 0")
# print('num ops: ', len(op))
# print('FV: ', sum([o.get_price() for o in op])/len(op))
# print('delta: ', sum([o.greeks()[0] for o in op]))
# print('gamma: ', sum([o.greeks()[1] for o in op]))
# print('theta: ', sum([o.greeks()[2] for o in op]))
# print('vega: ', sum([o.greeks()[3] for o in op]))
# print('-'*30)

# specify the hedging parameters
# gen_hedges = OrderedDict({'delta': [['static', 0, 1]]})
# hedge_dict_1 = OrderedDict({'gamma': [['bound', (3800, 4200), 1, 'straddle', 
#                                        'strike', 'atm', 'uid']]})
# pf = Portfolio(hedge_dict_1, name='test')
# pf.add_security(op, 'OTC')

# # zeroing deltas
# dic = pf.get_net_greeks()
# hedge_fts = []
# for pdt in dic:
#     for mth in dic[pdt]:
#         # get the delta value. 
#         delta = round(dic[pdt][mth][0])
#         shorted = True if delta > 0 else False
#         delta = abs(delta) 
#         ft, ftprice = create_underlying(pdt, mth, pdf, shorted=shorted, lots=delta)
#         hedge_fts.append(ft)

# if hedge_fts:
#     pf.add_security(hedge_fts, 'hedge')

# print(pf)

# results = run_simulation(vdf, pdf, pf, flat_vols=True, plot_results=False)
# plot_output(results[0])
