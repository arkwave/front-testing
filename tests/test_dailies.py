# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2018-07-17 20:09:38
# @Last Modified by:   arkwave
# @Last Modified time: 2018-07-17 22:00:50
import sys
sys.path.append('../')
from scripts.fetch_data import grab_data
from scripts.util import create_vanilla_option, create_barrier_option

####################################################
# default variables
start = '2018-07-16'
end = '2018-07-17'
pdt = ['KC']
# pull the data used to initialize option objects. 
vdf, pdf, _ = grab_data(pdt, start, end)
volid = 'KC  Z8.Z8'

strike = 120
up_bar = 130
down_bar = 110
####################################################


def create_test_suite():
    call = create_vanilla_option(vdf, pdf, volid, 'call', False, lots=1, 
                                 bullet=False, strike=strike)
    put = create_vanilla_option(vdf, pdf, volid, 'put', False, lots=1, 
                                bullet=False, strike=strike)
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
    ops = {'call': call, 'put': put, 'ecuo': ecuo, 'ecui': ecui, 'epdo': epdo, 
           'epdi': epdi, 'cuo': cuo, 'cui': cui, 'cdo': cdo, 'cdi': cdi, 
           'puo': puo, 'pui': pui, 'pdo': pdo, 'pdi': pdi}
    return ops 


def test_euro_knockin():
    # ECUI test. 
    ecui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'euro', 'up', bullet=False, ki=up_bar, 
                                 lots=1)
    # EPDI test. 
    epdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'euro', 'down', bullet=False, ki=down_bar, 
                                 lots=1)

    pass 


def test_euro_knockout():
    ecuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'euro', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    epdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'euro', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    pass 


def test_amer_knockin():
    cui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'up', bullet=False, ki=up_bar, 
                                 lots=1)
    pui = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'up', bullet=False, ki=up_bar, 
                                 lots=1)
    cdi = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ki=down_bar, 
                                 lots=1)
    pdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'down', bullet=False, ki=down_bar, 
                                 lots=1)
    pass 


def test_amer_knockout():
    cuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    puo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    pdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    pass 


def test_remove_expired():
    pass 
