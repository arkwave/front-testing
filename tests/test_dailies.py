# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2018-07-17 20:09:38
# @Last Modified by:   arkwave
# @Last Modified time: 2018-07-23 19:08:00
import sys
sys.path.append('../')
import numpy as np 
from scripts.fetch_data import grab_data
from copy import deepcopy
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
    #### ECUI test ####
    ecui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'euro', 'up', bullet=False, ki=up_bar, 
                                 lots=1)

    # option should be active, not expired, OTM, not knockedin. 
    assert ecui.check_active()
    assert not ecui.check_expired()
    assert ecui.moneyness() == -1
    assert not ecui.knockedin
    initial_greeks = ecui.greeks()

    # update the future price to 125; option should now be OTM, active and not expired. 
    ecui.get_underlying().update_price(125)
    ecui.update()
    assert ecui.check_active()
    assert not ecui.check_expired()
    assert ecui.moneyness() == -1
    assert not ecui.knockedin
    assert not np.array_equal(ecui.greeks(), initial_greeks)

    # update the future price to 130; option should now be OTM, active and not expired. 
    ecui.get_underlying().update_price(130)
    ecui.update()
    assert ecui.check_active()
    assert not ecui.check_expired()
    assert ecui.knockedin
    assert ecui.moneyness() == 1
    assert ecui.exercise()
    assert not np.array_equal(ecui.greeks(), initial_greeks)

    # update the future price to 135; option should now be ITM, active and not expired. 
    ecui.get_underlying().update_price(135)
    ecui.update()
    assert ecui.check_active()
    assert not ecui.check_expired()
    assert ecui.knockedin
    assert ecui.moneyness() == 1
    assert ecui.exercise()
    assert not np.array_equal(ecui.greeks(), initial_greeks)

    #### EPDI test ####
    epdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'euro', 'down', bullet=False, ki=down_bar, 
                                 lots=1)
    assert epdi.check_active()
    assert not epdi.check_expired()
    assert epdi.moneyness() == -1
    assert not epdi.knockedin
    initial_greeks = epdi.greeks()

    # update the future price to 116; option should now be OTM, active and not expired. 
    epdi.get_underlying().update_price(116)
    epdi.update()
    assert epdi.check_active()
    assert not epdi.check_expired()
    assert epdi.moneyness() == -1
    assert not epdi.exercise()
    assert not epdi.knockedin
    assert not np.array_equal(epdi.greeks(), initial_greeks)

    # update the future price to 110; option should now be OTM, active and not expired. 
    epdi.get_underlying().update_price(110)
    epdi.update()
    assert epdi.check_active()
    assert not epdi.check_expired()
    assert epdi.knockedin
    assert epdi.moneyness() == 1
    assert epdi.exercise()
    assert not np.array_equal(epdi.greeks(), initial_greeks)

    # update the future price to 105; option should now be ITM, active and not expired. 
    epdi.get_underlying().update_price(105)
    epdi.update()
    assert epdi.check_active()
    assert not epdi.check_expired()
    assert epdi.knockedin
    assert epdi.moneyness() == 1
    assert not np.array_equal(epdi.greeks(), initial_greeks)


def test_euro_knockout():
    
    #### ECUO Test ####
    ecuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'euro', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    # option should be active, not expired, OTM, not knockedout. 
    assert ecuo.check_active()
    assert not ecuo.check_expired()
    assert ecuo.moneyness() == -1
    assert not ecuo.knockedout
    initial_greeks = ecuo.greeks()

    # update the future price to 125; option should now be ITM, active and not expired. 
    ecuo.get_underlying().update_price(125)
    ecuo.update()
    assert ecuo.check_active()
    assert not ecuo.check_expired()
    assert ecuo.moneyness() == 1
    assert not ecuo.knockedout
    assert not np.array_equal(ecuo.greeks(), initial_greeks)

    # update the future price to 130; option should now be OTM, active, not expired and KOed.
    ecuo.get_underlying().update_price(130)
    ecuo.update()
    assert ecuo.check_active()
    assert not ecuo.check_expired()
    assert ecuo.moneyness() == -1
    assert not ecuo.knockedin
    assert not np.array_equal(ecuo.greeks(), initial_greeks)
    assert not np.array_equal(ecuo.greeks(), [0,0,0,0])
    assert ecuo.knockedout

    # update the future price to 135; option should now be OTM, active, not expired and KOed.
    ecuo.get_underlying().update_price(135)
    ecuo.update()
    assert ecuo.check_active()
    assert not ecuo.check_expired()
    assert ecuo.moneyness() == -1
    assert not ecuo.knockedin
    assert not np.array_equal(ecuo.greeks(), initial_greeks)
    assert ecuo.knockedout

    #### Test EPDO ####
    epdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'euro', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    # option should be active, not expired, ITM, not knockedout 
    assert epdo.check_active()
    assert not epdo.check_expired()
    assert epdo.moneyness() == 1
    assert not epdo.knockedout
    initial_greeks = epdo.greeks()

    # update the future price to 130; option should now be OTM, active, not expired
    epdo.get_underlying().update_price(130)
    epdo.update()
    assert epdo.check_active()
    assert not epdo.check_expired()
    assert epdo.moneyness() == -1
    assert not epdo.knockedout
    assert not np.array_equal(epdo.greeks(), initial_greeks)

    # update the future price to 110; option should now be OTM, active, not expired and KOed.
    epdo.get_underlying().update_price(110)
    epdo.update()
    assert epdo.check_active()
    assert not epdo.check_expired()
    assert epdo.moneyness() == -1
    assert not np.array_equal(epdo.greeks(), initial_greeks)
    assert epdo.knockedout


def test_cui():
    #### CUI #####
    cui = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'up', bullet=False, ki=up_bar, 
                                 lots=1)
    assert cui.moneyness() == -1
    assert not cui.exercise()
    assert not cui.knockedin 
    assert cui.check_active() 
    assert not cui.check_expired()

    # bump prices to above strike but below barrier. 
    cui.get_underlying().update_price(125)
    cui.update()
    assert cui.moneyness() == -1
    assert not cui.exercise()
    assert not cui.knockedin 
    assert cui.check_active() 
    assert not cui.check_expired()

    # bump prices to barrier; should trigger KI to vanilla option. 
    cui.get_underlying().update_price(130)
    cui.update()
    assert cui.moneyness() == 1
    assert cui.exercise()
    assert cui.knockedin 
    assert cui.check_active() 
    assert not cui.check_expired()
    assert cui.barrier is None

    # create vanilla option and compare greeks/market value. 
    comp_van = create_vanilla_option(vdf, pdf, volid, 'call', False, strike=strike, 
                                     bullet=False, lots=1)
    comp_van.get_underlying().update_price(130)
    comp_van.update() 
    try:
        assert np.array_equal(comp_van.greeks(), cui.greeks())
    except AssertionError as e:
        raise AssertionError(comp_van.greeks(), cui.greeks()) from e
    assert np.isclose(comp_van.get_price(), cui.get_price())


def test_pui():
    #### PUI #####
    pui = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'up', bullet=False, ki=up_bar, 
                                 lots=1)
    assert pui.moneyness() == -1
    assert not pui.exercise()
    assert not pui.knockedin 
    assert pui.check_active() 
    assert not pui.check_expired()

    # bump prices to above strike but below barrier. 
    pui.get_underlying().update_price(125)
    pui.update()
    assert pui.moneyness() == -1
    assert not pui.exercise()
    assert not pui.knockedin 
    assert pui.check_active() 
    assert not pui.check_expired()

    # bump prices to barrier; should trigger KI to vanilla option. 
    pui.get_underlying().update_price(130)
    pui.update()
    assert pui.moneyness() == -1
    assert not pui.exercise()
    assert pui.knockedin 
    assert pui.check_active() 
    assert not pui.check_expired()
    assert pui.barrier is None

    # create vanilla option and compare greeks/market value. 
    comp_van = create_vanilla_option(vdf, pdf, volid, 'put', False, strike=strike, bullet=False, lots=1)
    comp_van.get_underlying().update_price(130)
    comp_van.update() 
    assert np.array_equal(comp_van.greeks(), pui.greeks())
    assert np.isclose(comp_van.get_price(), pui.get_price())

    # bump prices to ITM 
    pui.get_underlying().update_price(105)
    pui.update()
    assert pui.moneyness() == 1
    assert pui.exercise()
    assert pui.check_active() 
    assert not pui.check_expired()
    assert pui.barrier is None


def test_cdi():
    #### CDI #####
    cdi = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ki=down_bar, 
                                 lots=1)
    assert cdi.moneyness() == -1
    assert not cdi.exercise()
    assert not cdi.knockedin 
    assert cdi.check_active() 
    assert not cdi.check_expired()

    # bump prices to above strike but below barrier. 
    cdi.get_underlying().update_price(125)
    cdi.update()
    assert cdi.moneyness() == -1
    assert not cdi.exercise()
    assert not cdi.knockedin 
    assert cdi.check_active() 
    assert not cdi.check_expired()

    # bump prices to barrier; should trigger KI to vanilla option. 
    cdi.get_underlying().update_price(110)
    cdi.update()
    assert cdi.knockedin 
    assert cdi.moneyness() == -1
    assert not cdi.exercise()
    assert cdi.check_active() 
    assert not cdi.check_expired()
    assert cdi.barrier is None

    # create vanilla option and compare greeks/market value. 
    comp_van = create_vanilla_option(vdf, pdf, volid, 'call', False, strike=strike, bullet=False)
    comp_van.get_underlying().update_price(110)
    comp_van.update() 
    assert np.array_equal(comp_van.greeks(), cdi.greeks())
    assert np.isclose(comp_van.get_price(), cdi.get_price())

    # bump prices to ITM 
    cdi.get_underlying().update_price(125)
    cdi.update()
    assert cdi.moneyness() == 1
    assert cdi.exercise()
    assert cdi.check_active() 
    assert not cdi.check_expired()
    assert cdi.barrier is None


def test_pdi():
    #### PDI #####
    pdi = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'down', bullet=False, ki=down_bar, 
                                 lots=1)
    assert pdi.moneyness() == -1
    assert not pdi.exercise()
    assert not pdi.knockedin 
    assert pdi.check_active() 
    assert not pdi.check_expired()

    # bump prices to move OTM while not KI
    pdi.get_underlying().update_price(125)
    pdi.update()
    assert pdi.moneyness() == -1
    assert not pdi.exercise()
    assert not pdi.knockedin 
    assert pdi.check_active() 
    assert not pdi.check_expired()

    # bump prices to barrier; should trigger KI to vanilla option. 
    pdi.get_underlying().update_price(110)
    pdi.update()
    assert pdi.moneyness() == 1
    assert pdi.exercise()
    assert pdi.knockedin 
    assert pdi.check_active() 
    assert not pdi.check_expired()
    assert pdi.barrier is None

    # create vanilla option and compare greeks/market value. 
    comp_van = create_vanilla_option(vdf, pdf, volid, 'put', False, strike=strike, bullet=False)
    comp_van.get_underlying().update_price(110)
    comp_van.update() 
    assert np.array_equal(comp_van.greeks(), pdi.greeks())
    assert np.isclose(comp_van.get_price(), pdi.get_price())

    
def test_cuo():
    cuo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    assert cuo.moneyness() == -1
    assert not cuo.exercise()
    assert not cuo.knockedout 
    assert cuo.check_active() 
    assert not cuo.check_expired()

    # bump prices to move ITM while not KO
    cuo.get_underlying().update_price(125)
    cuo.update()
    assert cuo.moneyness() == 1
    assert cuo.exercise()
    assert not cuo.knockedout
    assert cuo.check_active() 
    assert not cuo.check_expired()

    # bump prices to barrier; should trigger KO 
    cuo.get_underlying().update_price(130)
    cuo.update()
    assert cuo.knockedout
    assert cuo.moneyness() == -1
    assert not cuo.exercise()
    assert not cuo.check_active() 
    assert cuo.check_expired()
    assert np.array_equal(cuo.greeks(), [0,0,0,0])


def test_cdo():
    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)

    assert cdo.moneyness() == -1
    assert not cdo.exercise()
    assert not cdo.knockedout
    assert cdo.check_active() 
    assert not cdo.check_expired()

    # bump prices to move ITM while not KO
    cdo.get_underlying().update_price(125)
    cdo.update()
    assert cdo.moneyness() == 1
    assert cdo.exercise()
    assert not cdo.knockedout
    assert cdo.check_active() 
    assert not cdo.check_expired()

    # bump prices to barrier; should trigger KO 
    cdo.get_underlying().update_price(110)
    cdo.update()
    assert cdo.knockedout
    assert cdo.moneyness() == -1
    assert not cdo.exercise()
    assert not cdo.check_active() 
    assert cdo.check_expired()
    assert np.array_equal(cdo.greeks(), [0,0,0,0])


def test_puo():
    puo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'up', bullet=False, ko=up_bar, 
                                 lots=1)
    assert puo.moneyness() == 1
    assert puo.exercise()
    assert not puo.knockedout 
    assert puo.check_active() 
    assert not puo.check_expired()

    # bump prices to move OTM while not KO
    puo.get_underlying().update_price(125)
    puo.update()
    assert puo.moneyness() == -1
    assert not puo.exercise()
    assert not puo.knockedout
    assert puo.check_active() 
    assert not puo.check_expired()

    # bump prices to barrier; should trigger KO 
    puo.get_underlying().update_price(130)
    puo.update()
    assert puo.knockedout
    assert puo.moneyness() == -1
    assert not puo.exercise()
    assert puo.check_expired()
    assert not puo.check_active() 
    assert np.array_equal(puo.greeks(), [0,0,0,0])


def test_pdo():
    pdo = create_barrier_option(vdf, pdf, volid, 'put', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    assert pdo.moneyness() == 1
    assert pdo.exercise()
    assert not pdo.knockedout
    assert pdo.check_active() 
    assert not pdo.check_expired()

    # bump prices to move OTM while not KO
    pdo.get_underlying().update_price(125)
    pdo.update()
    assert pdo.moneyness() == -1
    assert not pdo.exercise()
    assert not pdo.knockedout
    assert pdo.check_active() 
    assert not pdo.check_expired()

    # bump prices to barrier; should trigger KO 
    pdo.get_underlying().update_price(110)
    pdo.update()
    assert pdo.knockedout
    assert pdo.moneyness() == -1
    assert not pdo.exercise()
    assert not pdo.check_active() 
    assert pdo.check_expired()
    assert np.array_equal(pdo.greeks(), [0,0,0,0])


def test_remove_expired():
    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    assert all([x > 0 for x in cdo.get_ttms()])
    init_len = len(cdo.get_ttms())
    cdo.update_tau(1/365)
    assert not all([x > 0 for x in cdo.get_ttms()])
    cdo.remove_expired_dailies() 
    assert all([x > 0 for x in cdo.get_ttms()])
    assert init_len - len(cdo.get_ttms()) == 1


def test_expiry():
    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    assert not cdo.check_expired() 
    cdo.update_tau(1/365)
    cdo.update() 
    assert not cdo.check_expired()
    cdo.remove_expired_dailies() 

    cdo.update_tau(cdo.tau - 1/365)
    cdo.update() 
    assert not cdo.check_expired()
    cdo.remove_expired_dailies()
    assert len(cdo.get_ttms()) == 1

    cdo.update_tau(1/365)
    cdo.update() 
    assert not cdo.check_expired() 
    cdo.remove_expired_dailies()
    assert len(cdo.get_ttms()) == 0
    assert cdo.check_expired()


def test_reverse_timestep():
    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    init_ttms = deepcopy(cdo.get_ttms())
    # print('init_ttms: ', np.array(init_ttms)*365)

    cdo.update_tau(1/365)
    new_ttms = deepcopy(cdo.get_ttms())
    # print('new_ttms: ', np.array(new_ttms)*365)

    # check that timestep has been deceremented by 1 appropriately.
    assert all([np.isclose(init_ttms[i] - new_ttms[i], 1/365) for i in range(len(init_ttms)) if init_ttms[i] > 0])

    # undo the timestep.
    cdo.update_tau(-1/365)
    new_ttms = deepcopy(cdo.get_ttms())
    assert np.array_equal(new_ttms, init_ttms)

    cdo = create_barrier_option(vdf, pdf, volid, 'call', strike, False, 
                                 'amer', 'down', bullet=False, ko=down_bar, 
                                 lots=1)
    init_ttms = deepcopy(cdo.get_ttms())
    cdo.update_tau(2/365)
    new_ttms = deepcopy(cdo.get_ttms())
    try:
        assert all([(init_ttms[i] - new_ttms[i] <= 2/365) or (np.isclose(init_ttms[i] - new_ttms[i], 2/365)) 
                    for i in range(len(init_ttms))])
    except AssertionError as e:
        print([init_ttms[i] - new_ttms[i] <= 2/365 for i in range(len(init_ttms))])
        print('new ttms: ', np.array(new_ttms)*365)
        print('old ttms: ', np.array(init_ttms)*365)
        raise AssertionError from e 

    # undo the timestep.
    cdo.update_tau(-2/365)
    new_ttms = deepcopy(cdo.get_ttms())
    try:
        assert np.allclose(new_ttms, init_ttms)
    except AssertionError as e:
        print('new ttms: ', new_ttms)
        print('old ttms: ', init_ttms)
        raise AssertionError from e
