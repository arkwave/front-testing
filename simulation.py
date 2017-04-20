"""
File Name      : simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 18/4/2017
Python version : 3.5
Description    : Overall script that runs the simulation

"""


################################ imports ###################################
import numpy as np
import pandas as pd
# from scripts.calc import get_barrier_vol
from scripts.classes import Option, Future
from scripts.calc import find_vol_change
from scripts.prep_data import read_data, prep_portfolio, get_rollover_dates, generate_hedges
from math import ceil
import copy
import time
import matplotlib.pyplot as plt
import pprint

###########################################################################
######################## initializing variables ###########################
###########################################################################
# Dictionary of multipliers for greeks/pnl calculation.
# format  =  'product' : [dollar_mult, lot_mult, futures_tick,
# options_tick, pnl_mult]

multipliers = {

    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'LCC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 10, 50],
    'C':   [0.3936786, 127.00717, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}


# list of hedging conditions.
# hedges = {'delta': 'zero',
#           'vega': (-1000, 1000)}
# 'gamma': (-5000, 5000)}
# 'theta': (-1000, 1000)}

# slippage/brokerage
# slippage = 1
# brokerage = 1

# passage of time
timestep = 1/365

########################################################################
########################################################################


#####################################################
############## Main Simulation Loop #################
#####################################################

def run_simulation(voldata, pricedata, expdata, pf, hedges):
    """Each run of the simulation consists of 5 steps:

    1) Feed data into the portfolio.

    2) Compute:
            > change in greeks from price and vol update
            > change in overall value of portfolio from price and vol update.
            > Check for expiry/exercise/knock-in/knock-out. Expiry can be due to barriers or tau = 0. Record changes to:
                    - futures bought/sold as the result of exercise. [PnL]
                    - changes in monthly greeks from options expiring. [PnL]
                    - total number of securities in the portfolio; remove expired options.

    3) Handle the options component:
            > Check if option is bullet or daily.
            > handle exericse appropriately.

    4) PnL calculation. Components include:
            > PnL contribution from changes in price/vols.
            > PnL Contribution from Options
            > PnL from shorting straddles (gamma/vega hedging)

    5) Rebalance the Greeks
            > buy/sell options to hedge gamma/vega according to conditions
            > buy/sell futures to zero delta (if required)

    Process then repeats from step 1 for the next input.

    Inputs:
    1) voldata   : dataframe containing price and vol series for all futures.
    2) pricedata : Portfolio object.
    3) expdata   :
    4) pf        :
    5) hedges    :

    Outputs:
    1) Graph of daily PnL
    2) Graph of cumulative PnL
    3) Various summary statistics.

    """
    t = time.time()
    rollover_dates = get_rollover_dates(pricedata)
    pnl = 0

    daily_values = []
    cumul_values = []

    date_range = sorted(voldata.value_date.unique())  # [1:]

    xvals = range(1, len(date_range)+1)
    # print('date range: ', date_range)

    # hedging frequency counters for delta, gamma, theta, vega respectively.
    counters = [1, 1, 1, 1]

    # Step 1 & 2
    init_val = 0
    broken = False
    for i in range(len(date_range[:3])):
        if broken:
            print('DATA MISSING; ENDING SIMULATION')
            break
        date = date_range[i]
        try:
            next_date = date_range[i+1]
        except IndexError:
            next_date = None
        try:
            prev_date = date_range[i-1]
        except IndexError:
            prev_date = None
        # isolate data relevant for this day.
        date = pd.to_datetime(date)
        print('##################### date: ', date, '################')
        # init_val = pf.compute_value()
        # print('INITIAL VALUE: ', init_val)
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]
        # getting data pertinent to that day.
        # raw_change to be the difference between old and new value per
        # iteration.
        raw_change, pf, broken = feed_data(
            vdf, pdf, pf, rollover_dates, date, prev_date, voldata)
        # pnl += raw_change
        # if broken:
        #     break
    # Step 3
        expenditure, pf = handle_exercise(pf, date, min(date_range))
        pnl += expenditure

        # compute value after updating greeks
        updated_val = pf.compute_value()
        dailypnl = updated_val - init_val if init_val != 0 else 0
        daily_values.append(dailypnl)
        pnl += dailypnl
        cumul_values.append(pnl)
        print('[10]   EOD PNL: ', dailypnl)
        print('[10.5] Cumulative PNL: ', pnl)
    # Step 4
        pf, counters = rebalance(vdf, pdf, pf, hedges, counters)
        print('[13]  EOD PORTFOLIO: ', pf)
        init_val = pf.compute_value()
    # Step 5: Decrement timestep after all steps.
        # calculate number of days to step
        num_days = 0 if next_date is None else (
            pd.Timestamp(next_date) - pd.Timestamp(date)).days
        print('NUM DAYS: ', num_days)
        pf.timestep(num_days * timestep)

    # Step 6: Plotting results/data viz
    elapsed = time.time() - t
    print('Time elapsed: ', elapsed)
    print('##################### PNL: #####################')
    print(pnl)
    print('################# Portfolio: ###################')
    print(pf)

    plt.figure()
    plt.plot(xvals, daily_values, c='c', alpha=0.6, label='daily pnl')
    plt.plot(xvals, cumul_values, c='m', alpha=0.7, label='cumulative pnl')
    plt.legend()
    plt.show()
    # print('Portfolio: ', pf)
    return pnl, pf


##########################################################################
##########################################################################
##########################################################################


##########################################################################
########################## Helper functions ##############################
##########################################################################


def feed_data(voldf, pdf, pf, dic, date, prev_date, voldata):
    """This function does the following:
    1) Computes current value of portfolio.
    2) Checks for rollovers and expiries.
    3) Feeds relevant information into the portfolio.
    4) Asseses knockin/knockouts.
    5) Computes new value of portfolio.
    6) returns change in value, as well as updated portfolio.

    Args:
        voldf (pandas dataframe): dataframe of vols in same format as returned by read_data
        pdf (pandas dataframe): dataframe of prices in same format as returned by read_data
        pf (portfolio object): portfolio specified by portfolio_specs.txt
        dic (dictionary): dictionary of rollover dates, in the format
                {product_i: [c_1 rollover, c_2 rollover, ... c_n rollover]}

    Returns:
        tuple: change in value and updated portfolio object.
    """
    broken = False
    if voldf.empty:
        raise ValueError('vol df is empty!')
    date = voldf.value_date.unique()[0]
    raw_diff = 0

    # 1) initial value of the portfolio before updates.
    prev_val = pf.compute_value()

    # [DEBUGGING STATEMENTS]
    # print('[0]    PREVIOUS VALUE: ', prev_val)
    # print('[0.1]  PRICE BEFORE VOL UPDATE: ', x[0].get_price())
    # print('[0.2]  VOL BEFORE VOL UPDATE: ', x[0].vol)
    # print('[0.3]  GREEKS BEFORE VOL UPDATE: ', pf.OTC['C']['N7'][2:])
    # print('[0.4]  PORFOLIO BEFORE VOL UPDATE: ', pf)

    # 2) Check for rollovers and expiries rollovers
    for product in dic:
        ro_dates = dic[product]
        # rollover date for this particular product
        if date in ro_dates:
            pf.decrement_ordering(product, 1)
    # expiries; also removes options for which ordering = 0
    pf.remove_expired()

    # 3)  update prices of futures, underlying & portfolio alike.

    # 3.5) getting list of all prices that are available for this day. This
    # checks if there are high/low prices, and uses them to deal with barriers
    # if they are available.
    # priceList = []

    # update option attributes by feeding in vol.
    # all_options = pf.get_all_options()
    if not broken:
        for op in pf.get_all_options():
            # info reqd: strike, order, product.
            strike, order, product, tau = op.K, op.ordering, op.product, op.tau
            cpi = 'C' if op.char == 'call' else 'P'
            # interpolate or round? currently rounding, interpolation easy.
            strike = round(strike/10) * 10
            try:
                # print(type(voldf.tau))
                # print(voldf.tau)
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                            (voldf.order == order) & (voldf.call_put_id == cpi) &
                            (np.isclose(voldf.tau.values, tau))].settle_vol.values[0]
                # find the strike corresponding to this delta in the previous
                # day's data
                if prev_date is None:
                    vol_change = 0
                else:
                    date, prev_date = pd.to_datetime(
                        date), pd.to_datetime(prev_date)

                    vol_change = find_vol_change(
                        voldata, val, op, date, prev_date)

                val += vol_change
                op.update_greeks(vol=val)
                # print('UPDATED - new vol: ', val, op)
            except IndexError:
                print('### DATA MISSING ###')
                broken = True
                break

    if not broken:
        for ft in pf.get_all_futures():
            pdt, ordering = ft.get_product(), ft.get_ordering()
            try:
                val = pdf[(pdf.pdt == pdt) & (
                    pdf.order == ordering)].settle_value.values[0]
                returns = pdf[(pdf.pdt == pdt) & (
                    pdf.order == ordering)].returns.values[0]
                # update val with returns
                val = val * (1 + returns)
                ft.update_price(val)
                # print('UPDATED - new price: ', val)

            # index error would occur only if data is missing.
            except IndexError:
                print('###### DATA MISSING #######')
                broken = True
                break

    # updating portfolio after modifying underlying objects
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)

    # [DEBUGGING STATEMENTS]
    # print('[1]  TAU: ', x[0].tau)
    # print('[2]  TTM: ', x[0].tau * 365)
    # print('[3]  VOL AFTER UPDATE: ', x[0].vol)
    # cpi = 'C' if x[0].char == 'call' else 'P'
    # if x[0].barrier:
    #     blvl = x[0].ki if x[0].ki else x[0].ko
    # print('BARRIER LEVEL: ', blvl)
    # print('[3.5]BARRIER VOL: ', get_barrier_vol(
    #     voldf, x[0].get_product(), x[0].tau, cpi, blvl))

    # print('[4]  PRICE AFTER UPDATE: ', x[0].compute_price())
    # print('[5]  GREEKS AFTER UPDATE: ', pf.OTC['C']['N7'][2:])

    # if y:
    #     print('[5.1] GREEKS HEDGE: ', y[0].greeks())
    # print('[5.2] GREEKS HEDGE: ', y[1].greeks())

    # print('[6]  PORFOLIO AFTER UPDATE: ', pf)

    print('[7]  NET GREEKS: ', str(pprint.pformat(pf.net_greeks)))

    # 5) computing new value
    new_val = pf.compute_value()
    raw_diff = new_val - prev_val
    # print('[8]  NEW VALUE AFTER FEED: ', new_val)
    return raw_diff, pf, broken


def handle_exercise(pf, date, sim_start):
    """ Handles option exercise, as well as bullet vs daily payoff.
    Args:
        pf (Portfolio object) : the portfolio being run through the simulator.

    Returns:
        tuple: the combined PnL from exercising (if appropriate) and daily/bullet payoffs, as well as the updated portfolio.

    Notes on implementation:

    1) options are exercised if less than or equal to 2 days to maturity, and option is in the money. Futures obtained are immediately sold and profit is locked in.

    """
    expenditure = 0
    tol = 2/365
    # handle options exercise
    # all_ops = pf.get_all_options()
    otc_ops = pf.OTC_options
    hedge_ops = pf.hedge_options

    for op in otc_ops:
        if op.tau <= tol and op.exercise():
            print("----- EXERCISING CASE: OTC OPS ------")
            date = pd.to_datetime(date)
            sim_start = pd.to_datetime(sim_start)
            print('Time Elapsed: ', (date - sim_start).days)
            print(pf)
            print(op.tau, op.exercise())
            print(op)
            op.update_tau(op.tau)
            # once for exercise, another for selling/buying to cover the
            # future obtained.
            product = op.get_product()
            pnl_mult = multipliers[product][-1]
            # fees = 2*brokerage if not op.shorted else 0
            expenditure += op.lots * op.get_price()*pnl_mult  # - fees

    for op in hedge_ops:
        if op.tau <= tol and op.exercise():
            print("----- EXERCISING CASE: hedge OPS ------")
            print(op.tau, op.exercise())
            print(op)
            op.update_tau(op.tau)
            # once for exercise, another for selling/buying to cover the
            # future obtained.
            product = op.get_product()
            pnl_mult = multipliers[product][-1]
            # fees = 2*brokerage if not op.shorted else 0
            expenditure += op.lots * op.get_price()*pnl_mult  # - fees

    return expenditure, pf


###############################################################################
###############################################################################
###############################################################################


###############################################################################
########### Hedging-related functions (generation and implementation) #########
###############################################################################

def rebalance(vdf, pdf, pf, hedges, counters):
    """ Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.
    Notes:
    1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed.
    2) hedging delta done by shorting/buying -delta * lots futures.
    3)

    Args:
        vdf (pandas dataframe): Dataframe of volatilities
        pdf (pandas dataframe): Dataframe of prices
        pf (object): portfolio object
        hedges (dict): Dictionary of hedging conditions

    Returns:
        tuple: portfolio, counters 
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    # hedging delta, gamma, vega.
    delta_freq, gamma_freq, theta_freq, vega_freq = counters
    print('delta freq: ', delta_freq)
    print('vega freq: ', vega_freq)
    dic = copy.deepcopy(pf.get_net_greeks())
    hedgearr = [False, False, False, False]
    # updating counters
    for strat in hedges:
        if strat == 'delta':
            if delta_freq == hedges[strat][2]:
                counters[0] = 1
                hedgearr[0] = True
            else:
                print('delta freq not met.')
                counters[0] += 1
        elif strat == 'gamma':
            if gamma_freq == hedges[strat][2]:
                counters[1] = 1
                hedgearr[1] = True
            else:
                print('gamma freq not met')
                counters[1] += 1
        elif strat == 'vega':
            if vega_freq == hedges[strat][2]:
                counters[3] = 1
                hedgearr[3] = True
            else:
                print('vega freq not met')
                counters[3] += 1
        elif strat == 'theta':
            if theta_freq == hedges[strat][2]:
                counters[2] = 1
                hedgearr[2] = True
            else:
                print('gamma freq not met')
                counters[2] += 1

    for product in dic:
        for month in dic[product]:
            ordering = pf.compute_ordering(product, month)
            # vega/gamma/theta hedging. loop allows for dynamic hedging dict.
            for strat in hedges:
                # print(strat)
                if strat == 'delta':
                    continue
                # updating counters and setting bool
                elif strat == 'gamma' and hedgearr[1]:
                    inputs = gen_hedge_inputs(
                        hedges, vdf, pdf, month, pf, product, ordering, strat)
                    pf = hedge(pf, inputs, product, month, strat)
                elif strat == 'vega' and hedgearr[3]:
                    inputs = gen_hedge_inputs(
                        hedges, vdf, pdf, month, pf, product, ordering, strat)
                    pf = hedge(pf, inputs, product, month, strat)
                elif strat == 'theta' and hedgearr[2]:
                    inputs = gen_hedge_inputs(
                        hedges, vdf, pdf, month, pf, product, ordering, strat)
                    pf = hedge(pf, inputs, product, month, strat)
            if hedgearr[0]:
                pf, dhedges = hedge_delta(hedges['delta'][1], vdf, pdf,
                                          pf, month, product, ordering)
    print('counters:', counters)
    return pf, counters


# TODO: update this with new objects in mind.
def gen_hedge_inputs(hedges, vdf, pdf, month, pf, product, ordering, flag):
    """Helper function that generates the inputs required to construct atm
    straddles for hedging, based on the flag.

    Args:
        hedges (dict): hedging rules.
        vdf (pandas dataframe): volatility dataframe
        pdf (pandas dataframe): price dataframe
        month (string): month being hedged
        pf (object): portfolio being hedged
        product (string): product being hedged
        ordering (int): ordering corresponding to month being hedged
        flag (string): gamma, vega or theta 

    Returns:
        list : inputs required to construct atm straddles.
    """
    net_greeks = pf.get_net_greeks()
    greeks = net_greeks[product][month]
    if flag == 'gamma':
        greek = greeks[1]
        bound = hedges['gamma'][1]
    elif flag == 'vega':
        greek = greeks[3]
        bound = hedges['vega'][1]
    elif flag == 'theta':
        greek = greeks[2]
        bound = hedges['theta'][1]
    # relevant data for constructing Option and Future objects.
    price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]

    k = round(price/10) * 10
    # print('[8]  STRIKE: ', k)
    cvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'C') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    pvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    tau = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].tau.values[0]
    underlying = Future(month, price, product, ordering=ordering)

    return [price, k, cvol, pvol, tau, underlying, greek, bound, ordering]


def hedge(pf, inputs, product, month, flag):
    """This function does the following:
    1) constructs atm straddles with the inputs from _inputs_
    2) hedges the greek in question (specified by flag) with the straddles.

    Args:
        pf (portfolio object): portfolio object
        inputs (list): list of inputs reqd to construct straddle objects
        product (string): the product being hedged
        month (string): month being hedged

    Returns:
        tuple: cost of the hedge, and the updated portfolio
    """
    # print(flag + ' ' + product + ' ' + month + ' ' + str(inputs))
    price, k, cvol, pvol, tau, underlying, greek, bound, ordering = inputs

    # creating straddle components.
    callop = Option(k, tau, 'call', cvol, underlying,
                    'euro', month=month, ordering=ordering, shorted=None)

    putop = Option(k, tau, 'put', pvol, underlying,
                   'euro', month=month, ordering=ordering, shorted=None)

    lm, dm = multipliers[product][1], multipliers[product][0]

    # gamma and vega hedging.
    upper = bound[1]
    lower = bound[0]
    # print('upper: ', upper)
    # print('lower: ', lower)
    if greek > upper or greek < lower:
        # gamma hedging logic.
        # print(product + ' ' + month + ' ' + flag + ' hedging')
        if greek < lower:
            # print('lower')
            # need to buy straddles for gamma/vega, short for theta.
            if flag == 'gamma':
                callop.shorted = False
                putop.shorted = False
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cgamma * dm) / (callop.lots * lm)
                greek_p = (pgamma * dm) / (putop.lots * lm)
                pgreek, cgreek = pgamma, cgamma
                lots_req = round((abs(greek) * dm) /
                                 ((greek_c + greek_p) * lm))

            elif flag == 'vega':
                callop.shorted = False
                putop.shorted = False
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cvega * 100) / (callop.lots * lm * dm)
                greek_p = (pvega * 100) / (putop.lots * lm * dm)
                pgreek, cgreek = pvega, cvega
                lots_req = round((abs(greek) * 100) /
                                 ((greek_c + greek_p) * lm * dm))

            elif flag == 'theta':
                # print('theta < lower: shorting straddles.')
                callop.shorted = True
                putop.shorted = True
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (ctheta * 365) / (callop.lots * lm * dm)
                greek_p = (ctheta * 365) / (putop.lots * lm * dm)
                pgreek, cgreek = ptheta, ctheta
                lots_req = round((abs(greek) * 365) /
                                 ((greek_c + greek_p) * lm * dm))

            callop.lots, putop.lots = lots_req, lots_req
            num_required = ceil(abs(greek)/(pgreek + cgreek))

            # print('lots req: ', lots_req)
            # print('greek: ', greek)
            # print('pgreek: ', pgreek)
            # print('cgreek: ', cgreek)
            # print('num_req: ', num_required)

            # if flag == 'gamma':
            #     print('callop gamma: ', callop.gamma)
            #     print('putop gamma: ', putop.gamma)
            #     print('total ' + flag + ' added: ',
            #           (callop.gamma + putop.gamma) * num_required)
            # elif flag == 'vega':
            #     print('callop vega: ', callop.vega)
            #     print('putop vega: ', putop.vega)
            #     print('total ' + flag + ' added: ',
            #           (callop.vega + putop.vega) * num_required)
            # elif flag == 'theta':
            #     print('callop theta: ', callop.theta)
            #     print('putop theta: ', putop.theta)
            #     print('total ' + flag + ' added: ',
            #           (callop.theta + putop.theta) * num_required)

        elif greek > upper:
            # need to short straddles for gamma/vega, long theta.
            # print('upper')
            if flag == 'gamma':
                callop.shorted = True
                putop.shorted = True
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cgamma * dm) / (callop.lots * lm)
                greek_p = (pgamma * dm) / (putop.lots * lm)
                pgreek, cgreek = pgamma, cgamma
                lots_req = round((greek * dm)/(abs(greek_c + greek_p) * lm))

            elif flag == 'vega':
                # print('vega')
                callop.shorted = True
                putop.shorted = True
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cvega * 100) / (callop.lots * lm * dm)
                greek_p = (pvega * 100) / (putop.lots * lm * dm)
                pgreek, cgreek = pvega, cvega
                lots_req = round(
                    (greek * 100)/(abs(greek_c + greek_p) * lm * dm))

            elif flag == 'theta':
                # print('theta > upper: buying straddles.')
                callop.shorted = False
                putop.shorted = False
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (ctheta * 365) / (callop.lots * lm * dm)
                greek_p = (ctheta * 365) / (putop.lots * lm * dm)
                pgreek, cgreek = ptheta, ctheta
                lots_req = round(
                    (greek * 365)/(abs(greek_c + greek_p) * lm * dm))

            callop.lots, putop.lots = lots_req, lots_req
            num_required = ceil((greek) / abs(pgreek + cgreek))

            # print('greek: ', greek)
            # print(greek_c, greek_p, lm, dm)
            # print('pgreek: ', pgreek)
            # print('cgreek: ', cgreek)
            # print('lots req: ', lots_req)

            # print('num_required: ', num_required)

            # if flag == 'gamma':
            #     print('callop gamma: ', callop.gamma)
            #     print('putop gamma: ', putop.gamma)
            #     print('total ' + flag + ' added: ',
            #           (callop.gamma + putop.gamma) * num_required)
            # elif flag == 'vega':
            #     print('callop vega: ', callop.vega)
            #     print('putop vega: ', putop.vega)
            #     print('total ' + flag + ' added: ',
            #           (callop.vega + putop.vega) * num_required)
            # elif flag == 'theta':
            #     print('callop theta: ', callop.theta)
            #     print('putop theta: ', putop.theta)
            #     print('total ' + flag + ' added: ',
            #           (callop.theta + putop.theta) * num_required)

        callops = [callop] * num_required
        putops = [putop] * num_required
        cd, cg, ct, cv = callop.greeks()
        pd, pg, pt, pv = putop.greeks()

        pf.add_security(callops, 'hedge')
        pf.add_security(putops, 'hedge')
    else:
        print(str(product) + ' ' + str(month) + ' ' + flag.upper() +
              ' WITHIN BOUNDS. SKIPPING HEDGING')
        pass
    return pf  # [callop, putop]


# NOTE: assuming that future can be found with the exact number of
# lots to delta hedge.
def hedge_delta(cond, vdf, pdf, pf, month, product, ordering):
    """Helper function that implements delta hedging. General idea is to zero out delta at the end of the day by buying/selling -delta * lots futures. Returns expenditure (which is negative if shorting and postive if purchasing delta) and the updated portfolio object.

    Args:
        cond (string): condition for delta hedging
        vdf (dataframe): Dataframe of volatilities
        pdf (dataframe): Dataframe of prices
        net (list): greeks associated with net_greeks[product][month]
        month (str): month of underlying future.
        pf (portfolio): portfolio object specified by portfolio_specs.txt

    Returns:
        tuple: hedging costs and final portfolio with hedges added.

    """
    # print('cond: ', cond)
    # print('delta hedging: ', product + '  ' + month)
    future_price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    net_greeks = pf.get_net_greeks()
    if cond == 'zero':
        # flag that indicates delta hedging.
        vals = net_greeks[product][month]
        delta = vals[0]
        # print('[12] ' + product + ' ' + month + ' delta', delta)
        shorted = True if delta > 0 else False
        num_lots_needed = abs(round(delta))
        # print('lots needed: ', num_lots_needed)
        if num_lots_needed == 0:
            print(str(product) + ' ' + str(month) +
                  ' DELTA IS ZEROED. SKIPPING HEDGING')
            return pf, None
        else:
            ft = Future(month, future_price, product,
                        shorted=shorted, ordering=ordering, lots=num_lots_needed)
            pf.add_security([ft], 'hedge')
    return pf, ft


##########################################################################
##########################################################################
##########################################################################

# Commands upon running the file

if __name__ == '__main__':
    datapath = 'data_loc.txt'
    t = time.time()
    vdf, pdf, edf = read_data(datapath)
    # check sanity of data
    vdates = pd.to_datetime(vdf.value_date.unique())
    # print(vdates)

    pdates = pd.to_datetime(pdf.value_date.unique())
    # print(pdates)
    if not np.array_equal(vdates, pdates):
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range.')

    # generate portfolio
    # filepath = 'specs.csv'
    filepath = 'datasets/corn_portfolio_specs.csv'
    # filepath = 'datasets/bo_portfolio_specs.csv'
    pf, sim_start = prep_portfolio(vdf, pdf, filepath=filepath)
    print(pf)
    vdf, pdf = vdf[vdf.value_date >= sim_start], \
        pdf[pdf.value_date >= sim_start]
    print('NUM OPS: ', len(pf.OTC_options))
    e1 = time.time() - t
    print('[data & portfolio prep]: ', e1)
    # generate hedges
    hedge_path = 'hedging.csv'
    hedges = generate_hedges(hedge_path)
    print(hedges)
    pnl, pf1 = run_simulation(vdf, pdf, edf, pf, hedges=hedges)

    e2 = time.time() - e1
    print('[simulation]: ', e2)


#######################################################################
#######################################################################
#######################################################################

# code dump

    # proceed to run simulation
    # pnl_list = []

    # for i in range(10):
    #     pf, sim_start = prep_portfolio(vdf, pdf, filepath=filepath)
    #     print(pf)
    #     vdf, pdf = vdf[vdf.value_date >= sim_start], \
    #         pdf[pdf.value_date >= sim_start]
    #     pnl, pf1 = run_simulation(vdf, pdf, edf, pf)
    #     pnl_list.append(pnl)
    #     print('run ' + str(i) + ' completed')

    # print('mean pnl: ', np.mean(pnl_list))
    # print('pnl sd: ', np.std(pnl_list))
