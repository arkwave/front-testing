"""
File Name      : simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 4/5/2017
Python version : 3.5
Description    : Overall script that runs the simulation

"""


################################ imports ###################################
import numpy as np
import pandas as pd
# from scripts.calc import get_barrier_vol
from scripts.classes import Option, Future
# from scripts.calc import find_vol_change
from scripts.prep_data import read_data, prep_portfolio, get_rollover_dates, generate_hedges, get_min_start_date

from scripts.calc import compute_strike_from_delta
# from math import ceil
import copy
import time
import matplotlib.pyplot as plt
import pprint
import scripts.global_vars as gv
import os

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
seed = 7
np.random.seed(seed)

########################################################################
########################################################################


#####################################################
############## Main Simulation Loop #################
#####################################################

def run_simulation(voldata, pricedata, expdata, pf, hedges, rollover_dates, end_date=None, brokerage=None, slippage=None):
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
    t = time.clock()
    # rollover_dates = get_rollover_dates(pricedata)
    grosspnl = 0
    netpnl = 0

    gross_daily_values = []
    gross_cumul_values = []
    net_daily_values = []
    net_cumul_values = []

    date_range = sorted(voldata.value_date.unique())  # [1:]

    # hedging frequency counters for delta, gamma, theta, vega respectively.
    counters = [1, 1, 1, 1]

    init_val = 0
    broken = False
    for i in range(len(date_range)):

        # get the current date
        date = date_range[i]

    # Steps 1 & 2: Error checks to prevent useless simulation runs.

        # broken = True if there is missing data.
        if broken:
            print('DATA MISSING; ENDING SIMULATION...')
            break

        # checks to make sure if there are still non-hedge securities in pf
        if len(pf.OTC_options) == 0 and len(pf.OTC_futures) == 0:
            print('PORTFOLIO IS EMTPY. ENDING SIMULATION...')
            break

        # if end_date is inputted, check to see if the current date exceeds
        # end_date
        if end_date:
            if date >= end_date:
                print('REACHED END OF SIMULATION.')
                break

        try:
            next_date = date_range[i+1]
        except IndexError:
            next_date = None

        # isolate data relevant for this day.
        date = pd.to_datetime(date)
        print('##################### date: ', date, '################')

        # filter data specific to the current day of the simulation.
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]

        print('Portfolio before feed: ', pf)

    # Step 3: Feed data into the portfolio.
        raw_change, pf, broken = feed_data(
            vdf, pdf, pf, rollover_dates, brokerage=brokerage, slippage=slippage)

    # Step 4: Compute pnl for the day
        updated_val = pf.compute_value()
        dailypnl = updated_val - init_val if init_val != 0 else 0

    # Step 5: Hedge.
        # cost = 0
        pf, counters, cost = rebalance(
            vdf, pdf, pf, hedges, counters, brokerage=brokerage, slippage=slippage)

    # Step 6: Subtract brokerage/slippage costs from rebalancing. Append to
    # relevant lists.
        gross_daily_values.append(dailypnl)
        net_daily_values.append(dailypnl - cost)
        grosspnl += dailypnl
        gross_cumul_values.append(grosspnl)
        netpnl += (dailypnl - cost)
        net_cumul_values.append(netpnl)

        print('[10]   EOD PNL (GROSS): ', dailypnl)
        print('[10.1] EOD PNL (NET) :', dailypnl - cost)
        print('[10.2] Cumulative PNL (GROSS): ', grosspnl)
        print('[10.3] Cumulative PNL (net): ', netpnl)
        print('[13]  EOD PORTFOLIO: ', pf)
        init_val = pf.compute_value()

    # Step 7: Placeholders for the signal-generating program.
        # sec, hedgeDict, close_pos = get_signal()
        # pf.add_security(sec, 'OTC')
        # if hedgeDict is not None:
        #     hedges = hedgeDict

    # Step 8: Decrement timestep after all steps.
        # calculate number of days to step
        num_days = 0 if next_date is None else (
            pd.Timestamp(next_date) - pd.Timestamp(date)).days
        print('NUM DAYS: ', num_days)
        pf.timestep(num_days * timestep)

    # Step 9: Plotting results/data viz
    elapsed = time.clock() - t
    print('Time elapsed: ', elapsed)
    print('##################### PNL: #####################')
    print('gross pnl: ', grosspnl)
    print('net pnl: ', netpnl)
    print('################# Portfolio: ###################')
    print(pf)

    return grosspnl, netpnl, pf, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values


##########################################################################
##########################################################################
##########################################################################


##########################################################################
########################## Helper functions ##############################
##########################################################################


def feed_data(voldf, pdf, pf, dic, brokerage=None, slippage=None):
    """
    This function does the following:
        1) Computes current value of portfolio.
        2) Checks for rollovers and expiries.
        3) Feeds relevant information into the portfolio.
        4) Asseses knockin/knockouts. <- taken care of automatically upon feeding in data.
        5) Computes new value of portfolio.
        6) returns change in value, as well as updated portfolio.

    Args:
        voldf (pandas dataframe): dataframe of vols in same format as returned by read_data
        pdf (pandas dataframe): dataframe of prices in same format as returned by read_data
        pf (portfolio object): portfolio specified by portfolio_specs.txt
        dic (dictionary): dictionary of rollover dates, in the format
                {product_i: [c_1 rollover, c_2 rollover, ... c_n rollover]}
        brokerage (int, optional): brokerage fees per lot.
        slippage (int, optional): slippage cost

    Returns:
        tuple: change in value, updated portfolio object, and whether or not there is missing data.

    Raises:
        ValueError: Raised if voldf is empty.
    """
    broken = False
    if voldf.empty:
        raise ValueError('vol df is empty!')
    date = voldf.value_date.unique()[0]
    raw_diff = 0

    # 1) initial value of the portfolio before updates, and handling exercises
    # before feeding data.
    prev_val = pf.compute_value()
    expenditure, pf = handle_exercise(pf, brokerage, slippage)
    raw_diff += expenditure

    # 2) Check for rollovers and expiries rollovers
    for product in dic:
        ro_dates = dic[product]
        # rollover date for this particular product
        if date in ro_dates:
            pf.decrement_ordering(product, 1)
    # expiries; also removes options for which ordering = 0
    pf.remove_expired()

    # 3)  update prices of futures, underlying & portfolio alike.
    # update option attributes by feeding in vol.
    if not broken:
        for op in pf.get_all_options():
            # info reqd: strike, order, product, tau
            strike, order, product, tau = op.K, op.ordering, op.product, op.tau
            # print('price: ', op.compute_price())
            # print('OP GREEKS: ', op.greeks())
            cpi = 'C' if op.char == 'call' else 'P'
            # interpolate or round? currently rounding, interpolation easy.
            strike = round(strike/10) * 10
            try:
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                            (voldf.order == order) & (voldf.call_put_id == cpi)]
                df_tau = min(val.tau, key=lambda x: abs(x-tau))
                val = val[val.tau == df_tau].settle_vol.values[0]
                op.update_greeks(vol=val)
                print('UPDATED - new vol: ', val)
            except IndexError:
                print('### DATA MISSING ###')
                print('product: ', product)
                print('strike: ', strike)
                print('order: ', order)
                print('call put id: ', cpi)
                print('tau: ', df_tau)
                broken = True
                break
            # print(str(op) + ' OP VALUE AFTER FEED: ', op.compute_price())
            # print('OP GREEKS: ', op.greeks())
    if not broken:
        for ft in pf.get_all_futures():
            pdt, ordering = ft.get_product(), ft.get_ordering()
            try:
                val = pdf[(pdf.pdt == pdt) & (
                    pdf.order == ordering)].settle_value.values[0]
                print('UPDATED - new price: ', val)
                ft.update_price(val)

            # index error would occur only if data is missing.
            except IndexError:
                print('###### DATA MISSING #######')
                broken = True
                break

    # updating portfolio after modifying underlying objects
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)

    # print('Portfolio After Feed: ', pf)
    print('[7]  NET GREEKS: ', str(pprint.pformat(pf.net_greeks)))

    # 5) computing new value
    new_val = pf.compute_value()
    raw_diff = new_val - prev_val
    return raw_diff, pf, broken


def handle_exercise(pf, brokerage=None, slippage=None):
    """ Handles option exercise, as well as bullet vs daily payoff.
    Args:
        pf (Portfolio object) : the portfolio being run through the simulator.

    Returns:
        tuple: the combined PnL from exercising (if appropriate) and daily/bullet payoffs, as well as the updated portfolio.

    Notes on implementation:

    1) options are exercised if less than or equal to 2 days to maturity, and option is in the money. Futures obtained are immediately sold and profit is locked in.

    """
    t = time.clock()
    profit = 0
    tol = 1/365
    # handle options exercise
    # all_ops = pf.get_all_options()
    otc_ops = pf.OTC_options
    hedge_ops = pf.hedge_options
    for op in otc_ops:
        if op.tau < tol:
            if op.exercise():
                if op.settlement == 'cash':
                    print("----- CASH SETTLEMENT: OTC OP ------")
                    print("exercising OTC op " + str(op))
                    op.update_tau(op.tau)
                    product = op.get_product()
                    pnl_mult = multipliers[product][-1]
                    oppnl = op.lots * op.get_price()*pnl_mult
                    print('profit on this exercise: ', oppnl)
                    print("-------------------------------------")

                    profit += oppnl  # - fees
                elif op.settlement == 'futures':
                    print('----- FUTURE SETTLEMENT: OTC OP ------')
                    pf.exercise_option(op, 'OTC')
            else:
                print('letting OTC op ' + str(op) + ' expire.')

    for op in hedge_ops:
        if op.tau < tol:
            if op.exercise():
                if op.settlement == 'cash':
                    print("----- CASH SETTLEMENT: HEDGE OPS ------")
                    print('exercising hedge op ' + str(op))
                    op.update_tau(op.tau)
                    product = op.get_product()
                    pnl_mult = multipliers[product][-1]
                    oppnl = op.lots * op.get_price()*pnl_mult
                    print('profit on this exercise: ', oppnl)
                    print('---------------------------------------')
                    profit += oppnl
                elif op.settlement == 'futures':
                    print('----- FUTURE SETTLEMENT: HEDGE OPS ------')
                    pf.exercise_option(op, 'hedge')
            else:
                print('letting hedge op ' + str(op) + ' expire.')

    print(' ### net exercise profit: ', profit)
    print('handle expiry time: ', time.clock() - t)
    return profit, pf


###############################################################################
###############################################################################
###############################################################################


###############################################################################
########### Hedging-related functions (generation and implementation) #########
###############################################################################

def rebalance(vdf, pdf, pf, hedges, counters, brokerage=None, slippage=None):
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
    # print('delta freq: ', delta_freq)
    # print('vega freq: ', vega_freq)
    dic = copy.deepcopy(pf.get_net_greeks())
    hedgearr = [False, False, False, False]
    # updating counters
    for greek in hedges:
        if greek == 'delta':
            if delta_freq == hedges[greek][0][2]:
                counters[0] = 1
                hedgearr[0] = True
            else:
                print('delta freq not met.')
                counters[0] += 1
        elif greek == 'gamma':
            if gamma_freq == hedges[greek][0][2]:
                counters[1] = 1
                hedgearr[1] = True
            else:
                print('gamma freq not met')
                counters[1] += 1
        elif greek == 'vega':
            if vega_freq == hedges[greek][0][2]:
                counters[3] = 1
                hedgearr[3] = True
            else:
                print('vega freq not met')
                counters[3] += 1
        elif greek == 'theta':
            if theta_freq == hedges[greek][0][2]:
                counters[2] = 1
                hedgearr[2] = True
            else:
                print('gamma freq not met')
                counters[2] += 1
    done_hedging = hedges_satisfied(pf, hedges)
    roll_hedged = check_roll_status(pf, hedges)

    cost = 0

    if not roll_hedged:
        roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedge[
            'delta'][i][0] == 'roll']
        pf, exp = hedge_delta_roll(
            pf, roll_cond, pdf, brokerage=brokerage, slippage=slippage)
        cost += exp

    while not done_hedging:
        for product in dic:
            for month in dic[product]:
                ordering = pf.compute_ordering(product, month)
                # vega/gamma/theta hedging. loop allows for dynamic hedging
                # dict.
                for strat in hedges:
                    # print(strat)
                    if strat == 'delta':
                        continue
                    # updating counters and setting bool
                    elif strat == 'gamma' and hedgearr[1]:
                        inputs = gen_hedge_inputs(
                            hedges, vdf, pdf, month, pf, product, ordering, strat)
                        pf, fees = hedge(pf, inputs, product, month,
                                         strat, brokerage=brokerage, slippage=slippage)
                    elif strat == 'vega' and hedgearr[3]:
                        inputs = gen_hedge_inputs(
                            hedges, vdf, pdf, month, pf, product, ordering, strat)
                        pf, fees = hedge(pf, inputs, product, month,
                                         strat, brokerage=brokerage, slippage=slippage)

                    elif strat == 'theta' and hedgearr[2]:
                        inputs = gen_hedge_inputs(
                            hedges, vdf, pdf, month, pf, product, ordering, strat)
                        pf, fees = hedge(pf, inputs, product, month,
                                         strat, brokerage=brokerage, slippage=slippage)
                    cost += fees
                if hedgearr[0]:
                    # grabbing condition that indicates zeroing condition on
                    # delta
                    delta_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
                        'delta'][i][0] == 'static'][0][1]
                    # print('delta conds: ', delta_cond)
                    pf, dhedges, fees = hedge_delta(
                        delta_cond, vdf, pdf, pf, month, product, ordering, brokerage=brokerage, slippage=slippage)
                    cost += fees
        done_hedging = hedges_satisfied(pf, hedges)
    return pf, counters, cost


# TODO: update this with new objects in mind.
def gen_hedge_inputs(hedges, vdf, pdf, month, pf, product, ordering, flag):
    """Helper function that generates the inputs required to construct atm
    straddles for hedging, based on the flag.

    Args:
        hedges (dict): hedging rules.
        vdf (pandas dataframe): volatility dataframef
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
        # bound = [hedges['gamma'][i][1] for i in range(len(hedges)) if hedges[
        #     'gamma'][i][0] == 'bound'][0]
    elif flag == 'vega':
        greek = greeks[3]
        # bound = hedges['vega'][0][1]
    elif flag == 'theta':
        greek = greeks[2]
        # bound = hedges['theta'][0][1]

    # grabbing bound
    relevant_conds = [hedges[flag][i] for i in range(len(hedges[flag])) if hedges[
        flag][i][0] == 'bound'][0]
    bound = relevant_conds[1]
    # print('relevant_conds: ', relevant_conds)
    # print('bound: ', bound)

    # relevant data for constructing Option and Future objects.
    price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]

    k = round(price/10) * 10
    # print('[8]  STRIKE: ', k)
    cvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'C') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    pvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    # if no tau provided, default to using same-month atm straddles.
    tau_vals = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].tau
    # print('max tau: ', max(tau_vals))
    # print('min tau: ', min(tau_vals))
    # if not (np.isclose(max(tau_vals), min(tau_vals))):
    #     print('tau differential present')

    if len(relevant_conds) == 3:
        tau = max(tau_vals)
        # print('order: ', ordering)
        # print('ttm assigned: ', tau*365)
    else:
        target = relevant_conds[3]
        tau = min(tau_vals, key=lambda x: abs(x-target))
        # print('order: ', ordering)
        # print('ttm assigned: ', tau*365)

    underlying = Future(month, price, product, ordering=ordering)

    return [price, k, cvol, pvol, tau, underlying, greek, bound, ordering]


def hedge(pf, inputs, product, month, flag, brokerage=None, slippage=None):
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
    fees = 0
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
        print(product + ' ' + month + ' ' + flag + ' hedging')

        print('cvol: ', cvol)
        print('pvol: ', pvol)
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
                # print('LONG debug [1] - vega below lower', greek, lower)
                callop.shorted = False
                putop.shorted = False
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cvega * 100) / (callop.lots * lm * dm)
                greek_p = (pvega * 100) / (putop.lots * lm * dm)
                pgreek, cgreek = pvega, cvega
                lots_req = round((abs(greek) * 100) /
                                 ((greek_c + greek_p) * lm * dm))
                # print('actual greeks: ', greek * 100,
                #       abs(greek_c + greek_p), lm, dm)

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
                # print('SHORT debug [1] - vega exceeds upper', greek, upper)
                callop.shorted = True
                putop.shorted = True
                cdelta, cgamma, ctheta, cvega = callop.greeks()
                pdelta, pgamma, ptheta, pvega = putop.greeks()
                greek_c = (cvega * 100) / (callop.lots * lm * dm)
                greek_p = (pvega * 100) / (putop.lots * lm * dm)
                pgreek, cgreek = pvega, cvega
                lots_req = round(
                    (greek * 100)/(abs(greek_c + greek_p) * lm * dm))
                # print('actual greeks: ', greek * 100,
                #       abs(greek_c + greek_p), lm, dm)

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

        callops = [callop]
        putops = [putop]
        cd, cg, ct, cv = callop.greeks()
        pd, pg, pt, pv = putop.greeks()

        if brokerage:
            fees = brokerage * 2 * lots_req

        # TODO: determine if slippage by vol or slippage by ticksize
        if slippage:
            fees += (slippage * 2 * lots_req)

        pf.add_security(callops, 'hedge')
        pf.add_security(putops, 'hedge')
    else:
        print(str(product) + ' ' + str(month) + ' ' + flag.upper() +
              ' WITHIN BOUNDS. SKIPPING HEDGING')
        pass
    print('hedging fees - ' + flag + ': ', fees)
    return pf, fees   # [callop, putop]


# NOTE: assuming that future can be found with the exact number of
# lots to delta hedge.
def hedge_delta(cond, vdf, pdf, pf, month, product, ordering, brokerage=None, slippage=None):
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
    print('delta hedging: ', product + '  ' + month)
    future_price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    net_greeks = pf.get_net_greeks()
    fees = 0
    print('cond: ', cond)
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
            return pf, None, 0
        else:
            ft = Future(month, future_price, product,
                        shorted=shorted, ordering=ordering, lots=num_lots_needed)
            pf.add_security([ft], 'hedge')
    if brokerage:
        fees = brokerage * num_lots_needed
    if slippage:
        # TODO
        pass
    print('hedging fees - delta: ', fees)
    return pf, ft, fees


# FIXME: Needs to be implemented
def hedge_delta_roll(pf, rolL_cond, pdf, brokerage=None, slippage=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta exceeds certain bounds.

    Args:
        pf (TYPE): Portfolio being hedged
        roll_cond (TYPE): list of the form ['roll', value, frequency, bound]
        vdf (TYPE): volatility data frame containing information for current day
        pdf (TYPE): price dataframe containing information for current day
        brokerage (int, optional): brokerage fees per lot
        slippage (int, optional): slippage loss

    Returns:
        tuple: updated portfolio and cost of purchasing/selling options.
    """
    cost = 0
    roll_val, bounds = roll_cond[1]/100, roll_cond[3]
    pfx = copy.deepcopy(pf)
    for op in pfx.OTC_options:
        delta = abs(op.delta/op.lots)
        # case: delta not in bounds. roll delta.
        if delta > bounds[1] or delta < bounds[0]:
            # get strike corresponding to delta
            cpi = 'C' if op.char == 'call' else 'P'
            # get the vol from the vol_by_delta part of pdf
            col = str(roll_val) + 'd'
            try:
                vol = pdf[(pdf.call_put_id == cpi) &
                          (pdf.order == op.ordering) &
                          (pdf.pdt == op.get_product()) &
                          (pdf.tau == op.tau)][col].values[0]
            except IndexError:
                print(cpi, op.ordering, op.get_product(), op.tau)
                print('vol not found in volbydelta')
                vol = op.vol

            strike = compute_strike_from_delta(op, delta1=roll_val, vol=vol)

            newop = Option(strike, op.tau, op.char, vol, op.underlying,
                           op.payoff, op.shorted, op.month, direc=op.direc,
                           barrier=op.barrier, lots=op.lots, bullet=op.bullet,
                           ki=op.ki, ko=op.ko, rebate=op.rebate,
                           ordering=op.ordering, settlement=op.settlement)
            pf.remove_security(op, 'OTC')
            pf.add_security(newop, 'OTC')

            # handle expenses: brokerage and old op price - new op price
            cost += (brokerage * (op.lots + newop.lots)) + \
                (op.compute_price() - newop.compute_price())

    return pf, cost


###############################################################################
############################ Helper functions #################################
###############################################################################

def hedges_satisfied(pf, hedges):
    """Helper method that ascertains if all entries in net_greeks satisfy the conditions laid out in hedges.

    Args:
        pf (portfolio object): portfolio being hedged
        hedges (ordered dictionary): contains hedge information/specifications

    Returns:
        Boolean: indicating if the hedges are all satisfied or not.
    """
    strs = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
    net_greeks = pf.net_greeks
    # delta condition:
    conditions = []
    for greek in hedges:
        conds = hedges[greek]
        for cond in conds:
            # static bound case
            if cond[0] == 'static':
                conditions.append((strs[greek], (-1, 1)))
            elif cond[0] == 'bound':
                # print('to be literal eval-ed: ', hedges[greek][1])
                c = cond[1]
                tup = (strs[greek], c)
                conditions.append(tup)
    # bound_and_static = True
    for pdt in net_greeks:
        for month in net_greeks[pdt]:
            greeks = net_greeks[pdt][month]
            for cond in conditions:
                bound = cond[1]
                if (greeks[cond[0]] > bound[1]) or (greeks[cond[0]] < bound[0]):
                    return False
    # rolls_satisfied = check_roll_hedges(pf, hedges)
    return True


def check_roll_status(pf, hedges):
    """Checks to see if delta-roll conditions, if they exist, are satisfied.

    Args:
        pf (portfolio): Portfolio object
        hedges (dictionary): dictionary of hedges.

    Returns:
        boolean: True if delta is within bounds, false otherwise.
    """
    delta_conds = hedges['delta']
    found = False
    # search for the roll condition
    for cond in delta_conds:
        if cond[0] == 'roll':
            rollbounds = cond[3]
            found = True
    # if roll conditions actually exist, proceed.
    if found:
        # print('roll bounds: ', rollbounds)
        for op in pf.OTC_options:
            d = abs(op.delta/op.lots)
            utol, ltol = rollbounds[1], rollbounds[0]
            if d > utol or d < ltol:
                return False
        return True
    # if roll condition doesn't exist, then default to True.
    else:
        return True


#######################################################################
#######################################################################
#######################################################################


if __name__ == '__main__':

    #################### initializing default params ##########################
    # filepath to datasets #
    # datapath = 'data_loc.txt'

    # fix portfolio start date #
    # start_date = gv.start_date

    # filepath to portfolio specs. #
    # specpath = gv.portfolio_path
    specpath = 'specs.csv'
    # specpath = 'datasets/bo_portfolio_specs.csv'

    # fix portfolio internal date #
    # internal_date = pd.Timestamp('2014-08-21')
    # internal_date = gv.internal_date

    # fix end date of simulation #
    # end_date = gv.end_date
    end_date = None

    # path to hedging conditions #
    hedge_path = gv.hedge_path

    print('####################################################')
    print('DEFAULT PARAMETERS INITIALIZED. READING IN DATA... [1/7]')
    ##########################################################################

    t = time.clock()

    # reading in data. check to see if cleaned dfs exist, if not generate from
    # scratch #

    ### small data ###
    volpath, pricepath, exppath, rollpath = gv.small_final_vol_path,\
        gv.small_final_price_path,\
        gv.small_final_exp_path,\
        gv.small_cleaned_price

    ### full data ###
    # volpath, pricepath, exppath, rollpath = gv.final_vol_path,
    # gv.final_price_path, gv.final_exp_path, gv.cleaned_price

    writeflag = 'small' if 'small' in volpath else 'full'

    if os.path.exists(volpath) and os.path.exists(pricepath)\
            and os.path.exists(exppath)\
            and os.path.exists(rollpath):
        print('####################################################')
        print('DATA EXISTS. READING IN... [2/7]')
        print('datasets listed below: ')
        print('voldata : ', volpath)
        print('pricepath: ', pricepath)
        print('exppath: ', exppath)
        print('rollpath: ', rollpath)
        print('####################################################')

        vdf, pdf, edf = pd.read_csv(volpath), pd.read_csv(
            pricepath), pd.read_csv(exppath)
        rolldf = pd.read_csv(rollpath)
        rolldf.value_date, rolldf.expdate = pd.to_datetime(
            rolldf.value_date), pd.to_datetime(rolldf.expdate)
        vdf.value_date = pd.to_datetime(vdf.value_date)
        pdf.value_date = pd.to_datetime(pdf.value_date)
        edf.expiry_date = pd.to_datetime(edf.expiry_date)

    else:
        print('####################################################')
        print('current_dir: ', os.getcwd())
        print('paths inputted: ', volpath, pricepath, exppath, rollpath)
        print('DATA NOT FOUND. PREPARING... [2/7]')

        # full-size data
        # volpath, pricepath, epath = gv.raw_vol_path, gv.raw_price_path, gv.raw_exp_path
        # small-size data
        volpath, pricepath, epath = gv.small_vol_path, gv.small_price_path, gv.raw_exp_path

        print('datasets listed below: ')
        print('voldata : ', volpath)
        print('pricepath: ', pricepath)
        print('exppath: ', epath)
        # print('rollpath: ', rollpath)
        print('####################################################')
        vdf, pdf, edf, rolldf = read_data(
            volpath, pricepath, epath, test=False,  writeflag=writeflag)

    print('DATA READ-IN COMPLETE. SANITY CHECKING... [3/7]')
    print('READ-IN RUNTIME: ', time.clock() - t)

    ######################### check sanity of data ###########################

    start_date = get_min_start_date(vdf, pdf, vdf.vol_id.unique())
    vdf, pdf = vdf[vdf.value_date >= start_date], \
        pdf[pdf.value_date >= start_date]

    vdates = pd.to_datetime(sorted(vdf.value_date.unique()))
    pdates = pd.to_datetime(sorted(pdf.value_date.unique()))
    # check to see that date ranges are equivalent for both price and vol data
    if not np.array_equal(vdates, pdates):
        print('vol_dates: ', vdates)
        print('price_dates: ', pdates)
        print('difference: ', [x for x in vdates if x not in pdates])
        print('difference 2: ', [x for x in pdates if x not in vdates])
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range. Aborting run.')
    # check that the right set of data has been filtered
    # if min(vdates) != start_date or min(pdates) != start_date:
    #     print('min_vdate: ', min(vdates))
    #     print('min_pdate: ', min(pdates))
    #     print('start_date: ', start_date)
    #     raise ValueError('Inconsistency in start date. Aborting run.')

    # # check to ensure that initialization data is contained in dataset.
    # if (internal_date not in set(vdates)) or (internal_date not in set(pdates)):
    #     raise ValueError('Internal portfolio date is not in datasets')

    # if end_date specified, check that it makes sense (i.e. is greater than
    # start date)
    if end_date:
        if start_date > end_date or start_date > end_date:
            raise ValueError(
                'Invalid end_date entered; current end_date is less than start_date')
    print('####################################################')
    print('SANITY CHECKING COMPLETE. PREPPING PORTFOLIO... [4/7]')
    ##########################################################################

    # generate portfolio #
    pf, start_date = prep_portfolio(vdf, pdf, filepath=specpath)
    print('####################################################')
    print('PORTFOLIO PREPARED. GENERATING HEDGES... [5/7]')

    # generate hedges #
    hedges = generate_hedges(hedge_path)

    e1 = time.clock() - t
    print('TOTAL PREP TIME: ', e1)

    # print statements for informational purposes #
    print('START DATE: ', start_date)
    # print('INTERNAL_DATE: ', internal_date)
    print('END DATE: ', end_date)
    print('Portfolio: ', pf)
    print('NUM OPS: ', len(pf.OTC_options))
    print('Hedge Conditions: ', hedges)
    print('Brokerage: ', gv.brokerage)
    print('Slippage: ', gv.slippage)
    # print('[prep_data and prep_portfolio time taken]: ', e1)
    print('####################################################')
    print('HEDGES GENERATED. RUNNING SIMULATION... [6/7]')

    rollover_dates = get_rollover_dates(rolldf)
    print('ROLLOVER DATES: ', rollover_dates)
    # run simulation #
    grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values = run_simulation(
        vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage, slippage=gv.slippage)
    print('####################################################')
    print('SIMULATION COMPLETE. PRINTING RELEVANT OUTPUT... [7/7]')

    print('############################ PNLS ############################')
    print('net pnl: ', netpnl)
    print('gross pnl: ', grosspnl)
    print('daily pnls     [gross]: ', gross_daily_values)
    print('daily pnls     [net]: ', net_daily_values)
    print('cumulative pnl [gross]: ', gross_cumul_values)
    print('cumulative pnl [net]: ', net_cumul_values)
    print('##############################################################')

    gvar = np.percentile(gross_daily_values, 5)
    cvar = np.percentile(net_daily_values, 5)

    # print('num_daily_pnls: ', len(pnl_per_block))
    # print('num_cumul_pnls: ', len(cumul_pnl))

    print('VaR [gross]: ', gvar)
    print('VaR [net]: ', cvar)
    # calculate max drawdown
    print('Max Drawdown [gross]: ', min(np.diff(gross_daily_values)))
    print('Max Drawdown [net]: ', min(np.diff(net_daily_values)))

    # time elapsed 2 #
    e2 = time.clock() - e1
    print('SIMULATION RUNTIME: ', e2)

    # plotting histogram of daily pnls
    plt.figure()
    plt.hist(gross_daily_values, bins=20,
             alpha=0.6, label='gross pnl distribution')
    plt.hist(net_daily_values, bins=20,
             alpha=0.6, label='net pnl distribution')
    plt.title('PnL Distribution: Gross/Net')
    plt.legend()
    plt.show()

    # plotting gross pnl values
    plt.figure()
    colors = ['c' if x >= 0 else 'r' for x in gross_daily_values]
    xvals = list(range(1, len(gross_daily_values) + 1))
    plt.bar(xvals, gross_daily_values, align='center',
            color=colors, alpha=0.6, label='gross daily values')
    plt.plot(xvals, gross_cumul_values, c='k',
             alpha=0.8, label='gross cumulative pnl')
    plt.title('gross pnl daily')
    plt.legend()
    plt.show()

    # plotting net pnl values
    plt.figure()
    colors = ['c' if x >= 0 else 'r' for x in net_daily_values]
    xvals = list(range(1, len(net_daily_values) + 1))
    plt.bar(xvals, net_daily_values, align='center',
            color=colors, alpha=0.6, label='net daily values')
    plt.plot(xvals, net_cumul_values, c='k',
             alpha=0.8, label='net cumulative pnl')
    plt.title('net pnl daily')
    plt.legend()
    plt.show()

#######################################################################
#######################################################################
#######################################################################


##########################################################################
##########################################################################
##########################################################################


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

    # # Commands upon running the file

# if __name__ == '__main__':
#     datapath = 'data_loc.txt'
#     t = time.clock()
#     vdf, pdf, edf = read_data(datapath)
#     # check sanity of data
#     vdates = pd.to_datetime(vdf.value_date.unique())
#     # print(vdates)

#     pdates = pd.to_datetime(pdf.value_date.unique())
#     # print(pdates)
#     if not np.array_equal(vdates, pdates):
#         raise ValueError(
#             'Invalid data sets passed in; vol and price data must have the same date range.')

#     # generate portfolio
#     # filepath = 'specs.csv'
#     filepath = 'datasets/corn_portfolio_specs.csv'
#     # filepath = 'datasets/bo_portfolio_specs.csv'
#     pf, sim_start = prep_portfolio(vdf, pdf, filepath=filepath)
#     print(pf)
#     vdf, pdf = vdf[vdf.value_date >= sim_start], \
#         pdf[pdf.value_date >= sim_start]
#     print('NUM OPS: ', len(pf.OTC_options))
#     e1 = time.clock() - t
#     print('[data & portfolio prep]: ', e1)
#     # generate hedges
#     hedge_path = 'hedging.csv'
#     hedges = generate_hedges(hedge_path)
#     print(hedges)
#     pnl, pf1 = run_simulation(vdf, pdf, edf, pf, hedges=hedges)

#     e2 = time.clock() - e1
#     print('[simulation]: ', e2)
