"""
File Name      : simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 9/5/2017
Python version : 3.5
Description    : Overall script that runs the simulation

"""

################################ imports ###################################
import numpy as np
import pandas as pd
from scripts.portfolio import Portfolio
from scripts.classes import Option, Future
from scripts.prep_data import read_data, prep_portfolio, get_rollover_dates, generate_hedges, find_cdist
from collections import OrderedDict
from scripts.calc import compute_strike_from_delta
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

contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'LCC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V' 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


# Dictionary mapping month to symbols and vice versa
month_to_sym = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
sym_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
                'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
decade = 10


# passage of time
timestep = 1/365
seed = 7
np.random.seed(seed)

########################################################################
########################################################################


#####################################################
############## Main Simulation Loop #################
#####################################################

def run_simulation(voldata, pricedata, expdata, pf, hedges, rollover_dates, end_date=None, brokerage=None, slippage=None, signals=None):
    """
    Each run of the simulation consists of 5 steps:
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

    Args:
        signals (TYPE): Description
        voldata (TYPE): Description
        pricedata (TYPE): Description
        expdata (TYPE): Description
        pf (TYPE): Description
        hedges (TYPE): Description
        rollover_dates (TYPE): Description
        end_date (None, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        TYPE: Description

    """
    t = time.clock()
    # rollover_dates = get_rollover_dates(pricedata)
    grosspnl = 0
    netpnl = 0

    gross_daily_values = []
    gross_cumul_values = []
    net_daily_values = []
    net_cumul_values = []

    loglist = []

    date_range = sorted(voldata.value_date.unique())  # [1:]
    print('dates: ', pd.to_datetime(date_range))
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
        if len(pf.OTC_options) == 0 and len(pf.OTC_futures) == 0 and not pf.empty():
            print('ALL OTC OPTIONS HAVE EXPIRED. ENDING SIMULATION...')
            break
        # if end_date is inputted, check to see if the current date exceeds
        # end_date
        if end_date:
            if date >= end_date:
                print('REACHED END OF SIMULATION.')
                break
        # try to get next date
        try:
            next_date = date_range[i+1]
        except IndexError:
            next_date = None
        # try to get previous date
        try:
            prev_date = date_range[i-1]
        except IndexError:
            prev_date = None

        # isolate data relevant for this day.
        date = pd.to_datetime(date)
        print('##################### date: ', date, '################')

        # filter data specific to the current day of the simulation.
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]
        print('Portfolio before any ops: ', pf)
        print('SOD Vega Pos: ', pf.net_vega_pos())

    # Step 3: Feed data into the portfolio.
        raw_change, pf, broken = feed_data(
            vdf, pdf, pf, rollover_dates, prev_date, brokerage=brokerage, slippage=slippage)

    # Step 4: Compute pnl for the day
        updated_val = pf.compute_value()
        dailypnl = updated_val - init_val if init_val != 0 else 0
        # print('Vegas: ', [(str(op), op.vega) for op in pf.OTC_options])

    # Step 5: Apply signal
        if signals is not None:
            roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
                'delta'][i][0] == 'roll'][0]
            pf = apply_signal(pf, vdf, pdf, signals,
                              date, next_date, roll_cond, strat='filo')

    # Step 6: Hedge.
        # cost = 0
        pf, counters, cost, roll_hedged = rebalance(
            vdf, pdf, pf, hedges, counters, brokerage=brokerage, slippage=slippage)

    # Step 7: Subtract brokerage/slippage costs from rebalancing. Append to
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

    # Step 8: Initialize init_val to be used in the next loop.
        init_val = pf.compute_value()
        print('[13]  EOD PORTFOLIO: ', pf)

    # Step 9: computing stuff to be logged
        lst = [date, dailypnl, dailypnl-cost, grosspnl, netpnl, roll_hedged]
        dic = pf.get_net_greeks()
        call_vega, put_vega = 0, 0
        cols = ['value_date', 'eod_pnl_gross', 'eod_pnl_net', 'cu_pnl_gross',
                'cu_pnl_net', 'delta_rolled', 'pdt', 'month', 'delta', 'gamma',
                'theta', 'vega', 'net_call_vega', 'net_put_vega', 'net_ft_pos']
        for pdt in dic:
            for mth in dic[pdt]:
                # getting net greeks
                delta, gamma, theta, vega = dic[pdt][mth]
                ops = pf.OTC[pdt][mth][0]
                ftpos = 0
                # net future position
                ft = pf.hedges[pdt][mth][1]
                for f in ft:
                    val = f.lots if not f.shorted else -f.lots
                    ftpos += val
                calls = [op for op in ops if op.char == 'call']
                puts = [op for op in ops if op.char == 'put']
                # net call vega, net put vega
                call_vega = sum([op.vega for op in calls])
                put_vega = sum([op.vega for op in puts])
                lst.extend([pdt, mth, delta, gamma, theta,
                            vega, call_vega, put_vega, ftpos])
                dic = OrderedDict(zip(cols, lst))
                loglist.append(dic)

    # Step 10: Decrement timestep after all computation steps

        # calculate number of days to step
        num_days = 0 if next_date is None else (
            pd.Timestamp(next_date) - pd.Timestamp(date)).days
        print('TIME STEPPING: ', str(num_days) + ' days')

        pf.timestep(num_days * timestep)

        print('pf after timestep: ', pf)
        for op in pf.OTC_options:
            print('Option: ', op)
            print('vol: ', op.vol)
            print('price: ', op.underlying.get_price())

        print('Net vega pos: ', pf.net_vega_pos())

    # Step 11: Plotting results/data viz

    # appending 25d vol changes and price changes
    signals['underlying_id'] = signals.pdt + '  ' + signals.ftmth
    signals['vol_id'] = signals.pdt + '  ' + \
        signals.opmth + '.' + signals.ftmth

    df = pd.merge(signals, pricedata[['value_date', 'underlying_id', 'settle_value']], on=[
                  'value_date', 'underlying_id'])
    df = df.drop_duplicates()
    df['price_change'] = df.settle_value.shift(-1) - df.settle_value
    df['25d_call_change'] = df.delta_call_25_a.shift(-1) - df.delta_call_25_a
    df['25d_put_change'] = df.delta_put_25_b.shift(-1) - df.delta_put_25_b
    df['25d_call_change'] = df['25d_call_change'].shift(1)
    df['25d_put_change'] = df['25d_put_change'].shift(1)

    df = df.fillna(0)
    log = pd.DataFrame(loglist)

    # merge the log and the vol/price changes
    log = pd.merge(log, df[['value_date', 'vol_id', 'price_change',
                            '25d_call_change', '25d_put_change']], on=['value_date'])

    log.to_csv('log.csv', index=False)

    # plotting greeks
    plt.figure()
    plt.plot(log.value_date, log.delta, c='c', label='delta')
    plt.plot(log.value_date, log.gamma, c='g', label='gamma')
    plt.plot(log.value_date, log.theta, c='r', label='theta')
    plt.plot(log.value_date, log.vega, c='k', label='vega')
    plt.legend()
    plt.show()

    elapsed = time.clock() - t

    print('Time elapsed: ', elapsed)
    print('##################### PNL: #####################')
    print('gross pnl: ', grosspnl)
    print('net pnl: ', netpnl)
    print('################# Portfolio: ###################')
    print(pf)

    return grosspnl, netpnl, pf, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log


##########################################################################
##########################################################################
##########################################################################


##########################################################################
########################## Helper functions ##############################
##########################################################################


def feed_data(voldf, pdf, pf, dic, prev_date, brokerage=None, slippage=None):
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
        prev_date (TYPE): Description
        brokerage (int, optional): brokerage fees per lot.
        slippage (int, optional): slippage cost

    Returns:
        tuple: change in value, updated portfolio object, and whether or not there is missing data.

    Raises:
        ValueError: Raised if voldf is empty.
    """
    broken = False

    # sanity checks
    if pf.empty():
        return 0, pf, False

    if voldf.empty:
        raise ValueError('vol df is empty!')

    date = voldf.value_date.unique()[0]
    raw_diff = 0

    # 1) initial value of the portfolio before updates, and handling exercises
    # before feeding data.
    prev_val = pf.compute_value()
    expenditure, pf = handle_exercise(pf, brokerage, slippage)
    raw_diff += expenditure

    curr_date = pd.to_datetime(pdf.value_date.unique()[0])
    # print('curr_date: ', curr_date)
    prev_date = pd.to_datetime(prev_date)

    # 2) Check for rollovers and expiries rollovers
    for product in dic:
        ro_dates = dic[product]
        # rollover date for this particular product
        if prev_date is None:
            if date in ro_dates:
                print('ROLLOVER DETECTED; DECREMENTING ORDERING')
                pf.decrement_ordering(product, 1)
        else:
            relevant_rollovers = [x for x in ro_dates if (
                x.month == prev_date.month) and (x.year == prev_date.year)]
            for date in relevant_rollovers:
                # rollover in between dates
                if date.day < curr_date.day and date.day > prev_date.day:
                    print('ROLLOVER DETECTED; DECREMENTING ORDERING')
                    pf.decrement_ordering(product, 1)

    # expiries; also removes options for which ordering = 0
    pf.remove_expired()

    # print('pf after rollovers and expiries: ', pf)

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
            ticksize = multipliers[op.get_product()][-2]
            # get strike corresponding to closest available ticksize.
            # print('feed_data - ticksize: ', ticksize, op.get_product())
            strike = round(round(strike/ticksize) * ticksize, 2)
            try:
                vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                            (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                df_tau = min(val.tau, key=lambda x: abs(x-tau))
                val = val[val.tau == df_tau].settle_vol.values[0]
                op.update_greeks(vol=val)
                # print('UPDATED - new vol: ', val)
            except (IndexError, ValueError):
                print('### VOLATILITY DATA MISSING ###')
                print('product: ', product)
                print('strike: ', strike)
                print('order: ', order)
                print('vid: ', vid)
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
                uid = ft.get_product() + '  ' + ft.get_month()
                val = pdf[(pdf.pdt == pdt) &
                          (pdf.underlying_id == uid)].settle_value.values[0]
                # print('UPDATED - new price: ', val)
                ft.update_price(val)

            # index error would occur only if data is missing.
            except IndexError:
                print('###### PRICE DATA MISSING #######')
                print('ordering: ', ordering)
                print('uid: ', uid)
                print('pdt: ', pdt)
                print('debug 1: ', pdf[(pdf.pdt == pdt) &
                                       (pdf.order == ordering)])
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
    """Handles option exercise, as well as bullet vs daily payoff.

    Args:
        pf (Portfolio object): the portfolio being run through the simulator.
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        tuple: the combined PnL from exercising (if appropriate) and daily/bullet payoffs, as well as the updated portfolio.

    Notes on implementation:
        1) options are exercised if less than or equal to 2 days to maturity, and option is in the money. Futures obtained are immediately sold and profit is locked in.

    """
    if pf.empty():
        return 0, pf

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
    """Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.

    Notes:
        1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed.
        2) hedging delta done by shorting/buying -delta * lots futures.
        3)

    Args:
        vdf (pandas dataframe): Dataframe of volatilities
        pdf (pandas dataframe): Dataframe of prices
        pf (object): portfolio object
        hedges (dict): Dictionary of hedging conditions
        counters (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

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
    droll = None
    # updating counters
    if pf.empty():
        return pf, counters, 0, False

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
    droll = not roll_hedged

    cost = 0

    if roll_hedged:
        print('deltas within bounds. skipping roll_hedging')

    if not roll_hedged:
        print(' ++ ROLL HEDGING REQUIRED ++ ')
        for op in pf.OTC_options:
            print('delta: ', abs(op.delta/op.lots))
        roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
            'delta'][i][0] == 'roll'][0]
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
                        'delta'][i][0] == 'static']
                    if delta_cond:
                        delta_cond = delta_cond[0][1]
                    # print('delta conds: ', delta_cond)
                        pf, dhedges, fees = hedge_delta(
                            delta_cond, vdf, pdf, pf, month, product, ordering, brokerage=brokerage, slippage=slippage)
                        cost += fees
                    else:
                        print('no delta hedging specifications found')
        done_hedging = hedges_satisfied(pf, hedges)

    return (pf, counters, cost, droll)


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
        list: inputs required to construct atm straddles.
    """
    net_greeks = pf.get_net_greeks()
    greeks = net_greeks[product][month]
    if flag == 'gamma':
        greek = greeks[1]
    elif flag == 'vega':
        greek = greeks[3]
    elif flag == 'theta':
        greek = greeks[2]
    # grabbing bound
    relevant_conds = [hedges[flag][i] for i in range(len(hedges[flag])) if hedges[
        flag][i][0] == 'bound'][0]
    bound = relevant_conds[1]
    # print('relevant_conds: ', relevant_conds)
    # print('bound: ', bound)
    uid = product + '  ' + month
    # relevant data for constructing Option and Future objects.
    price = pdf[(pdf.pdt == product) &
                (pdf.underlying_id == uid)].settle_value.values[0]
    ticksize = multipliers[product][-2]
    print('gen_hedge_inputs - ticksize: ', ticksize, product)
    k = round(round(price / ticksize) * ticksize, 2)
    # print('[8]  STRIKE: ', k)
    cvol = vdf[(vdf.pdt == product) &
               (vdf.call_put_id == 'C') &
               (vdf.underlying_id == uid) &
               (vdf.strike == k)].settle_vol.values[0]

    pvol = vdf[(vdf.pdt == product) &
               (vdf.call_put_id == 'P') &
               (vdf.underlying_id == uid) &
               (vdf.strike == k)].settle_vol.values[0]

    # if no tau provided, default to using same-month atm straddles.
    tau_vals = vdf[(vdf.pdt == product) &
                   (vdf.call_put_id == 'P') &
                   (vdf.underlying_id == uid) &
                   (vdf.strike == k)].tau

    if len(relevant_conds) == 3:
        tau = max(tau_vals)

    else:
        target = relevant_conds[3]
        tau = min(tau_vals, key=lambda x: abs(x-target))

    underlying = Future(month, price, product, ordering=ordering)

    return [price, k, cvol, pvol, tau, underlying, greek, bound, ordering]


def hedge(pf, inputs, product, month, flag, brokerage=None, slippage=None):
    """
    This function does the following:
        1) constructs atm straddles with the inputs from _inputs_
        2) hedges the greek in question (specified by flag) with the straddles.

    Args:
        pf (portfolio object): portfolio object
        inputs (list): list of inputs reqd to construct straddle objects
        product (string): the product being hedged
        month (string): month being hedged
        flag (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

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
    print('>>>>>>>>>>> hedging ' + product + ' ' +
          month + ' ' + flag + ' <<<<<<<<<<<<<')
    if greek > upper or greek < lower:
        # gamma hedging logic.

        # print(product + ' ' + month + ' ' + flag + ' hedging')
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
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<')
    return pf, fees   # [callop, putop]


# NOTE: assuming that future can be found with the exact number of
# lots to delta hedge.
def hedge_delta(cond, vdf, pdf, pf, month, product, ordering, brokerage=None, slippage=None):
    """Helper function that implements delta hedging. General idea is to zero out delta at the end of the day by buying/selling -delta * lots futures. Returns expenditure (which is negative if shorting and postive if purchasing delta) and the updated portfolio object.

    Args:
        cond (string): condition for delta hedging
        vdf (dataframe): Dataframe of volatilities
        pdf (dataframe): Dataframe of prices
        pf (portfolio): portfolio object specified by portfolio_specs.txt
        month (str): month of underlying future.
        product (TYPE): Description
        ordering (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        tuple: hedging costs and final portfolio with hedges added.

    Deleted Parameters:
        net (list): greeks associated with net_greeks[product][month]

    """
    # print('cond: ', cond)
    print('>>>>>>>>>>> delta hedging: ', product +
          '  ' + month + ' <<<<<<<<<<<<<')
    uid = product + '  ' + month
    future_price = pdf[(pdf.pdt == product) &
                       (pdf.underlying_id == uid)].settle_value.values[0]

    net_greeks = pf.get_net_greeks()
    fees = 0
    # print('cond: ', cond)
    if cond == 'zero':
        # flag that indicates delta hedging.
        vals = net_greeks[product][month]
        delta = vals[0]
        shorted = True if delta > 0 else False
        num_lots_needed = abs(round(delta))
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
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<')
    return pf, ft, fees


#####################################################################
####################### Skew-Related Functions ######################
#####################################################################

def hedge_delta_roll(pf, roll_cond, pdf, brokerage=None, slippage=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta exceeds certain bounds.

    Args:
        pf (object): Portfolio being hedged
        roll_cond (list): list of the form ['roll', value, frequency, bound]
        vdf (pandas df): volatility data frame containing information for current day
        pdf (pandas df): price dataframe containing information for current day
        brokerage (int, optional): brokerage fees per lot
        slippage (int, optional): slippage loss

    Returns:
        tuple: updated portfolio and cost of purchasing/selling options.
    """
    # print('hedge_delta_roll conds: ', roll_cond)

    cost = 0
    roll_val, bounds = roll_cond[1], np.array(roll_cond[3])/100

    # pfx = copy.deepcopy(pf)
    # print('PFX: ', pfx)

    toberemoved = []
    tobeadded = []

    for op in pf.OTC_options:
        delta = abs(op.delta/op.lots)
        # case: delta not in bounds. roll delta.
        if delta > bounds[1] or delta < bounds[0]:
            # get strike corresponding to delta
            print('delta not in bounds: ', op, delta)
            cpi = 'C' if op.char == 'call' else 'P'
            # get the vol from the vol_by_delta part of pdf
            col = str(int(roll_val)) + 'd'
            try:
                vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()
                vol = pdf.loc[(pdf.call_put_id == cpi) &
                              (pdf.vol_id == vid) &
                              (pdf.pdt == op.get_product()) &
                              (np.isclose(pdf.tau, op.tau)), col].values[0]
                # print('vol found')
            except IndexError:
                print('[ERROR] -', cpi, vid, op.get_product(), op.tau)
                print('[ERROR] - tau: ', op.tau)
                vol = op.vol

            strike = compute_strike_from_delta(
                op, delta1=roll_val/100, vol=vol)
            # print('roll_hedging - newop tau: ', op.tau)
            newop = Option(strike, op.tau, op.char, vol, op.underlying,
                           op.payoff, op.shorted, op.month, direc=op.direc,
                           barrier=op.barrier, lots=op.lots, bullet=op.bullet,
                           ki=op.ki, ko=op.ko, rebate=op.rebate,
                           ordering=op.ordering, settlement=op.settlement)

            toberemoved.append(op)
            tobeadded.append(newop)

            # handle expenses: brokerage and old op price - new op price
            val = (op.compute_price() - newop.compute_price())
            cost += val

            if brokerage:
                cost += (brokerage * (op.lots + newop.lots))
    for op in tobeadded:
        print('roll hedging - op added: ', op)
        print('roll hedging - op delta: ', abs(op.delta/op.lots))
    pf.remove_security(toberemoved, 'OTC')
    pf.add_security(tobeadded, 'OTC')
    # print('final portfolio: ', pf)
    # print('initial copy (pfx): ', pfx)
    return pf, cost


def apply_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat='dist', tol=1000):
    """Applies the signal generated by the recommendation program to the portfolio.
    Args:
        pf (object): portfolio
        vdf (pandas dataframe): dataframe of volatilities
        pdf (pandas dataframe): dataframe of prices
        signals (pandas dataframe): signals
        date (pandas Timestamp): current date
        next_date (pandas Timestamp): next date in simulation
        roll_cond (list): list of delta_roll conditions
        strat (str, optional): Description

    Returns:
        portfolio object: the portfolio with the requisite changes applied.
    """

    # identifying relevant columns; hardcoded due to formatting, change if
    # necessary.
    cols = ['delta_call_25_a', 'delta_put_25_b',
            'signal', 'opmth', 'ftmth', 'pdt', 'lots', 'vega']
    # getting inputs from signals dataframe
    print('next_date: ', next_date)

    if next_date is None:
        print('reached end of signal period')
        return pf

    cvol, pvol, sig, opmth, ftmth, pdt, lots, vega_req = \
        signals.loc[signals.value_date == next_date, cols].values[0]

    inputs = [cvol, pvol, sig, opmth, ftmth, pdt, lots, vega_req]

    print('________APPLYING SIGNAL_______: ', sig)
    ret = None
    next_date = pd.to_datetime(next_date)

    # Case 1: flatten signal
    if sig == 0:
        ret = Portfolio()
    # Case 2: Nonzero signal
    else:
        dval = roll_cond[1]/100
        net_call_vega, net_put_vega = pf.net_vega_pos()
        print('net call/put vega: ', net_call_vega, net_put_vega)
        target_call_vega, target_put_vega = vega_req * sig,  -vega_req * sig
        print('target call/put vega: ', target_call_vega, target_put_vega)
        handle_calls, handle_puts = True, True
        # Case 2-1: Adding to empty portfolio.
        if net_call_vega == 0 and net_put_vega == 0:
            print('empty portfolio; adding skews')
            pf = add_skew(pf, vdf, pdf, inputs, date, dval)
            ret = pf
        # Case 2-2: Adding to nonempty portfolio
        else:
            # check if call vegas are within bounds
            if abs(net_call_vega - target_call_vega) < tol:
                print('call vega within bounds. Skipping handling.')
                handle_calls = False
            # check if put vegas are within bounds
            if abs(net_put_vega - target_put_vega) < tol:
                print('put vega within bounds. Skipping handling.')
                handle_puts = False

            if handle_calls:
                calls = [op for op in pf.OTC_options if op.char == 'call']
                # update_pos(char, target_vega, curr_vega, dval, ops, pf, strat, tol, vdf, pdf, inputs, date)
                pf = update_pos('call', target_call_vega,
                                net_call_vega, dval, calls, pf, strat, tol, vdf, pdf, inputs, date)

            if handle_puts:
                puts = [op for op in pf.OTC_options if op.char == 'put']
                pf = update_pos('put', target_put_vega,
                                net_put_vega, dval, puts, pf, strat, tol, vdf, pdf, inputs, date)

        ret = pf

    print('________SIGNAL APPLIED _________')
    return ret


###############################################################################
############################ Helper functions #################################
###############################################################################

def generate_skew_op(char, vdf, pdf, inputs, date, dval):
    """Helper function that generates the options comprising the skew position, based on inputs passed in.
    Used when dealing with individual legs of the skew position, not when dealing with adding skews as a whole. 

    Args:
        char (str): 'call' or 'put'. 
        vdf (pd dataframe): dataframe of vols
        pdf (pd dataframe): dataframe of prices 
        inputs (list): 
        date (pd Timestamp): Description
        dval (double): Description
    """
    # unpack inputs
    vol, sig, opmth, ftmth, pdt, vega_req = inputs
    print('generate_skew_ops inputs: ')
    print('vol: ', vol)
    print('vega_reqd: ', vega_req)
    print('sig: ', sig)
    # determining if options are to be shorted or not
    if sig < 0 and char == 'call':
        shorted = True
    elif sig < 0 and char == 'put':
        shorted = False
    elif sig > 0 and char == 'call':
        shorted = False
    elif sig > 0 and char == 'put':
        shorted = True
    print('char: ', char)
    print('shorted: ', shorted)

    # num_skews = abs(sig)
    vol = vol/100

    # computing ordering
    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])

    # create the underlying future
    uid = pdt + '  ' + ftmth

    try:
        ftprice = pdf[(pdf.value_date == date) &
                      (pdf.underlying_id == uid)].settle_value.values[0]
    except IndexError:
        print('inputs: ', date, uid)

    ft = Future(ftmth, ftprice, pdt, shorted=False, ordering=order)

    # create the options; long one dval call, short on dval put
    vol_id = pdt + '  ' + opmth + '.' + ftmth

    # computing tau
    tau = vdf[(vdf.value_date == date) &
              (vdf.vol_id == vol_id)].tau.values[0]

    # computing strikes
    strike = compute_strike_from_delta(
        None, delta1=dval, vol=vol, s=ftprice, tau=tau, char=char, pdt=pdt)

    op = Option(strike, tau, char, vol, ft, 'amer',
                shorted, opmth, ordering=order)

    pnl_mult = multipliers[pdt][-1]
    op_vega = (op.vega * 100) / (op.lots * pnl_mult)
    print(char + ' vega: ', op_vega)
    # calculate lots required for requisite vega specified; done according
    # to callop.
    lots_req = round((abs(vega_req) * 100) /
                     abs(op_vega * pnl_mult))
    print('generate_skew_op - lots required: ', lots_req)
    op = Option(strike, tau, char, vol, ft, 'amer',
                shorted, opmth, lots=lots_req, ordering=order)
    print('generate_skew_op - total vega: ', op.vega)

    return op


def add_skew(pf, vdf, pdf, inputs, date, dval):
    """Helper method that adds a skew position based on the inputs provided.

    Args:
        pf (TYPE): portfolio being hedged. 
        vdf (TYPE): vol dataframe with info pertaining to date. 
        pdf (TYPE): price dataframe with info pertaining to date.
        inputs (TYPE): list corresponding to values in signals.loc[date = next_date]
        date (TYPE): current date in the simulation 
        dval (TYPE): the delta value of the skew desired. 
    """

    # unpack inputs
    cvol, pvol, sig, opmth, ftmth, pdt, lots, vega_req = inputs
    print('add_skew inputs: ')
    print('cvol: ', cvol)
    print('pvol: ', pvol)
    print('sig: ', sig)
    print('vega req: ', vega_req)

    # determining if options are to be shorted or not
    shorted = True if sig < 0 else False
    num_skews = abs(sig)
    cvol, pvol = cvol/100, pvol/100
    # computing ordering
    curr_mth = date.month
    curr_mth_sym = month_to_sym[curr_mth]
    curr_yr = date.year % (2000 + decade)
    curr_sym = curr_mth_sym + str(curr_yr)
    order = find_cdist(curr_sym, ftmth, contract_mths[pdt])

    # create the underlying future
    uid = pdt + '  ' + ftmth

    try:
        ftprice = pdf[(pdf.value_date == date) &
                      (pdf.underlying_id == uid)].settle_value.values[0]
    except IndexError:
        print('inputs: ', date, uid)

    ft = Future(ftmth, ftprice, pdt, shorted=False,
                lots=lots, ordering=order)

    # create the options; long one dval call, short on dval put
    vol_id = pdt + '  ' + opmth + '.' + ftmth

    # computing tau
    tau = vdf[(vdf.value_date == date) &
              (vdf.vol_id == vol_id)].tau.values[0]

    # computing strikes
    c_strike = compute_strike_from_delta(
        None, delta1=dval, vol=cvol, s=ftprice, tau=tau, char='call', pdt=pdt)
    p_strike = compute_strike_from_delta(
        None, delta1=dval, vol=pvol, s=ftprice, tau=tau, char='put', pdt=pdt)

    # creating placeholder options objects for computation purposes
    callop = Option(c_strike, tau, 'call', cvol, ft, 'amer',
                    shorted, opmth, lots=lots, ordering=order)
    putop = Option(p_strike, tau, 'put', pvol, ft, 'amer',
                   not shorted, opmth, lots=lots, ordering=order)

    tobeadded = []

    pnl_mult = multipliers[pdt][-1]
    op_vega = (callop.vega * 100) / (callop.lots * pnl_mult)
    print('call vega: ', op_vega)
    # calculate lots required for requisite vega specified; done according
    # to callop.
    lots_req = round((abs(vega_req * num_skews) * 100) /
                     abs(op_vega * pnl_mult))

    # creating and appending relevant options.
    callop = Option(c_strike, tau, 'call', cvol, ft, 'amer',
                    shorted, opmth, lots=lots_req, ordering=order)
    putop = Option(p_strike, tau, 'put', pvol, ft, 'amer',
                   not shorted, opmth, lots=lots_req, ordering=order)
    print('vegas: ', callop.vega, putop.vega)
    pf.add_security([callop, putop], 'OTC')
    tobeadded.extend([callop, putop])

    # debug statement.

    for op in tobeadded:
        print('added op deltas: ', op, abs(op.delta/op.lots))
    tobeadded.clear()

    return pf


def update_pos(char, target_vega, curr_vega, dval, ops, pf, strat, tol, vdf, pdf, inputs, date):
    """
    Helper function that updates individual legs of the skew position in question. Two major cases are handled:
    1) Increasing current position. I.e. negative to negative, positive to positive, negative to positive, positive to negative. 

    2) Liquidating positions. Negative to larger negative (i.e. -30,000 -> -10,000), positive to smaller positive (30,000 -> 10,000)

    Notes:
        - for readability: vpl = vega per lot.

    Args:
        char (TYPE): leg of the skew being handled. 
        target_vega (TYPE): target vega for this leg. 
        curr_vega (TYPE): current vega of this leg. 
        dval (TYPE): delta value required. 
        ops (TYPE): relevant options; when handling call leg, all call options passed in. vice versa for puts. 
        pf (TYPE): portfolio object being subjected to the signal
        strat (TYPE): the regime used to determine which skew positions are liquidated first.
        tol (TYPE): tolerance value within which portfolio is left alone. 
        vdf (TYPE): dataframe of volatilities
        pdf (TYPE): dataframe of prices
        inputs (TYPE): row corresponding to signal.loc[index]
        date (TYPE): current date in the simulation. 

    Returns:
        portfolio: the updated portfolio with the appropriate equivalent position liquidated.

    """

    cvol, pvol, sig, opmth, ftmth, pdt, lots = inputs[:-1]

    vol = cvol if char == 'call' else pvol

    print('HANDLING ' + char.upper() + ' LEG')
    print('target: ', target_vega)
    print('current: ', curr_vega)

    toberemoved, tobeadded = [], []
    # shorted = -1 if curr_vega > target_vega else 1
    buy = True if target_vega > curr_vega else False
    vega_req = abs(target_vega - curr_vega)

    # Case 1: need to buy vega.
    if buy:
        # Case 1-1: negative to positive pos. (e.g. -30,000 -> 10,000)
        if curr_vega < 0 and target_vega > 0:
            print(char.upper() + ': flipping leg from neg to pos')
            # reset leg to 0, then add required vega.
            # remove all short options on this leg.
            shorts = [op for op in ops if op.shorted]
            pf.remove_security(shorts.copy(), 'OTC')
            index = 0 if char == 'call' else 1
            # get residual vega on this leg after removing all shorts.
            curr_vega = pf.net_vega_pos()[index]
            vega_req = target_vega - curr_vega
            # create input list
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            # generate option
            op = generate_skew_op(char, vdf, pdf, op_inputs, date, dval)
            tobeadded.append(op)
            # case 1-2: nonnegative to positive pos (e.g. 10,000 -> 20,000)
        elif curr_vega > 0 and target_vega > 0:
            print(char.upper() + ' - increasing long pos')
            vega_req = target_vega - curr_vega
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            op = generate_skew_op(char, vdf, pdf, op_inputs, date, dval)
            tobeadded.append(op)
        elif curr_vega < 0 and target_vega < 0:
            print('liquidating short positions - buying ' + char + ' leg')
            shortops = [op for op in ops if op.shorted]
            resid_vega = abs(curr_vega - target_vega)
            # print('resid_vega: ', resid_vega)
            pf = liquidate_pos(char, resid_vega, shortops, pf, strat, dval)

    # Case 2: Need to sell vega from this leg.
    else:
        # negative to negative; add 25 delta shorts
        if curr_vega < 0 and target_vega < 0:
            print(char.upper() + ' - increasing short pos')
            vega_req = target_vega - curr_vega
            op_inputs = op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            op = generate_skew_op(char, vdf, pdf, op_inputs, date, dval)
            tobeadded.append(op)
        # positive to negative - same as negative to positive.
        elif curr_vega > 0 and target_vega < 0:
            print(char.upper() + ': flipping leg from pos to neg')
            # remove all long options on this leg, find difference, then add
            # requisite vega.
            longs = [op for op in ops if not op.shorted]
            pf.remove_security(longs.copy(), 'OTC')
            index = 0 if char == 'call' else 1
            curr_vega = pf.net_vega_pos()[index]
            vega_req = target_vega - curr_vega
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            # generate option
            op = generate_skew_op(char, vdf, pdf, op_inputs, date, dval)
            tobeadded.append(op)

        elif curr_vega > 0 and target_vega > 0:
            print('liquidating long positions - selling ' + char + ' leg')
            longops = [op for op in ops if not op.shorted]
            resid_vega = curr_vega - target_vega
            pf = liquidate_pos(char, resid_vega, longops, pf, strat, dval)

    # add any securities that need adding.
    pf.add_security(tobeadded, 'OTC')

    # update any lot size changes
    pf.update_sec_by_month(False, 'OTC', update=True)

    # debug statement
    for op in toberemoved:
        print('op removed deltas: ', op, abs(op.delta/op.lots))
    toberemoved.clear()
    # print('pf afte: ', pf)
    return pf


def liquidate_pos(char, resid_vega, ops, pf, strat, dval):
    """Buys/Sells (vega_req * num_req) worth of dval skew composites. For example 
       if vega = 10000, num_skews =1 and dval = 25, then the function figures out 
       how to liquidate 10,000 vega worth of that day's 25 Delta skew positions ON 
       EACH LEG (i.e. a Long 25 Delta Call and a Short 25 Delta Put) from the 
       currently held position.

    Does so by selecting  a skew position to liquidate based on strat;
    currently implemented strategies for liquidation are 'dist' (the options
    selected are furthest away from dval) and 'filo' (first in last out),
    computing the vega per lot of that skew position, and liquidating the
    required number of lots. If the number of lots contained in the selected
    skew position is smaller than the number of lots required, the skew
    position is completely closed out (i.e. shorts are bought back, longs
    are sold) and the leftover vega is liquidated using the next highest
    skew pos, selected according to strat.

    If all options of a given type have been spent in this manner and
    liquidation is still not complete, the algorithm simply skips to handle
    options of the next type. In the pathological case where the entire
    portfolio's contents are insufficient to meet the liquidation condition,
    the entire portfolio is closed, and remaining deltas from hedging are
    closed off.

    In practice, since we're interested in the call-side (long skew), lots
    required for the requisite vega is determined by the call option. Put
    option is instantiated with exactly the same number of lots, which could
    lead to a slight vega differential (currently being ignored, but might
    need to be addressed in the future).


    Notes:
        - for readability: vpl = vega per lot.
    Args:
        vega_req (TYPE): total amount of vega to be liquidated.
        num_skews (TYPE): number of buy/sell signals on this day. total vega to be liquidated would be vega_req * num_skews
        dval (TYPE): delta value of the skew pos we want to run.
        calls (TYPE): list of call options selected according to which liquidation scenario is in effect (i.e. liquidating shorts with a buy signal, or liquidating longs with a sell signal)
        puts (TYPE): list of put options selected according to which liquidation scenario is in effect.
        pf (TYPE): portfolio object being subjected to the signal.
        strat (TYPE): the regime used to determine which skew positions are liquidated first.

    Returns:
        portfolio: the updated portfolio with the appropriate equivalent position liquidated.
    """

    toberemoved = []

    # handling puts
    print('HANDLING ' + char.upper())
    # print('resid_vega: ', resid_vega)

    while resid_vega > 0:
        # print('residual vega: ', resid_vega)
        if strat == 'dist':
            print('selecting skew acc to dist')
            print('portfolio at loop start: ', pf)
            ops = sorted(ops, key=lambda x: abs(
                abs(x.delta/x.lots) - dval))
            max_op = ops[-1] if ops else None

            # print('putops: ', [str(p) for p in put_ops])
        elif strat == 'filo':
            print('selecting skew acc to filo')
            print('portfolio at loop start: ', pf)
            max_op = ops[0] if ops else None
            # print('op selected: ', max_op)

        if max_op is not None:
            print('op selected: ', max_op)
            # handle puts
            vpl = abs(max_op.vega/max_op.lots)
            print('puts vega per lot: ', vpl)
            lots_req = round(abs(resid_vega / vpl))
            print('put lots req: ', lots_req)

            # Case 1: lots required < lots available.
            if lots_req < max_op.lots:
                print('l_puts: lots available.')
                resid_vega = 0
                newlots = max_op.lots - lots_req
                max_op.update_lots(newlots)
                print('puts - new lots: ', newlots)
                # break

            # Case 2: lots required > lots available.
            else:
                print('l_puts: lots unavailable.')
                resid_vega -= max_op.lots * vpl
                toberemoved.append(max_op)
                ops.remove(max_op)
        else:
            print(
                'cannot liquidate existing positions any further. continuing to handle calls..')
            break

    # update any lot size changes
    pf.update_sec_by_month(False, 'OTC', update=True)

    # remove securities flagged for removal
    pf.remove_security(toberemoved, 'OTC')

    # debug statement
    for op in toberemoved:
        print('op removed deltas: ', op, abs(op.delta/op.lots))
    toberemoved.clear()
    print('pf after liquidation: ', pf)
    return pf


# TODO: turn this into a dictionary of prices.
def close_out_deltas(pf, price):
    """Checks to see if the portfolio is closed out, but with residual deltas. Closes out all remaining
    future positions, resulting in an empty portfolio. 

    Args:
        pf (portfolio object): Description
        price (float): price of the underlying commodity 

    Returns
        tuple: updated portfolio, and cost of closing out deltas
    """
    print('closing out deltas')
    cost = 0
    if (not pf.OTC_options) and (not pf.hedge_options):
        if pf.hedge_futures:
            # close off all futures while keeping track of the cost of doing so
            for ft in pf.hedge_futures:
                if ft.shorted:
                    cost += price
                else:
                    cost -= price
            pf.remove_security(pf.hedge_futures.copy(), 'hedge')

        if pf.OTC_futures:
            for ft in pf.OTC_futures:
                if ft.shorted:
                    cost += price
                else:
                    cost -= price
            pf.remove_security(pf.OTC_futures.copy(), 'OTC')

    return pf


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
            rollbounds = np.array(cond[3])/100
            found = True
    # if roll conditions actually exist, proceed.
    if found:
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

    # fix portfolio start date #
    start_date = None

    # filepath to portfolio specs. #
    # specpath = gv.portfolio_path
    specpath = 'specs.csv'
    # specpath = 'datasets/bo_portfolio_specs.csv'

    # fix portfolio internal date #
    # internal_date = pd.Timestamp('2014-08-21')
    # internal_date = gv.internal_date

    # fix end date of simulation #
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
    volpath, pricepath, exppath, rollpath, sigpath = gv.small_final_vol_path,\
        gv.small_final_price_path,\
        gv.small_final_exp_path,\
        gv.small_cleaned_price,\
        gv.signal_path

    ### full data ###
    # volpath, pricepath, exppath, rollpath = gv.final_vol_path,
    # gv.final_price_path, gv.final_exp_path, gv.cleaned_price

    writeflag = 'small' if 'small' in volpath else 'full'
    print('writeflag: ', writeflag)

    if os.path.exists(volpath) and os.path.exists(pricepath)\
            and os.path.exists(exppath)\
            and os.path.exists(rollpath) \
            and os.path.exists(sigpath):
        print('####################################################')
        print('DATA EXISTS. READING IN... [2/7]')
        print('datasets listed below: ')
        print('voldata : ', volpath)
        print('pricepath: ', pricepath)
        print('exppath: ', exppath)
        print('rollpath: ', rollpath)
        print('signals: ', sigpath)
        print('####################################################')

        vdf, pdf, edf = pd.read_csv(volpath), pd.read_csv(
            pricepath), pd.read_csv(exppath)
        rolldf = pd.read_csv(rollpath)
        signals = pd.read_csv(sigpath)

        # sorting out date types
        signals.value_date = pd.to_datetime(signals.value_date)
        rolldf.value_date, rolldf.expdate = pd.to_datetime(
            rolldf.value_date), pd.to_datetime(rolldf.expdate)
        vdf.value_date = pd.to_datetime(vdf.value_date)
        pdf.value_date = pd.to_datetime(pdf.value_date)
        edf.expiry_date = pd.to_datetime(edf.expiry_date)

    else:
        print('####################################################')
        print('current_dir: ', os.getcwd())
        print('paths inputted: ', volpath,
              pricepath, exppath, rollpath, sigpath)
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

        signals = pd.read_csv(sigpath)
        signals.value_date = pd.to_datetime(signals.value_date)

        vdf, pdf, edf, rolldf = read_data(
            volpath, pricepath, epath, specpath, signals=signals, test=False,  writeflag=writeflag, start_date=start_date)

        signals = pd.read_csv(sigpath)
        signals.value_date = pd.to_datetime(signals.value_date)

    # vdf, pdf = match_to_signals(vdf, pdf, signals)

    print('DATA READ-IN COMPLETE. SANITY CHECKING... [3/7]')
    print('READ-IN RUNTIME: ', time.clock() - t)

    ######################### check sanity of data ###########################

    # handle data types.
    vdates = pd.to_datetime(sorted(vdf.value_date.unique()))
    pdates = pd.to_datetime(sorted(pdf.value_date.unique()))
    sig_dates = pd.to_datetime(sorted(signals.value_date.unique()))

    # check to see that date ranges are equivalent for both price and vol data
    if not np.array_equal(vdates, pdates):
        print('vol_dates: ', vdates)
        print('price_dates: ', pdates)
        print('difference: ', [x for x in vdates if x not in pdates])
        print('difference 2: ', [x for x in pdates if x not in vdates])
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range. Aborting run.')

    if not np.array_equal(sig_dates, vdates):
        print('v - sig difference: ',
              [x for x in vdates if x not in sig_dates])
        print('sig - v difference: ',
              [x for x in sig_dates if x not in vdates])
        raise ValueError('signal dates dont match up with vol dates')

    if not np.array_equal(sig_dates, pdates):
        print('p - sig difference: ',
              [x for x in pdates if x not in sig_dates])
        print('sig - v differenceL ',
              [x for x in sig_dates if x not in pdates])
        raise ValueError('signal dates dont match up with price dates')

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

    # get rollover dates
    rollover_dates = get_rollover_dates(rolldf)

    e1 = time.clock() - t
    print('TOTAL PREP TIME: ', e1)

    ###############################################################
    # print statements for informational purposes #
    print('START DATE: ', start_date)
    # print('INTERNAL_DATE: ', internal_date)
    print('END DATE: ', end_date)
    print('Portfolio: ', pf)
    print('NUM OPS: ', len(pf.OTC_options))
    print('Hedge Conditions: ', hedges)
    print('Brokerage: ', gv.brokerage)
    print('Slippage: ', gv.slippage)
    print('####################################################')
    print('HEDGES GENERATED. RUNNING SIMULATION... [6/7]')

    print('ROLLOVER DATES: ', rollover_dates)

    # # run simulation #
    grosspnl, netpnl, pf1, gross_daily_values, gross_cumul_values, net_daily_values, net_cumul_values, log = run_simulation(
        vdf, pdf, edf, pf, hedges, rollover_dates, brokerage=gv.brokerage, slippage=gv.slippage, signals=signals)

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

    # plotting greeks with pnl
    plt.figure()
    # plt.plot(xvals, net_cumul_values, c='k', alpha=0.8, label='net cumulative pnl')
    plt.plot(log.value_date, log.delta, c='y', alpha=0.8, label='delta')
    plt.plot(log.value_date, log.gamma, c='g', alpha=0.8, label='gamma')
    plt.plot(log.value_date, log.theta, c='b', alpha=0.8, label='theta')
    plt.plot(log.value_date, log.vega, c='r', alpha=0.8, label='vega')
    plt.plot(log.value_date, log.cu_pnl_net, c='k',
             alpha=0.6, label='cumulative pnl')
    plt.legend()
    plt.show()

    plt.figure()
    y = (log.cu_pnl_net - log.cu_pnl_net.mean()) / \
        (log.cu_pnl_net.max() - log.cu_pnl_net.min())
    y1 = log['25d_call_change'] - log['25d_put_change']
    plt.plot(log.value_date, y, c='k',
             alpha=0.8, label='cumulative pnl')
    plt.plot(log.value_date, y1,
             c='c', alpha=0.7, label='25d_c_vol - 25d_p_vol')
    # plt.plot(log.value_date, log['25d_put_change'],
    #          c='m', alpha=0.5, label='25 Delta Put Vol Change')
    plt.plot(log.value_date, log.price_change,
             c='b', alpha=0.4, label='Price change')

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
###############
