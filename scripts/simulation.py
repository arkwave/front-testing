# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2017-03-07 21:31:13
# @Last Modified by:   arkwave
# @Last Modified time: 2017-09-13 18:19:05

################################ imports ###################################

# general imports
import numpy as np
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import pprint


# user defined imports
# from .scripts.prep_data import prep_portfolio, generate_hedges, sanity_check
# from .scripts.fetch_data import prep_datasets, pull_alt_data
from .util import create_underlying, create_vanilla_option, close_out_deltas, create_composites, blockPrint
from .calc import get_barrier_vol
from .hedge import Hedge
from .signals import apply_signal


# blockPrint()
# enablePrint()
###########################################################################
######################## initializing variables ###########################
###########################################################################
# Dictionary of multipliers for greeks/pnl calculation.
# format  =  'product' : [dollar_mult, lot_mult, futures_tick,
# options_tick, pnl_mult]

multipliers = {

    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 20, 50],
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
    'QC': ['H', 'K', 'N', 'U', 'Z'],
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
timestep = 1 / 365
seed = 7
np.random.seed(seed)

########################################################################
########################################################################


#####################################################
############## Main Simulation Loop #################
#####################################################

def run_simulation(voldata, pricedata, expdata, pf, flat_vols=False, flat_price=False,
                   end_date=None, brokerage=None, slippage=None, signals=None,
                   plot_results=True, drawdown_limit=None):
    """
    Each run of the simulation consists of 5 steps:
        1) Feed data into the portfolio.

        2) Compute:
                > change in greeks from price and vol update
                > change in overall value of portfolio from price and vol update.
                > Check for expiry/exercise/ki/ko. Expiry can be due to barriers or tau = 0.
                 Record changes to:
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
        voldata (TYPE): Description
        pricedata (TYPE): Description
        expdata (TYPE): Description
        pf (TYPE): Description
        end_date (None, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description
        signals (TYPE): Description
        roll_portfolio (None, optional): Description
        pf_ttm_tol (None, optional): Description
        pf_roll_product (None, optional): Description
        roll_hedges (None, optional): Description
        h_ttm_tol (None, optional): Description
        h_roll_product (None, optional): Description

    Returns:
        TYPE: Description


    Deleted Parameters:
        hedges (TYPE): Description
        roll_product (None, optional): Description
        ttm_tol (None, optional): Description


    """

    ##### timers #####
    e1 = time.clock()
    t = time.clock()
    ##################

    ########### initializing pnl variables ###########
    grosspnl = 0
    netpnl = 0
    vegapnl = 0
    gammapnl = 0

    gamma_pnl_daily = []
    vega_pnl_daily = []
    gamma_pnl_cumul = []
    vega_pnl_cumul = []

    gross_daily_values = []
    gross_cumul_values = []
    net_daily_values = []
    net_cumul_values = []
    ##################################################

    ############ Other useful variables: #############
    loglist = []

    date_range = sorted(voldata.value_date.unique())

    # initial value for pnl calculations.
    init_val = 0

    # highest cumulative pnl until this point
    highest_value = 0

    # preallocating signals if sigval dict
    print(signals is not None)
    sigvals = {}
    if signals is not None:
        pdts = signals.pdt.unique()
        for pdt in pdts:
            print('pdt: ', pdt)
            sigvals[(pdt, 'call')] = 0
            sigvals[(pdt, 'put')] = 0

    print('sigvals: ', sigvals)
    # boolean flag indicating missing data
    # Note: [partially depreciated]
    broken = False
    ##################################################
    # print('signals: ', signals)

    for i in range(len(date_range)):
        # get the current date
        date = date_range[i]
        dailycost = 0

    # Steps 1 & 2: Error checks to prevent useless simulation runs.
        # checks to make sure if there are still non-hedge securities in pf
        # isolate data relevant for this day.
        date = pd.to_datetime(date)
        print('##################### date: ', date, '################')

        if len(pf.OTC_options) == 0 and len(pf.OTC_futures) == 0 and not pf.empty():
            print('ALL OTC OPTIONS HAVE EXPIRED. ENDING SIMULATION...')
            break
        # if end_date is inputted, check to see if the current date exceeds
        # end_date
        if end_date:
            if date >= end_date:
                print('REACHED END OF SIMULATION.')
                break

        if drawdown_limit is not None:
            curr_pnl = net_cumul_values[-1] if net_cumul_values else 0
            if highest_value - curr_pnl >= drawdown_limit:
                print('Current Drawdown: ', highest_value - curr_pnl)
                print('DRAWDOWN LIMIT HAS BEEN BREACHED. ENDING SIMULATION...')
                break
            else:
                print('Current Drawdown: ', highest_value - curr_pnl)
                print('Current Drawdown Percentage: ',
                      ((highest_value - curr_pnl)/(drawdown_limit)) * 100)

                print('Drawdown within bounds. Continuing...')

        # try to get next date
        try:
            next_date = date_range[i + 1]
        except IndexError:
            next_date = None

        # filter data specific to the current day of the simulation.
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]

        print('Portfolio before any ops: ', pf)

    # Step 3: Feed data into the portfolio.
        pf, broken, gamma_pnl, vega_pnl, exercise_profit, exercise_futures, barrier_futures \
            = feed_data(vdf, pdf, pf, init_val, flat_vols=flat_vols, flat_price=flat_price)

    # Step 4: Compute pnl for the day
        updated_val = pf.compute_value()
        # print('updated value after feed: ', updated_val)
        # print('init val: ', init_val)
        dailypnl = (updated_val -
                    init_val) + exercise_profit if (init_val != 0 and updated_val != 0) else 0

    # Detour: add in exercise & barrier futures if required.
        if exercise_futures:
            pf.add_security(exercise_futures, 'OTC')
        if barrier_futures:
            pf.add_security(barrier_futures, 'hedge')
        # print('pf after resid. futures: ', pf)

    # Step 5: Apply signal
        if signals is not None:
            print('signals not none, applying. ')
            relevant_signals = signals[signals.value_date == date]
            pf, cost, sigvals = apply_signal(pf, pdf, vdf, relevant_signals, date,
                                             sigvals, strat='filo',
                                             brokerage=brokerage, slippage=slippage)
            dailycost += cost

    # Step 6: rolling over portfolio and hedges if required.
        # simple case
        pf, cost, all_deltas = roll_over(pf, vdf, pdf, date, brokerage=brokerage,
                                         slippage=slippage)
        pf.refresh()

    # Step 7: Hedge - bring greek levels across portfolios (and families) into
    # line with target levels.
        pf, cost, roll_hedged = rebalance(vdf, pdf, pf, brokerage=brokerage,
                                          slippage=slippage, next_date=next_date)

    # Step 9: Subtract brokerage/slippage costs from rebalancing. Append to
    # relevant lists.

        # gamma/vega pnls
        gamma_pnl_daily.append(gamma_pnl)
        vega_pnl_daily.append(vega_pnl)
        gammapnl += gamma_pnl
        vegapnl += vega_pnl
        gamma_pnl_cumul.append(gammapnl)
        vega_pnl_cumul.append(vegapnl)
        # gross/net pnls
        gross_daily_values.append(dailypnl)
        net_daily_values.append(dailypnl - dailycost)
        grosspnl += dailypnl
        gross_cumul_values.append(grosspnl)
        netpnl += (dailypnl - cost)
        net_cumul_values.append(netpnl)

    # Step 9.5: Update highest_value so that the next
    # loop can check for drawdown.
        if netpnl > highest_value:
            highest_value = netpnl

    # Step 10: Initialize init_val to be used in the next loop.
        num_days = 0 if next_date is None else (
            pd.Timestamp(next_date) - pd.Timestamp(date)).days
        init_val = pf.compute_value()

    # Step 11: timestep after all operations have been performed.
        print('actual timestep...')
        pf.timestep(num_days * timestep)

        print('[1.0]   EOD PNL (GROSS): ', dailypnl)
        print('[1.0.1] EOD Vega PNL: ', vega_pnl)
        print('[1.0.2] EOD Gamma PNL: ', gamma_pnl)
        print('[1.0.3] EOD PNL (NET) :', dailypnl - dailycost)
        print('[1.0.4] Cumulative PNL (GROSS): ', grosspnl)
        print('[1.0.5] Cumulative Vega PNL: ', vegapnl)
        print('[1.0.6] Cumulative Gamma PNL: ', gammapnl)
        print('[1.0.7] Cumulative PNL (net): ', netpnl)
        print('[1.1]  EOD PORTFOLIO: ', pf)

    # Step 12: Logging relevant output to csv
    ##########################################################################

        # Portfolio-wide information
        dic = pf.get_net_greeks()
        call_vega = sum([op.vega for op in pf.get_all_options()
                         if op.K >= op.underlying.get_price()])

        put_vega = sum([op.vega for op in pf.get_all_options()
                        if op.K < op.underlying.get_price()])
        if drawdown_limit is not None:
            drawdown_val = highest_value - net_cumul_values[-1]
            drawdown_pct = (
                highest_value - net_cumul_values[-1])/(drawdown_limit)

        # option specific information
        for op in pf.OTC_options:
            ftprice = op.underlying.get_price()
            op_value = op.get_price()
            # pos = 'long' if not op.shorted else 'short'
            char = op.char
            oplots = -op.lots if op.shorted else op.lots
            opvol = op.vol
            strike = op.K
            pdt, ftmth, opmth = op.get_product(), op.get_month(), op.get_op_month()
            vol_id = pdt + '  ' + opmth + '.' + ftmth
            tau = round(op.tau * 365)

            d, g, t, v = dic[pdt][ftmth]

            lst = [date, vol_id, char, tau, op_value, oplots,
                   ftprice, strike, opvol, dailypnl, dailypnl - dailycost, grosspnl, netpnl,
                   gamma_pnl, gammapnl, vega_pnl, vegapnl, roll_hedged, d, g, t, v]

            cols = ['value_date', 'vol_id', 'call/put', 'ttm', 'option_value', 'option_lottage',
                    'future price', 'strike', 'vol',
                    'eod_pnl_gross', 'eod_pnl_net', 'cu_pnl_gross', 'cu_pnl_net',
                    'eod_gamma_pnl', 'cu_gamma_pnl', 'eod_vega_pnl', 'cu_vega_pnl',
                    'delta_rolled', 'net_delta', 'net_gamma', 'net_theta', 'net_vega']

            adcols = ['pdt', 'ft_month', 'op_month', 'delta', 'gamma', 'theta',
                      'vega', 'net_call_vega', 'net_put_vega', 'b/s']

            if op.barrier is not None:
                cpi = 'C' if op.char == 'call' else 'P'
                barlevel = op.ki if op.ki is not None else op.ko
                bvol = get_barrier_vol(
                    vdf, op.get_product(), op.tau, cpi, barlevel)
                knockedin = op.knockedin
                knockedout = op.knockedout
                lst.extend([barlevel, bvol, knockedin, knockedout])
                cols.extend(['barlevel', 'barrier_vol',
                             'knockedin', 'knockedout'])

            if drawdown_limit is not None:
                lst.extend([drawdown_val, drawdown_pct])
                cols.extend(['drawdown', 'drawdown_pct'])

            cols.extend(adcols)

            # getting net greeks
            delta, gamma, theta, vega = op.greeks()

            lst.extend([pdt, ftmth, opmth, delta, gamma, theta,
                        vega, call_vega, put_vega, dailycost])

            l_dic = OrderedDict(zip(cols, lst))
            loglist.append(l_dic)

    ##########################################################################


##########################################################################
##########################################################################
##########################################################################

    # Step 10: Plotting results/data viz
    ######################### PRINTING OUTPUT ###########################
    log = pd.DataFrame(loglist)

    # appending 25d vol changes and price changes
    if signals is not None:
        signals.loc[signals['call/put'] == 'call', 'call_put_id'] = 'C'
        signals.loc[signals['call/put'] == 'put', 'call_put_id'] = 'P'

        df = pd.merge(signals, pricedata[['value_date', 'underlying_id',
                                          'settle_value', 'vol_id', 'call_put_id']],
                      on=['value_date', 'vol_id', 'call_put_id'])
        df = df.drop_duplicates()

        log = pd.merge(log, df[['value_date', 'vol_id', 'settle_value',
                                'signal']],
                       on=['value_date', 'vol_id'])

    elapsed = time.clock() - t

    print('Time elapsed: ', elapsed)

    print('################# Portfolio: ###################')
    print(pf)

    print('SIMULATION COMPLETE. PRINTING RELEVANT OUTPUT... [7/7]')

    print('##################### PNL: #####################')
    print('gross pnl: ', grosspnl)
    print('net pnl: ', netpnl)
    print('vega pnl: ', vegapnl)
    print('gamma pnl: ', gammapnl)
    print('################# OTHER INFORMATION: ##################')

    gvar = np.percentile(gross_daily_values, 10)
    cvar = np.percentile(net_daily_values, 10)

    print('VaR [gross]: ', gvar)
    print('VaR [net]: ', cvar)

    # calculate max drawdown
    if len(gross_daily_values) > 1:
        print('Max Drawdown [gross]: ', min(np.diff(gross_daily_values)))
        print('Max Drawdown [net]: ', min(np.diff(net_daily_values)))

    # time elapsed 2 #
    e2 = time.clock() - e1
    print('SIMULATION RUNTIME: ', e2)

    print('######################################################')
    # plotting histogram of daily pnls

    if plot_results:
        plt.figure()
        plt.hist(gross_daily_values, bins=20,
                 alpha=0.6, label='gross pnl distribution')
        plt.hist(net_daily_values, bins=20,
                 alpha=0.6, label='net pnl distribution')
        plt.title('PnL Distribution: Gross/Net')
        plt.legend()
        # plt.show()

        # plotting gross pnl values
        plt.figure()
        colors = ['c' if x >= 0 else 'r' for x in gross_daily_values]
        xvals = list(range(1, len(gross_daily_values) + 1))
        plt.bar(xvals, net_daily_values, align='center',
                color=colors, alpha=0.6, label='net daily values')
        plt.plot(xvals, gross_cumul_values, c='b',
                 alpha=0.8, label='gross cumulative pnl')
        plt.plot(xvals, net_cumul_values, c='r',
                 alpha=0.8, label='net cumulative pnl')
        plt.plot(xvals, gamma_pnl_cumul, c='g',
                 alpha=0.5, label='cu. gamma pnl')
        plt.plot(xvals, vega_pnl_cumul, c='y', alpha=0.5, label='cu. vega pnl')
        plt.title('gross/net pnl daily')
        plt.legend()
        # plt.show()

        # cumulative gamma/vega/cumulpnl values
        plt.figure()
        # colors = ['c' if x >= 0 else 'r' for x in gamma_pnl_daily]
        plt.plot(log.value_date, log.cu_gamma_pnl, color='c',
                 alpha=0.6, label='cu. gamma pnl')
        plt.plot(log.value_date, log.cu_vega_pnl,
                 color='m', alpha=0.6, label='cu. vega pnl')
        plt.plot(log.value_date, log.cu_pnl_gross, color='k',
                 alpha=0.8, label='gross pnl')
        plt.title('gamma/vega/cumulative pnls')
        plt.legend()
        # plt.show()

        # # net values
        # plt.figure()
        # # colors = ['c' if x >= 0 else 'r' for x in gamma_pnl_daily]
        # plt.plot(log.value_date, log.eod_gamma_pnl, color='c',
        #          alpha=0.6, label='eod gamma pnl')
        # plt.plot(log.value_date, log.eod_vega_pnl,
        #          color='m', alpha=0.6, label='eod vega pnl')
        # plt.plot(log.value_date, log.eod_pnl_gross, color='k',
        #          alpha=0.8, label='eod pnl')
        # plt.title('gamma/vega/daily eod pnls')
        # plt.legend()
        # # plt.show()

        # # plotting greeks with pnl
        # plt.figure()
        # # plt.plot(xvals, net_cumul_values, c='k',
        # #          alpha=0.8, label='net cumulative pnl')
        # plt.plot(log.value_date, log.net_delta, c='y', alpha=0.8, label='delta')
        # plt.plot(log.value_date, log.net_gamma, c='g', alpha=0.8, label='gamma')
        # plt.plot(log.value_date, log.net_theta, c='b', alpha=0.8, label='theta')
        # plt.plot(log.value_date, log.net_vega, c='r', alpha=0.8, label='vega')
        # # plt.plot(log.value_date, log.cu_pnl_net, c='k',
        # #          alpha=0.6, label='cumulative pnl')
        # plt.legend()
        # plt.title('Greeks over simulation period')
        # # plt.show()

        # if signals is not None:
        #     print('signals not none, proceeding to additional plots.')
        #     plt.figure()
        #     # y = (log.cu_pnl_net - np.mean(log.cu_pnl_net)) / \
        #     #     (log.cu_pnl_net.max() - log.cu_pnl_net.min())

        #     # y1 = log['dval_call_vol_change'] - log['dval_put_vol_change']
        #     unique_log = log.drop_duplicates(subset='value_date')

        #     settle_vals = unique_log.settle_value
        #     callvols = unique_log.call_vol
        #     putvols = unique_log.put_vol
        #     callvols *= 100
        #     callvolchange = (callvols.shift(-1) - callvols).shift(1).fillna(0)
        #     putvols *= 100
        #     putvolchange = (putvols.shift(-1) - putvols).shift(1).fillna(0)
        #     dates = unique_log.value_date

        #     spotchange = (settle_vals.shift(-1) - settle_vals).shift(1).fillna(0)

        #     plt.plot(dates, spotchange, c='k',
        #              alpha=0.8, label='spot change')

        #     plt.plot(dates, callvolchange,
        #              c='c', alpha=0.7, label='call_vol change')

        #     plt.plot(dates, putvolchange,
        #              c='r', alpha=0.7, label='put_vol change')

        #     plt.title('Spot Price, call_vol and put_vol changes')
        #     plt.legend()

        plt.show()

    # preparing the log file for analytics.
    # analytics_csv = log.drop_duplicates('eod_pnl_gross')
    # analytics_cols = ['value_date', 'option_lottage', 'strike',
    #                   'future price', 'vol', 'eod_pnl_net', 'cu_pnl_net', 'eod_gamma_pnl',
    #                   'cu_gamma_pnl', 'eod_vega_pnl', 'cu_vega_pnl', 'net_delta',
    #                   'net_gamma', 'net_theta', 'net_vega']
    # analytics_csv = analytics_csv[analytics_cols]

    return log
    # return grosspnl, netpnl, pf, gross_daily_values, gross_cumul_values,
    # net_daily_values, net_cumul_values, log


##########################################################################
##########################################################################
##########################################################################


##########################################################################
########################## Helper functions ##############################
##########################################################################

def feed_data(voldf, pdf, pf, init_val, brokerage=None,
              slippage=None, flat_vols=False, flat_price=False):
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
        init_val (TYPE): Description
        brokerage (int, optional): brokerage fees per lot.
        slippage (int, optional): slippage cost
        flat_vols (bool, optional): Description
        flat_price (bool, optional): Description

    Returns:
        tuple: change in value, updated portfolio object, and whether or not there is missing data.

    Raises:
        ValueError: Raised if voldf is empty.

    Notes:
        1) DO NOT ADD SECURITIES INSIDE THIS FUNCTION. Doing so will mess up the PnL calculations \
        and give you vastly overinflated profits or vastly exaggerated losses, due to reliance on \
        the compute_value function.

    Deleted Parameters:
        dic (dictionary): dictionary of rollover dates, in the format
            {product_i: [c_1 rollover, c_2 rollover, ... c_n rollover]}
        prev_date (TYPE): Description

    """
    barrier_futures = []
    exercise_futures = []
    broken = False
    exercise_profit = 0
    barrier_profit = 0

    # 1) sanity checks
    if pf.empty():
        return pf, False, 0, 0, 0, [], []

    if voldf.empty:
        raise ValueError('vol df is empty!')

    # print('curr_date: ', curr_date)
    date = pd.to_datetime(pdf.value_date.unique()[0])

    # 2)  update prices of futures, underlying & portfolio alike.
    if not broken:
        for ft in pf.get_all_futures():
            pdt = ft.get_product()
            try:
                # case: flat price.
                if flat_price:
                    ft.update_price(ft.get_price())
                else:
                    uid = ft.get_product() + '  ' + ft.get_month()
                    val = pdf[(pdf.pdt == pdt) &
                              (pdf.underlying_id == uid)].settle_value.values[0]
                    pf, cost, fts = handle_barriers(
                        voldf, pdf, ft, val, pf, date)
                    barrier_futures.extend(fts)
                    ft.update_price(val)

            # index error would occur only if data is missing.
            except IndexError:
                # print('###### PRICE DATA MISSING #######')
                # print('ordering: ', ordering)
                # print('uid: ', uid)
                # print('pdt: ', pdt)
                # print('debug 1: ', pdf[(pdf.pdt == pdt) &
                #                        (pdf.order == ordering)])
                # broken = True
                ft.update_price(ft.get_price())
                # break

    # handling pnl arising from barrier futures.
        if barrier_futures:
            for ft in barrier_futures:
                pdt = ft.get_product()
                pnl_mult = multipliers[pdt][-1]
                try:
                    uid = ft.get_product() + '  ' + ft.get_month()
                    val = pdf[(pdf.pdt == pdt) &
                              (pdf.underlying_id == uid)].settle_value.values[0]
                    # calculate difference between ki price and current price
                    diff = val - ft.get_price()
                    pnl_diff = diff * ft.lots * pnl_mult
                    barrier_profit += pnl_diff
                    # update the price of each future
                    ft.update_price(val)

                # index error would occur only if data is missing.
                except IndexError:
                    # print('###### PRICE DATA MISSING #######')
                    # print('ordering: ', ordering)
                    # print('uid: ', uid)
                    # print('pdt: ', pdt)
                    # print('debug 1: ', pdf[(pdf.pdt == pdt) &
                    #                        (pdf.order == ordering)])
                    ft.update_price(ft.get_price())
                    # broken = True
                    # break

    exercise_profit, pf, exercised, exercise_futures = handle_exercise(
        pf, brokerage, slippage)

    # print('handle exercise profit: ', exercise_profit)

    total_profit = exercise_profit + barrier_profit

    # refresh portfolio after price updates.
    pf.refresh()
    # pf.update_sec_by_month(None, 'OTC', update=True)
    # pf.update_sec_by_month(None, 'hedge', update=True)

    # removing expiries
    pf.remove_expired()

    # TODO: Need to re-implement the sanity checks for when portfolio needs
    # to be cleaned out.

    # sanity check: if no active options, close out entire portfolio.
    if not pf.OTC_options:
        print('all options have expired - removing hedge futures')
        deltas_to_close = set()
        for ft in pf.hedge_futures:
            pdt = ft.get_product()
            mth = ft.get_month()
            price = ft.get_price()
            iden = (pdt, mth, price)
            deltas_to_close.add(iden)

        # case: portfolio is being closed immediately after an exercise. Treat
        # as a cash settlement.
        # TODO: think through this and make sure it is right.
        if deltas_to_close:
            if barrier_futures:
                for pdt, mth, price in deltas_to_close:
                    bfts_to_remove = [x for x in barrier_futures if x.get_product() == pdt and
                                      x.get_month() == mth]
                    barrier_futures = [
                        x for x in barrier_futures if x not in bfts_to_remove]

            if exercise_futures:
                for pdt, mth, _ in deltas_to_close:
                    efts_to_remove = [x for x in exercise_futures if x.get_product() == pdt and
                                      x.get_month() == mth]
                    exercise_futures = [
                        x for x in exercise_futures if x not in efts_to_remove]

        print('feed_data - dtc: ', deltas_to_close)
        pf, cost = close_out_deltas(pf, deltas_to_close)
        total_profit -= cost

    # calculating gamma pnl
    intermediate_val = pf.compute_value()

    # TODO: Reimplement this.
    # print('intermediate value: ', intermediate_val)
    if exercised:
        if intermediate_val == 0:
            gamma_pnl = exercise_profit
        else:
            gamma_pnl = (intermediate_val + exercise_profit) - init_val
    else:
        gamma_pnl = intermediate_val + total_profit - init_val \
            if (init_val != 0 and intermediate_val != 0) else 0

    # print('feed_data - gamma pnl: ', gamma_pnl)

    # print('pf after rollovers and expiries: ', pf)

    # update option attributes by feeding in vol.
    for op in pf.get_all_options():
        # info reqd: strike, order, product, tau
        strike, product, tau = op.K, op.product, op.tau
        b_vol, strike_vol = None, None
        # print('price: ', op.compute_price())
        # print('OP GREEKS: ', op.greeks())
        cpi = 'C' if op.char == 'call' else 'P'
        # interpolate or round? currently rounding, interpolation easy.
        ticksize = multipliers[op.get_product()][-2]
        # get strike corresponding to closest available ticksize.
        # print('feed_data - ticksize: ', ticksize, op.get_product())
        strike = round(round(strike / ticksize) * ticksize, 2)
        vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()

        # case: flat vols.
        if flat_vols:
            op.update_greeks(vol=op.vol, bvol=op.bvol)
        else:
            try:
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                            (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                df_tau = min(val.tau, key=lambda x: abs(x - tau))
                strike_vol = val[val.tau == df_tau].settle_vol.values[0]
                # print('UPDATED - new vol: ', strike_vol)
            except (IndexError, ValueError):
                # print('### VOLATILITY DATA MISSING ###')
                # print('product: ', product)
                # print('strike: ', strike)
                # print('order: ', order)
                # print('vid: ', vid)
                # print('call put id: ', cpi)
                # print('tau: ', df_tau)
                # broken = True
                strike_vol = op.vol
                # break

            try:
                if op.barrier is not None:
                    barlevel = op.ki if op.ki is not None else op.ko
                    b_val = voldf[(voldf.pdt == product) & (voldf.strike == barlevel) &
                                  (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                    df_tau = min(b_val.tau, key=lambda x: abs(x - tau))
                    b_vol = val[val.tau == df_tau].settle_vol.values[0]
                    # print('UPDATED - new barrier vol: ', b_vol)
            except (IndexError, ValueError):
                # print('### BARRIER VOLATILITY DATA MISSING ###')
                # print('product: ', product)
                # print('strike: ', barlevel)
                # print('order: ', order)
                # print('vid: ', vid)
                # print('call put id: ', cpi)
                # print('tau: ', df_tau)
                # broken = True
                b_vol = op.bvol
                # break

            op.update_greeks(vol=strike_vol, bvol=b_vol)

    # (today's price, today's vol) - (today's price, yesterday's vol)
    vega_pnl = pf.compute_value() - intermediate_val \
        if (init_val != 0 and intermediate_val != 0) else 0

    # updating portfolio after modifying underlying objects
    pf.refresh()
    # pf.update_sec_by_month(None, 'OTC', update=True)
    # pf.update_sec_by_month(None, 'hedge', update=True)

    # print('Portfolio After Feed: ', pf)
    # print('[7]  NET GREEKS: ', str(pprint.pformat(pf.net_greeks)))

    return pf, broken, gamma_pnl, vega_pnl, total_profit, exercise_futures, barrier_futures


# TODO: implement slippage and brokerage
def handle_barriers(vdf, pdf, ft, val, pf, date):
    """Handles delta differential and spot hedging for knockin/knockout events.

    Args:
        vdf (TYPE): dataframe of volatilities
        pdf (TYPE): dataframe of prices
        ft (TYPE): future object whose price is being updated
        pf (TYPE): portfolio being simulated.

    """
    # print('handling barriers...')

    # print('future val: ', val)

    step = 0
    delta_diff = 0
    ret = None
    ft_price = 0
    pdt = ft.get_product()
    ftmth = ft.get_month()

    # op_ticksize = multipliers[pdt][-2]
    ft_ticksize = multipliers[pdt][2]
    ops = [op for op in pf.OTC_options if op.underlying == ft]

    # print('ops: ', [str(op) for op in ops])
    if not ops:
        return pf, 0, []

    op = ops[0]
    # print('barrierstatus: ', op.barrier)
    # opmth = op.get_op_month()

    # base case: option is a vanilla option, or has expired.

    if (op.barrier is None) or (op.check_expired()):
        # print('simulation.handle_barrers - vanilla/expired case ')
        ret = pf, 0, []
        return ret

    bar_op = copy.deepcopy(op)
    # create vanilla option with the same stats as the barrier option AFTER
    # knockin.

    volid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()
    vanop = create_vanilla_option(vdf, pdf, volid, op.char, op.shorted,
                                  lots=op.lots, vol=op.vol, strike=op.K,
                                  bullet=op.bullet)

    # case 1: knockin case - option is not currently knocked in.
    if op.ki is not None:
        print('simulation.handle_barriers - knockin case')
        # case 1-1: di option, val is below barrier.
        if op.direc == 'down' and val <= op.ki:
            print('simulation.handle_barriers - down-in case ' + str(op))
            step = ft_ticksize
        elif op.direc == 'up' and val >= op.ki:
            print('simulation.handle_barriers - up-in case ' + str(op))
            step = -ft_ticksize
        # knockin case with no action
        else:
            print('barriers handled')
            return pf, 0, []

        # get delta after converting to vanilla option.
        # round barrier to closest future ticksize price.
        ft_price = round(round(op.ki / ft_ticksize) * ft_ticksize, 2)
        print('ftprice: ', ft_price)
        vanop.underlying.update_price(ft_price)
        van_delta = vanop.greeks()[0]
        # van_delta = vanop.delta
        print('van_delta: ', van_delta)
        # get the delta one ticksize above barrier of the current option
        bar_diff = ft_price + step
        print('bar_diff: ', bar_diff)
        bar_op.underlying.update_price(bar_diff)

        barr_delta = bar_op.greeks()[0]
        print('bar_delta: ', barr_delta)
        delta_diff = barr_delta - van_delta
        print('delta diff: ', delta_diff)

        # create the requisite futures.
        ft_shorted = True if delta_diff < 0 else False
        lots_req = abs(round(delta_diff))
        fts, _ = create_underlying(pdt, ftmth, pdf, date, ftprice=ft_price,
                                   shorted=ft_shorted, lots=lots_req)
        print('future price: ', ft_price)
        print('ft added: ', str(fts))

        ret = pf, 0, [fts]

    # case 2: knockout.
    elif op.ko is not None:
        print('simulation.handle_barriers - knockout case')
        if op.knockedout:
            print('simulation.handle_barriers - knockedout')
            ret = pf, 0, []

        else:
            # case 1: updating price to val will initiate a knockout
            if op.direc == 'up' and val > op.ko:
                print('simulation.handle_barriers - up-out case ' + str(op))
                step = -ft_ticksize
            elif op.direc == 'down' and val < op.ko:
                print('simulation.handle_barriers - down-out case ' + str(op))
                step = ft_ticksize
            else:
                print('barriers handled')
                return pf, 0, []

            ft_price = round(
                round((op.ko + step) / ft_ticksize) * ft_ticksize, 2)
            bar_op.underlying.update_price(ft_price)
            print('future price: ', ft_price)
            delta_diff = bar_op.delta
            print('delta_diff: ', delta_diff)

            # creating the future object.
            ft_shorted = True if delta_diff < 0 else False
            lots_req = abs(round(delta_diff))
            fts, _ = create_underlying(pdt, ftmth, pdf, date, ftprice=ft_price,
                                       shorted=ft_shorted, lots=lots_req)

            ret = pf, 0, [fts]

    # regular vanilla option case; do nothing.
    else:
        print('simulation.handle_barriers - final else case')
        ret = pf, 0, []

    print('barriers handled')
    return ret


def handle_exercise(pf, brokerage=None, slippage=None):
    """Handles option exercise, as well as bullet vs daily payoff.

    Args:
        pf (Portfolio object): the portfolio being run through the simulator.
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        tuple: the combined PnL from exercising (if appropriate) and daily/bullet payoffs, \
        as well as the updated portfolio.

    Notes on implementation:
        1) options are exercised if less than or equal to 2 days to maturity, and
        option is in the money. Futures obtained are immediately sold and profit is locked in.

    """
    if pf.empty():
        return 0, pf, False, []
    exercised = False
    t = time.clock()
    profit = 0
    tol = 1 / 365
    # handle options exercise
    # all_ops = pf.get_all_options()
    otc_ops = pf.OTC_options
    hedge_ops = pf.hedge_options

    tobeadded = []
    for op in otc_ops:
        if np.isclose(op.tau, tol):
            exer = op.exercise()
            op.tau = 0
            if exer:
                print("exercising OTC op " + str(op))
                if op.settlement == 'cash':
                    print("----- CASH SETTLEMENT: OTC OP ------")
                elif op.settlement == 'futures':
                    print('----- FUTURE SETTLEMENT: OTC OP ------')
                    ft = op.get_underlying()
                    ft.update_lots(op.lots)
                    print('future added - ', str(ft))
                    print('lots: ', op.lots, ft.lots)
                    tobeadded.append(ft)

                # calculating the net profit from this exchange.
                product = op.get_product()
                pnl_mult = multipliers[product][-1]
                # op.get_price() defaults to max(k-s,0 ) or max(s-k, 0)
                # since op.tau = 0
                oppnl = op.lots * op.get_price() * pnl_mult
                print('profit on this exercise: ', oppnl)
                print("-------------------------------------")
                profit += oppnl
                exercised = True

            else:
                print('letting OTC op ' + str(op) + ' expire.')

    for op in hedge_ops:
        if np.isclose(op.tau, tol):
            exer = op.exercise()
            op.tau = 0
            if exer:
                print('exercising hedge op ' + str(op))
                if op.settlement == 'cash':
                    print("----- CASH SETTLEMENT: HEDGE OPS ------")
                elif op.settlement == 'futures':
                    if op.settlement == 'futures':
                        print('----- FUTURE SETTLEMENT: HEDGE OPS ------')
                        ft = op.get_underlying()
                        ft.update_lots(op.lots)
                        print('future added - ', str(ft))
                        print('lots: ', op.lots, ft.lots)
                        tobeadded.append(ft)

                # calculating the net profit from this exchange.
                pnl_mult = multipliers[op.get_product()][-1]
                # op.get_price() defaults to max(k-s,0 ) or max(s-k, 0)
                # since op.tau = 0
                oppnl = op.lots * op.get_price() * pnl_mult
                print('profit on this exercise: ', oppnl)
                print('---------------------------------------')
                profit += oppnl
                exercised = True
            else:
                print('letting hedge op ' + str(op) + ' expire.')

    # debug statement:
    print('handle_exercise - tobeadded: ', [str(x) for x in tobeadded])
    print('handle_exercise - options exercised: ', exercised)
    print('handle_exercise - net exercise profit: ', profit)
    print('handle exercise time: ', time.clock() - t)
    return profit, pf, exercised, tobeadded


###############################################################################
###############################################################################
###############################################################################


###############################################################################
########### Hedging-related functions (generation and implementation) #########
###############################################################################


def roll_over(pf, vdf, pdf, date, brokerage=None, slippage=None):
    """
        If ttm < ttm_tol, closes out that position (and all accumulated deltas), 
        saves lot size/strikes, and rolls it over into the
        next month.

    Args:
        pf (TYPE): portfolio being hedged
        vdf (TYPE): volatility dataframe
        pdf (TYPE): price dataframef
        brokerage (TYPE, optional): brokerage amount
        slippage (TYPE, optional): slippage amount
        ttm_tol (int, optional): tolerance level for time to maturity.

    Returns:
        tuple: updated portfolio and cost of operations undertaken
    """
    # from itertools import cycle
    print(' ---- beginning contract rolling --- ')
    total_cost = 0
    deltas_to_close = set()
    toberemoved = []
    roll_all = False

    print('roll_all: ', roll_all)
    fa_lst = pf.get_families() if pf.get_families() else [pf]
    processed = {}
    roll_all = False
    rolled_vids = set()

    # iterate over each family.
    for fa in fa_lst:
        # case: no rolling specified for this family.
        if not fa.roll:
            continue
        target_product = fa.roll_product

        # case: entire family is to be rolled basis a product.
        if target_product is not None:
            prod = target_product
            ops = [op for op in fa.get_all_options()
                   if op.get_product() == prod]
            if ops[0].tau * 365 <= fa.ttm_tol:
                roll_all = True

        # case: first time processing this family.
        if fa not in processed:
            processed[fa] = set()
        else:
            # case where all options have been processed.
            if len(processed[fa]) == len(fa.get_all_options()):
                continue

        # assign all options.
        dic = fa.get_all_options()
        print('OPTIONS TO ROLL: ', pprint.pformat([str(x) for x in dic]))

        # get list of already processed ops for this family.
        processed_ops = processed[fa]

        # iterate over each option in this family
        for op in dic.copy():
            flag = 'OTC' if op in fa.OTC_options else 'hedge'
            # case: op has already been processed since its parter was
            # processed.
            if op in processed_ops:
                continue
            composites = []

            # case: roll if ttm threshold is breached or roll_all is triggered.
            needtoroll = (((op.tau * 365) < fa.ttm_tol) or roll_all)
            if needtoroll:
                print('rolling option ' + str(op) + ' from ' + flag)
                toberemoved.append(op)
                rolled_vids.add(op.get_vol_id())
                # creating the underlying future object
                print('rolling op')
                fa, cost, newop, old_op, iden = contract_roll(
                    fa, op, vdf, pdf, date, flag)
                composites.append(newop)
                processed_ops.add(old_op)
                deltas_to_close.add(iden)
                total_cost += cost
                # if rolling, roll partners as well so that composite structure
                # can be maintained.
                print('handling partners...')
                deltas_to_close, composites, total_cost, processed, rolled_vids = \
                    roll_handle_partners(date, pf, fa, op, deltas_to_close,
                                         composites, total_cost, processed, vdf, pdf, rolled_vids)

                composites = create_composites(composites)
    pf.refresh()
    # edge case: in the case where a roll_product is specified and hedge
    # options are added, these hedge options may not be found in the above
    # loop since hedge options have no default behavior of being initialized
    # as partners with OTC options.
    edge_case_ops = set()
    print('rolled vids: ', rolled_vids)
    for vid in rolled_vids:
        ops = [op for op in pf.hedge_options if op.get_vol_id() == vid]
        if ops:
            print('ops: ', [str(x) for x in ops])
            for op in ops:
                if op in edge_case_ops:
                    continue
                partners = op.partners.copy()
                edge_case_ops.add(op)
                composites = [op]
                fam = fa if not pf.get_families() else pf.get_family_containing(op)
                flag = 'hedge'
                # roll the option
                fam, cost, newop, old_op, iden = contract_roll(
                    fam, op, vdf, pdf, date, flag)
                # roll the partners
                deltas_to_close, composites, total_cost, processed, rolled_vids\
                    = roll_handle_partners(date, pf, fa, op, deltas_to_close,
                                           composites, total_cost, processed, vdf, pdf, rolled_vids)
                edge_case_ops.update(partners)
                create_composites(composites)
    pf.refresh()

    pf, cost = close_out_deltas(pf, deltas_to_close)
    pf.refresh()

    total_cost += cost

    print(' --- done contract rolling --- ')
    return pf, total_cost, deltas_to_close


def roll_handle_partners(date, pf, fa, op, deltas_to_close, composites, total_cost, processed, vdf, pdf, rolled_vids):
    """Helper method that handles partners when rolling options. 

    Args:
        date (TYPE): Description
        pf (TYPE): Description
        fa (TYPE): Description
        op (TYPE): Description
        deltas_to_close (TYPE): Description
        composites (TYPE): Description
        total_cost (TYPE): Description
        processed (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
        rolled_vids (TYPE): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        cost (TYPE): Description
    """
    for opx in op.partners:
        print('opx: ', opx)

        # simple case means partner is in same famiily. composite
        # case calls finder.
        tar = fa if not pf.get_families() else pf.get_family_containing(opx)

        if tar not in processed:
            processed[tar] = set()

        if opx not in processed[tar]:
            rolled_vids.add(opx.get_vol_id())
            flag2 = 'OTC' if opx in tar.OTC_options else 'hedge'
            tar, cost2, new_opx, old_opx, iden_2 = contract_roll(
                tar, opx, vdf, pdf, date, flag2)
            composites.append(new_opx)
            deltas_to_close.add(iden_2)
            processed[tar].add(old_opx)
            total_cost += cost2

    return deltas_to_close, composites, total_cost, processed, rolled_vids


def contract_roll(pf, op, vdf, pdf, date, flag):
    """Helper method that deals with contract rolliing if needed.

    Args:
        pf (TYPE): Description
        op (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
    """

    # isolating a roll condition if it exists.
    d_cond = None
    if 'delta' in pf.hedge_params:
        d_cond = [x for x in pf.hedge_params['delta'] if x[0] == 'roll']
        d_cond = d_cond[0][1] if d_cond else None

    pdt = op.get_product()
    ftmth = op.underlying.get_month()
    ft_month, ft_yr = ftmth[0], ftmth[1]
    index = contract_mths[pdt].index(ft_month) + 1
    # case: rollover to the next year
    if index >= len(contract_mths[pdt]):
        ft_yr = str(int(ft_yr) + 1)

    new_ft_month = contract_mths[pdt][
        index % len(contract_mths[pdt])] + ft_yr

    new_ft, ftprice = create_underlying(
        pdt, new_ft_month, pdf, date)

    if new_ft is None:
        raise ValueError('Contract Roll was attempted, but the information for ' +
                         pdt + ' ' + new_ft_month + ' cannot be found in the dataset.\n\
                         Reasons you could be seeing this error: \n\
                            1) contract roll is checked but vol_ids are explicitly specified \n\
                            2) the data for the contract we want to roll into isnt available')

    # print('new ft: ', new_ft)

    # identifying deltas to close
    iden = (pdt, ftmth, ftprice)

    # creating the new options object - getting the tau and vol
    new_vol_id = pdt + '  ' + new_ft_month + '.' + new_ft_month
    lots = op.lots

    r_delta = None
    strike = None
    # case: delta rolling value is specified.
    if d_cond is not None:
        # print('d_cond not None: ', d_cond)
        if d_cond == 50:
            strike = 'atm'
        else:
            r_delta = d_cond
    if strike is None:
        strike = op.K
    # print('strike, r_delta: ', strike, r_delta)
    newop = create_vanilla_option(vdf, pdf, new_vol_id, op.char, op.shorted,
                                  date, lots=lots, strike=strike, delta=r_delta)

    # cost is > 0 if newop.price > op.price
    cost = newop.get_price() - op.get_price()

    pf.remove_security([op], flag)
    pf.add_security([newop], flag)

    return pf, cost, newop, op, iden


def rebalance(vdf, pdf, pf, buckets=None, brokerage=None, slippage=None, next_date=None):
    """Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.

    Args:
        vdf (pandas dataframe): Dataframe of volatilities
        pdf (pandas dataframe): Dataframe of prices
        pf (object): portfolio object
        buckets (None, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        tuple: portfolio, counters

    Deleted Parameters:
        hedges (dict): Dictionary of hedging conditions
        counters (TYPE): Description
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    # hedging delta, gamma, vega.
    hedgearr = [True, True, True, True]
    droll = None

    # sanity check
    if pf.empty():
        return pf, 0, False

    roll_hedged = check_roll_status(pf)
    droll = not roll_hedged

    cost = 0

    # first: handle roll-hedging.
    if roll_hedged:
        print('deltas within bounds. skipping roll_hedging')

    if not roll_hedged:
        print('-------- ROLL HEDGING REQUIRED ---------')

        pf, exp = hedge_delta_roll(
            pf, vdf, pdf, brokerage=brokerage, slippage=slippage)
        cost += exp
        print('-------- ROLL HEDGING COMPLETED ---------')

    hedge_count = 0

    # get unique date.
    date = vdf.value_date.unique()[0]

    # timestep before rebalancing but after delta rolling
    print('timstepping before rebalancing...')
    init_ops = pf.get_all_options()
    num_days = 0 if next_date is None else (
        pd.Timestamp(next_date) - pd.Timestamp(date)).days
    pf.timestep(num_days * timestep)

    # case: no families (i.e. simple portfolio)
    if not pf.get_families():
        print('simulation.rebalance - simple portfolio hedging. ')
        # print('hedges for this dep: ', dep.hedge_params)
        hedge_engine = Hedge(pf, pf.hedge_params, vdf, pdf,
                             buckets=buckets,
                             slippage=slippage, brokerage=brokerage)

        # initial boolean check
        done_hedging = hedge_engine.satisfied()

        # hedging non-delta greeks.
        while (not done_hedging and hedge_count < 3):
            # insert the actual business of hedging here.
            for flag in pf.hedge_params:
                if flag == 'gamma' and hedgearr[1]:
                    fee = hedge_engine.apply('gamma')
                    cost += fee
                    hedge_engine.refresh()
                elif flag == 'vega' and hedgearr[3]:
                    fee = hedge_engine.apply('vega')
                    cost += fee
                    hedge_engine.refresh()
                elif flag == 'theta' and hedgearr[2]:
                    fee = hedge_engine.apply('theta')
                    cost += fee
                    hedge_engine.refresh()

            # debug statements
            print('overall hedge params: ', pf.hedge_params)

            # hedging delta after non-delta greeks.
            if hedgearr[0] and 'delta' in pf.hedge_params:
                # grabbing condition that indicates zeroing condition on
                # delta
                print('hedging delta')
                fee = hedge_engine.apply('delta')
                # fee = ov_hedge.apply('delta')
                cost += fee

            hedge_count += 1
            done_hedging = hedge_engine.satisfied()
            print('pf hedges satisfied: ', done_hedging)

    # case: composite portfolio
    else:
        print('simulation.rebalance - composite portfolio hedging.')
        # initialize hedge engines for each family in the portfolio
        for dep in pf.get_families():
            hedge_engine = Hedge(dep, dep.hedge_params, vdf, pdf,
                                 buckets=buckets,
                                 slippage=slippage, brokerage=brokerage)

            # initial boolean check
            done_hedging = hedge_engine.satisfied()

            # hedging non-delta greeks.
            while (not done_hedging and hedge_count < 3):
                # insert the actual business of hedging here.
                for flag in dep.hedge_params:
                    if flag == 'gamma' and hedgearr[1]:
                        fee = hedge_engine.apply('gamma')
                        cost += fee
                        hedge_engine.refresh()
                    elif flag == 'vega' and hedgearr[3]:
                        fee = hedge_engine.apply('vega')
                        cost += fee
                        hedge_engine.refresh()
                    elif flag == 'theta' and hedgearr[2]:
                        fee = hedge_engine.apply('theta')
                        cost += fee
                        hedge_engine.refresh()

                hedge_count += 1
                done_hedging = hedge_engine.satisfied()
                print('dep hedges satisfied: ', done_hedging)

        # refresh after hedging individual families
        pf.refresh()

        print('simulation.rebalance - portfolio after refresh: ', pf)

        # debug statements
        print('overall hedge params: ', pf.hedge_params)

        # hedging delta overall after all family-specific hedges have been
        # handled.
        if hedgearr[0] and 'delta' in pf.hedge_params:
            # grabbing condition that indicates zeroing condition on
            # delta
            print('hedging delta')
            ov_hedge = Hedge(pf, pf.hedge_params, vdf, pdf)
            fee = ov_hedge.apply('delta')
            cost += fee

    # un-timestep OTC options.
    pf.timestep(-num_days * timestep, ops=init_ops)
    pf.refresh()
    print('hedging completed. ')
    print('pf after hedging: ', pf)
    return (pf,  cost, droll)


def hedge_delta_roll(fpf, vdf, pdf, brokerage=None, slippage=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta 
    exceeds certain bounds.

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

    # initializing dictionary mapping pf -> processed options in that pf.

    if not fpf.get_families():
        print('--- delta rolls: simple portfolio case ---')
        return hedge_delta_roll_simple(fpf, vdf, pdf, brokerage=brokerage, slippage=slippage)

    else:
        print(' --- delta rolls: composite case ---')
        return hedge_delta_roll_comp(fpf, vdf, pdf, brokerage=brokerage, slippage=slippage)


def hedge_delta_roll_simple(pf, vdf, pdf, brokerage=None, slippage=None):
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
    hedges = pf.hedge_params
    # isolating roll conditions.
    roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
        'delta'][i][0] == 'roll']
    if roll_cond:
        # case: roll conditions found.
        roll_cond = roll_cond[0]
        roll_val, bounds = roll_cond[1], np.array(roll_cond[3]) / 100
    else:
        # case: no roll conditions found.
        print('no roll conditions found for family')
        return pf, 0

    roll_val, bounds = roll_cond[1], np.array(roll_cond[3]) / 100

    toberemoved = []

    for op in pf.get_all_options().copy():
        flag = 'OTC' if op in pf.OTC_options else 'hedge'
        # case: option has already been processed due to its partner being
        # processed.
        print('simulation.hedge_delta_roll - option: ', op.get_product(),
              op.char,  round(abs(op.delta / op.lots), 2))
        if op in toberemoved:
            print('option already handled')
            continue
        composites = []
        delta = abs(op.delta / op.lots)
        # case: delta not in bounds.
        diff = (delta - roll_val/100)
        print('diff, bounds: ', diff, bounds)
        if (diff < bounds[0]) or (diff > bounds[1]):
            # if delta > bounds[1] or delta < bounds[0]:
            print('rolling delta: ', op.get_product(),
                  op.char, round(abs(op.delta / op.lots), 2))
            newop, old_op, rcost = delta_roll(pf, op, roll_val, vdf, pdf, flag,
                                              slippage=slippage, brokerage=brokerage)
            toberemoved.append(old_op)
            composites.append(newop)
            cost += rcost
            # if rolling option, roll all partners as well.
            for opx in op.partners:
                new_opx, old_opx, rcost = delta_roll(pf, opx, roll_val, vdf, pdf, flag,
                                                     slippage=slippage, brokerage=brokerage)
                composites.append(new_opx)
                toberemoved.append(old_opx)
                cost += rcost
        composites = create_composites(composites)
        print('composites: ', [str(x) for x in composites])

    print('number of ops rolled: ', len(toberemoved))
    # pf.remove_security(toberemoved, 'OTC')
    # pf.add_security(tobeadded, 'OTC')

    return pf, cost


def hedge_delta_roll_comp(fpf, vdf, pdf, brokerage=None, slippage=None):
    """Helper function that handles delta hedge rolling for portfolios consisting
    of multiple families. 

    Args:
        fpf (TYPE): Description
        pdf (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    cost = 0
    processed = {}
    for pf in fpf.get_families():
        # print(' --- handling rolling for family ' + str(pf.name) + ' ---')
        # initial sanity checks.
        if pf not in processed:
            processed[pf] = []
        else:
            # TODO: figure out what happens with hedge options.
            # case: number of options processed = number of relevant options.
            if len(processed[pf]) == len(pf.get_all_options()):
                # print(' --- finished rolling for family ' + str(pf.name) + ' ---')
                continue

        # grab list of processed options for this portfolio.
        processed_ops = processed[pf]
        hedges = pf.hedge_params

        # isolating roll conditions.
        roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
            'delta'][i][0] == 'roll']
        if roll_cond:
            # case: roll conditions found.
            roll_cond = roll_cond[0]
            roll_val, bounds = roll_cond[1], np.array(roll_cond[3]) / 100
        else:
            # case: no roll conditions found.
            # print('no roll conditions found for family ' + str(pf.name))
            continue

        # starting of per-option rolling logic.
        for op in pf.get_all_options().copy():
            # case: option has already been processed due to its partner being
            # processed.
            flag = 'OTC' if op in pf.OTC_options else 'hedge'
            if op in processed_ops:
                continue
            composites = []
            delta = abs(op.delta / op.lots)
            # case: delta not in bounds.
            diff = (delta - roll_val/100)
            # print('diff, bounds: ', diff, bounds)
            if (diff < bounds[0]) or (diff > bounds[1]):
                # if delta > bounds[1] or delta < bounds[0]:
                print('rolling delta: ', op.get_product(),
                      op.char, round(abs(op.delta / op.lots), 2))
                newop, old_op, rcost = delta_roll(pf, op, roll_val, vdf, pdf, flag,
                                                  slippage=slippage, brokerage=brokerage)
                processed_ops.append(old_op)
                composites.append(newop)
                cost += rcost
                # if rolling option, roll all partners as well.
                for opx in op.partners:
                    print('rolling delta: ', opx.get_product(),
                          opx.char, round(abs(opx.delta / opx.lots), 2))
                    if opx in pf.get_all_options():
                        tar = pf
                    else:
                        tar = fpf.get_family_containing(opx)
                    if tar is None:
                        raise ValueError(str(opx) + ' belongs to a \
                                            non-existent family.')

                    new_opx, old_opx, rcost = delta_roll(tar, opx, roll_val, vdf, pdf, flag,
                                                         slippage=slippage, brokerage=brokerage)
                    composites.append(new_opx)
                    if tar not in processed:
                        processed[tar] = []
                    processed[tar].append(old_opx)
                    cost += rcost
                composites = create_composites(composites)
            else:
                print('family is within bounds. skipping...')

    # print(' --- finished delta rolling for family ' + str(pf.name) + ' ---')
    for x in processed:
        # print('ops delta-rolled belonging to ' + x.name + ':')
        print([str(i) for i in processed[x]])

    fpf.refresh()
    return fpf, cost


def delta_roll(pf, op, roll_val, vdf, pdf, flag, slippage=None, brokerage=None):
    """Helper function that deals with delta-rolling options.

    Args:
        op (TYPE): Description
        roll_val (TYPE): Description
        slippage (None, optional): Description
        brokerage (None, optional): Description

    Returns:
        TYPE: Description
    """
    # print('delta not in bounds: ', op, abs(op.delta) / op.lots)

    # print('handling ' + str(op) + ' from family ' + pf.name)
    print('family rolling conds: ', pf.hedge_params['delta'])

    cost = 0

    vol_id = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()
    newop = create_vanilla_option(vdf, pdf, vol_id, op.char, op.shorted,
                                  lots=op.lots, delta=roll_val)

    # handle expenses: brokerage and old op price - new op price

    val = -(op.compute_price() - newop.compute_price())
    cost += val

    if brokerage:
        cost += (brokerage * (op.lots + newop.lots))

    if slippage:
        ttm = newop.tau * 365
        if ttm < 60:
            s_val = slippage[0]
        elif ttm >= 60 and ttm < 120:
            s_val = slippage[1]
        else:
            s_val = slippage[-1]
        cost += (s_val * (newop.lots + op.lots))

    pf.remove_security([op], flag)
    pf.add_security([newop], flag)

    return newop, op, cost


def check_roll_status(pf):
    """Checks to see if delta-roll conditions, if they exist, are satisfied.

    Args:
        pf (portfolio): Portfolio object
        hedges (dictionary): dictionary of hedges.

    Returns:
        boolean: True if delta is within bounds, false otherwise.
    """
    fa_lst = pf.get_families() if pf.get_families() else [pf]
    bool_list = [True] * len(fa_lst)
    for fa in fa_lst:
        hedges = fa.hedge_params
        index = fa_lst.index(fa)
        delta_conds = hedges['delta'] if 'delta' in hedges else None
        found = False
        dval = None
        if delta_conds is None:
            continue
        else:
            roll_cond = [x for x in delta_conds if x[0] == 'roll']
            if roll_cond:
                # search for the roll condition
                roll_cond = roll_cond[0]
                rollbounds = np.array(roll_cond[3]) / 100
                dval = roll_cond[1]
                found = True

        # if roll conditions actually exist, proceed.
        if found:
            for op in fa.get_all_options():
                d = abs(op.delta / op.lots)
                diff = (d - dval/100)
                ltol, utol = rollbounds[0], rollbounds[1]
                # sanity check
                if diff < ltol or diff > utol:
                    bool_list[index] = False

    return all([i for i in bool_list])

#######################################################################
#######################################################################
#######################################################################


# Steps that are run when simulation.py is run explicitly. In this case,
# parameters are assumed to be defined in scripts/global_vars.py rather
# than passed into to the simulation function. If calling simulation.py
# functions in an external scripts, global vars is not intrisically
# relevant, but can be used to specify filepaths which are then called.

# if __name__ == '__main__':

#     #################### initializing default params ###################

#     # fix portfolio start date #
#     start_date = gv.start_date
#     # filepath to portfolio specs. #
#     specpath = gv.portfolio_path
#     # fix end date of simulation #
#     end_date = gv.end_date
#     # path to hedging conditions #
#     hedge_path = gv.hedge_path
#     # final paths to datasets #
#     volpath, pricepath, exppath, sigpath = gv.final_paths
#     # raw paths
#     epath = gv.raw_exp_path

#     ####################################################################

#     print('--------------------------------------------------------')
#     print('DEFAULT PARAMETERS INITIALIZED. READING IN DATA... [1/7]')
#     ##########################################################################

#     t = time.clock()

#     writeflag = 'small' if 'small' in volpath else 'full'
#     print('writeflag: ', writeflag)

#     # printing existence of the paths
#     print('vol: ', os.path.exists(volpath))
#     print('price: ', os.path.exists(pricepath))
#     print('exp: ', os.path.exists(exppath))
#     # print('roll: ', os.path.exists(rollpath))

#     # error check in case signals are not being used this sim
#     signals = None if sigpath is None else pd.read_csv(sigpath)
#     if signals is not None:
#         signals.value_date = pd.to_datetime(signals.value_date)

#     # if data exists/has been prepared/saved, reads it in.
#     if os.path.exists(volpath) and os.path.exists(pricepath)\
#             and os.path.exists(exppath):
#             # and os.path.exists(rollpath):
#         print('--------------------------------------------------------')
#         print('DATA EXISTS. READING IN... [2/7]')
#         print('datasets listed below: ')
#         print('voldata : ', volpath)
#         print('pricepath: ', pricepath)
#         print('exppath: ', exppath)
#         # print('rollpath: ', rollpath)

#         vdf, pdf, edf = pd.read_csv(volpath), pd.read_csv(
#             pricepath), pd.read_csv(exppath)
#         # rolldf = pd.read_csv(rollpath)
#         if signals is not None:
#             print('sigpath: ', sigpath)

#         print('--------------------------------------------------------')

#         # sorting out date types
#         vdf.value_date = pd.to_datetime(vdf.value_date)
#         pdf.value_date = pd.to_datetime(pdf.value_date)
#         edf.expiry_date = pd.to_datetime(edf.expiry_date)

#         vdf, pdf, edf, roll_df, start_date = prep_datasets(
# vdf, pdf, edf, start_date, end_date, gv.pdt, signals=signals,
# test=False, write=True)

#     # if data does not exist as specified, reads in the relevant data.
#     else:
#         print('--------------------------------------------------------')
#         print('current_dir: ', os.getcwd())
#         print('DATA NOT FOUND. PREPARING... [2/7]')

#         print('paths inputted: ')
#         print('voldata : ', volpath)
#         print('pricepath: ', pricepath)
#         print('exppath: ', exppath)
#         print('--------------------------------------------------------')

#         if sigpath:
#             signals = pd.read_csv(sigpath)
#             signals.value_date = pd.to_datetime(signals.value_date)
#         else:
#             signals = None

#         vdf, pdf, raw_data = pull_alt_data(gv.pdt)
#         edf = pd.read_csv(epath)
#         edf.expiry_date = pd.to_datetime(edf.expiry_date)

#         vdf, pdf, edf, rolldf, alt_start_date = prep_datasets(
# vdf, pdf, edf, start_date, end_date, gv.pdt, signals=signals,
# test=False, write=True)

#         # vdf, pdf, edf, rolldf = read_data(
#         # epath, specpath, signals=signals, test=False,  writeflag=writeflag,
#         # start_date=start_date, end_date=end_date)

#     print('DATA READ-IN COMPLETE. SANITY CHECKING... [3/7]')
#     print('READ-IN RUNTIME: ', time.clock() - t)

#     ######################### check sanity of data #####################

#     # handle data types.
#     vdates = pd.to_datetime(sorted(vdf.value_date.unique()))
#     pdates = pd.to_datetime(sorted(pdf.value_date.unique()))
#     if signals is not None:
#         sig_dates = pd.to_datetime(sorted(signals.value_date.unique()))

#     # check to see that date ranges are equivalent for both price and vol data
#     sanity_check(vdates, pdates, start_date, end_date, signals=signals)

#     print('--------------------------------------------------------')
#     print('SANITY CHECKING COMPLETE. PREPPING PORTFOLIO... [4/7]')
#     ##########################################################################

#     # TODO: change this completely.
#     # generate portfolio #
#     pf, _ = prep_portfolio(vdf, pdf, filepath=specpath)

#     print('--------------------------------------------------------')
#     print('PORTFOLIO PREPARED. GENERATING HEDGES... [5/7]')

#     # TODO: need to find a new use for this.
#     # generate hedges #
#     hedges, roll_portfolio, pf_ttm_tol, pf_roll_product, \
#         roll_hedges, h_ttm_tol, h_roll_product = generate_hedges(hedge_path)

#     e1 = time.clock() - t
#     print('TOTAL PREP TIME: ', e1)

#     ###############################################################
#     # print statements for informational purposes #
#     print('START DATE: ', start_date)
#     print('END DATE: ', end_date)
#     print('Portfolio: ', pf)
#     print('NUM OPS: ', len(pf.OTC_options))
#     print('Hedge Conditions: ', hedges)
#     print('Brokerage: ', gv.brokerage)
#     print('Slippage: ', gv.slippage)
#     print('--------------------------------------------------------')
#     print('HEDGES GENERATED. RUNNING SIMULATION... [6/7]')

#     # # run simulation #
#     log = run_simulation(vdf, pdf, edf, pf,
# brokerage=gv.brokerage, slippage=gv.slippage, signals=signals)


##########################################################################
##########################################################################
##########################################################################

##########################################################################
##########################################################################
##########################################################################

##########################################################################
##########################################################################
##########################################################################
