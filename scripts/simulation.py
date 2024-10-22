# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2017-03-07 21:31:13
# @Last Modified by:   RMS08
# @Last Modified time: 2018-10-31 16:54:49


################################ imports ###################################
# general imports
import numpy as np
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

from pandas.tseries.offsets import BDay

# user defined imports
from .util import create_underlying, create_vanilla_option, close_out_deltas, create_composites, blockPrint, enablePrint
from .util import compute_market_minus, mark_to_vols, hedging_volid_handler
from .prep_data import reorder_ohlc_data, granularize
from .calc import _compute_value, get_vol_at_strike
from .signals import apply_signal
import datetime

# blockPrint()
# enablePrint()
###########################################################################
######################## initializing variables ###########################
###########################################################################
# Dictionary of multipliers for greeks/pnl calculation.
# format  =  'product' : [dollar_mult, lot_mult, futures_tick,
# options_tick, pnl_mult]

multipliers = {

    'LH':  [22.046, 18.143881, 0.025, 1, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 10, 50],
    'C':   [0.393678571428571, 127.007166832986, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'CA': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}

op_ticksize = {

    'QC': 1,
    'CC': 1,
    'SB': 0.01,
    'LSU': 0.05,
    'KC': 0.01,
    'DF': 1,
    'CT': 0.01,
    'C': 0.125,
    'S': 0.125,
    'SM': 0.05,
    'BO': 0.005,
    'W': 0.125,
    'MW': 0.125,
    'KW': 0.125
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
    'CA': ['H', 'K', 'U', 'Z'],
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

# 3 modes:
# 1) HSPS: _H_edge basis _S_ettlements, compute _P_nL basis _S_ettlements
# 2) HBPB: _H_edge basis _B_ook, compute _P_nL basis _B_ook vols with EOS update.
# 3) HBPS: _H_edge basis _B_ook, compute _P_nL basis _S_ettlements

#####################################################
############## Main Simulation Loop #################
#####################################################


def run_simulation(voldata, pricedata, pf, flat_vols=False, flat_price=False,
                   end_date=None, brokerage=None, slippage=None, signals=None,
                   plot_results=True, drawdown_limit=None, mode='HSPS', mkt_minus=1e7,
                   ohlc=False, remark_on_roll=False, remark_at_end=False,
                   roll_hedges_only=False, same_month_exception=False, verbose=True):
    """
    Each run of the simulation consists of the following steps:
        > for each day:
            - granularize the day's data. 
            > for each timestamp within the day's data:
                - feed data. 
                - intraday delta hedges. 
                - day's pnl += timestamp pnl 
            > Apply signals (if applicable)
            > contract roll (if applicable)
            > rebalance portfolio (eod delta hedges + hedging other greeks)
            > proceed to next day. 
    
    
    Args:
        voldata (pandas dataframe): Dataframe of volatilities (strikewise)
        pricedata (pandas dataframe): Dataframe of prices (either intraday or settlement-to-settlement)
        pf (portfolio object): The portfolio being run through the simulation
        flat_vols (bool, optional): boolean flag indicating if vols are to be kept constant
        flat_price (bool, optional): boolean flag indicating if prices are to be kept constant
        end_date (pandas datetime, optional): end date of the simulation if so desired.
        brokerage (float, optional): brokerage value.
        slippage (float, optional): slippage value.
        signals (pandas dataframe): dataframe of signals if they are applicable.
        plot_results (bool, optional): boolean flag indicating if results are to be plotted.
        drawdown_limit (float, optional): surpassing this limit causes the simulation to terminate.
        mode (str, optional): Determines what mode the simulation is run in.
        mkt_minus (float, optional): Market minuses if hedging is done basis book vols.
        ohlc (bool, optional): flag indicating if data is open-high-low-close.
        remark_on_roll (bool, optional): flag indicating if book vols are remarked to settlements on contract rolls.
        remark_at_end (bool, optional): flag indicating if book vols are remarked to settlement vols on end of simulation.
        roll_hedges_only (bool, optional): boolean that indicates if only hedges are to be rolled. else, if the pf has a 'roll' value, it will roll all options. 
        same_month_exception (bool, optional): boolean that indicates if a same-month roll exception holds for hedges. 
    
    Returns:
        dataframe: dataframe consisting of the logs on each day of the simulation.
    
    
    Raises:
        AssertionError: occurs if intraday loop isn't filtered appropriately for UIDs. 
        ValueError: if contract rolling is attempted but the price and volatility dataframes do not have the relevant information
                  :  if delta-rolling is attempted on an option's partner, but that partner belongs to a non-existent family. 
    
    """
    if not verbose:
        blockPrint()

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
    date_range = sorted(voldata.value_date.unique())

    ########################################
    # constructing/assigning hedge objects #
    book = True if mode in ('HBPS', 'HBPB') else False
    # pf = assign_hedge_objects(pf, book=book, slippage=slippage)

    print('slippage dict: ', pf.get_hedger().s)

    # initial timestep, because previous day's settlements were used to construct
    # the portfolio.
    # get the init diff and timestep
    # init_val = pf.compute_value()
    init_diff = (pd.to_datetime(date_range[1]) - 
                 pd.to_datetime(date_range[0])).days
    print('init diff: ', init_diff)
    # print('initial ttms: ', np.array(pf.OTC_options[0].get_ttms())*365)
    pf.timestep(init_diff * timestep)
    # pf.remove_expired()
    # pf = hedge_all_deltas(pf, pricedata)
    init_val = pf.compute_value()
    # print('post init ttms: ', np.array(pf.OTC_options[0].get_ttms())*365)

    # print('sim_start BOD init_value: ', init_val)
    # assign book vols; defaults to initialization date.
    book_vols = voldata[voldata.value_date ==
                        pd.to_datetime(date_range[0])]
    print('book vols initialized at ' +
          pd.to_datetime(date_range[0]).strftime('%Y-%m-%d'))
    # reassign the dates since the first datapoint is the init date.
    date_range = date_range[1:]
    ########################################

    ############## Bookkeeping #############    
    try:
        pricedata.time = pd.to_datetime(pricedata.time).dt.time
    except TypeError as e:
        print('time in pricedata is already in the right format.')
    voldata.loc[voldata.datatype == 'settlement', 'time'] = datetime.time.max
    pricedata.loc[pricedata.datatype == 'settlement', 'time'] = datetime.time.max
    ########################################

    ############ Other useful variables: #############
    loglist = []
    thetas = []
    breakevens = []

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

    # print('sigvals: ', sigvals)
    # boolean flag indicating missing data
    # Note: [partially depreciated]
    hedges_hit = []
    ##################################################

    ########### identifying simulation mode ###########
    if mode in ('HBPS', 'HBPB') or flat_vols:
        flat_vols = True
    else:
        flat_vols = False

    print('flatvols: ', flat_vols)
    print('auto detect volids: ', pf.get_hedger().auto_detect_volids())
    ###################################################

    for i in range(len(date_range)):
        # get the current date
        date = date_range[i]
        dailycost = 0

    # Steps 1 & 2: Error checks to prevent useless simulation runs.
        # checks to make sure if there are still non-hedge securities in pf
        # isolate data relevant for this day.
        date = pd.to_datetime(date)
        print('########################### date: ',
              date, '############################')

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

            if curr_pnl > highest_value:
                print('net pnl exceeds highest value, resetting drawdown benchmark.')
                highest_value = netpnl

            if highest_value - curr_pnl >= drawdown_limit:
                print('Current Drawdown: ', curr_pnl - highest_value)
                print('DRAWDOWN LIMIT HAS BEEN BREACHED. ENDING SIMULATION...')
                break
            else:
                print('Current Drawdown: ', curr_pnl - highest_value)
                print('Current Drawdown Percentage: ',
                      ((highest_value - curr_pnl)/(drawdown_limit)) * 100)

                print('Drawdown within bounds. Continuing...')

        # try to get next date
        try:
            next_date = pd.to_datetime(date_range[i + 1])
        except IndexError:
            next_date = None

        # filter data specific to the current day of the simulation.
        vdf_1 = voldata[voldata.value_date == date]
        pdf_1 = pricedata[pricedata.value_date == date]

        print("========================= INIT ==========================")
        print('Portfolio before any ops pf:', pf)
        print('init val BOD: ', init_val)
        log_pf = copy.deepcopy(pf)
        # print('ttms: ', np.array(pf.OTC_options[0].get_ttms()) * 365)
        print("==========================================================")

        dailypnl = 0
        dailygamma = 0
        dailyvega = 0
        exercise_futures = []
        barrier_futures = []

        print('updating book vols')
        book_vols.value_date = pd.to_datetime(date)
        book_vols.tau = (pd.to_datetime(book_vols.expdate) -
                         book_vols.value_date).dt.days/365
        print('finished updating book vol date to ' +
              pd.to_datetime(book_vols.value_date.unique()[0]).strftime('%Y-%m-%d'))

        # reset the breakeven dictionaries in the portfolio's hedger object.
        pf.update_hedger_breakeven()
        print('breakeven for the day: ', pf.hedger.breakeven)
        print('last hedgepoints: ', pf.hedger.last_hedgepoints)

        # grab and store the settlement vols/prices so intraday data
        # can be granularized with no problems, and saving for rebalancing/rollover 
        # steps if needed.
        settle_vols = vdf_1[vdf_1.datatype == 'settlement'].copy()
        settle_prices = pdf_1[pdf_1.datatype == 'settlement'].copy()

        # filter dataframes to get only pertinent UIDs data.
        pdf_1 = pdf_1[pdf_1.underlying_id.isin(
            pf.get_unique_uids())].reset_index(drop=True)
        vdf_1 = vdf_1[vdf_1.vol_id.isin(
            pf.get_unique_volids())].reset_index(drop=True)

        # optional parameter if OHLC is used, indicating the order in which entries were sorted.
        data_order = None

        # need this check because for intraday/settlement to settlement, no reordering
        # is required; granularize is called in OHLC case only after an order
        # has been determined
        if not ohlc:
            print('@@@@@@@@@@@@@@@@@ Granularizing: Intraday Case @@@@@@@@@@@@@@@@')
            pdf_1 = granularize(pdf_1, pf, intraday=True)
            if pdf_1.empty:
                print('DATAFRAME IS EMPTY')
                # continue
            # print('vdf_1: ', pdf_1)
            try:
                assert len(pdf_1.underlying_id.unique()) == 1
            except AssertionError as e:
                if 'intraday' in pdf_1.datatype.unique():
                    raise AssertionError("dataset not filtered for UIDS on " + str(date) + " : ",
                                         pdf_1.underlying_id.unique()) from e

        print('================ beginning intraday loop =====================')
        unique_ts = pdf_1.time.unique()
        # bookkeeping variable. 
        dailyhedges = []
        for ts in unique_ts:
            pdf = pdf_1[pdf_1.time == ts]
            pdf.reset_index(drop=True, inplace=True)
            print('pdf: ', pdf)
            if ohlc:
                print('@@@@@@@@@@@@@@@ OHLC STEP GRANULARIZING @@@@@@@@@@@@@@@@')
                init_pdf, pdf, data_order = reorder_ohlc_data(pdf, pf)
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # currently, vdf only exists for settlement anyway.
            vdf = vdf_1[vdf_1.time == ts]
            # if settlement data, pass in the entire dataframe. if not, pass in each individual timestamp. 
            datapoints = [pdf] if 'intraday' not in pdf.datatype.unique() else [pdf.iloc[i] for i in pdf.index]
            for pdf_ts in datapoints:
                # get the current row and variables
                datatype = pdf_ts.datatype.values[0]
                uid = pdf_ts.underlying_id.values[0]
                # OHLC case: if the UID is not in the portfolio, skip.
                if uid not in pf.get_unique_uids():
                    continue
                val = pdf_ts.price.values[0]
                diff = 0
                if ohlc:
                    # since OHLC timestamps are set after reorder,
                    # this is required to select settlement vols only
                    # when handling settlement price.
                    vdf = vdf_1[vdf_1.time == pdf_ts.time.values[0]]
                lp = pf.hedger.last_hedgepoints[uid]
                print('===================== time: ' +
                      str(pdf_ts.time.values[0]) + ' =====================')

                print('%s price move to %s for uid last hedged at %s' % (datatype, val, lp))
                if datatype == 'intraday':
                    dailyhedges.append({'date': date, 
                                        'time': ts, 
                                        'uid': uid, 
                                        'hedge point': val})

                pf.assign_hedger_dataframes(vdf, pdf_ts)

            # Step 3: Feed data into the portfolio.
                print("========================= FEED DATA ==========================")
                # NOTE: currently, exercising happens as soon as moneyness is triggered.
                # This should not be much of an issue since exercise is never
                # actually reached.

                pf, gamma_pnl, vega_pnl, exercise_barrier_profit, exercise_futures, barrier_futures, price_change, vol_change, \
                    = feed_data(vdf, pdf_ts, pf, init_val, flat_vols=flat_vols, flat_price=flat_price)

                print("==================== PNL & BARR/EX =====================")

            # Step 4: Compute pnl for the this timestep.
                updated_val = pf.compute_value()
                # sanity check: if the portfolio is closed out during this
                # timestep, pnl = exercise proft.
                if pf.empty():
                    pnl = exercise_barrier_profit
                else:
                    pnl = (updated_val - init_val) + exercise_barrier_profit

                print('timestamp pnl: ', pnl)
                print('timestamp gamma pnl: ', gamma_pnl)
                print('timestamp vega pnl: ', vega_pnl)

                # update the daily variables.
                dailypnl += pnl
                dailygamma += gamma_pnl
                dailyvega += vega_pnl

                # print('pf before adding bar fts/ex fts: ', pf)

                # Detour: add in exercise & barrier futures if required.
                if exercise_futures:
                    pf.add_security(exercise_futures, 'OTC')
                    
                if barrier_futures:
                    pf.add_security(barrier_futures, 'hedge')

                # check: if type is settlement, defer EOD delta hedging to
                # rebalance function.
                if 'settlement' not in pdf_ts.datatype.unique():
                    print('--- intraday hedge case ---')
                    hedge_engine = pf.get_hedger()
                    pnl = hedge_engine.apply(
                        'delta', intraday=True, ohlc=ohlc)
                    dailypnl += pnl
                    print('--- updating init_val post hedge ---')
                    init_val = pf.compute_value()
                    print('end timestamp init val: ', init_val)

                else:
                    init_val = pf.compute_value()
                    print('settlement delta case. deferring hedging to rebalance step.')
                    continue

                print("================================================================")
                print('pf at end: ', pf)
                print('============= end timestamp ' +
                      str(pdf_ts.time.values[0]) + '===================')

        dailyhedges = pd.DataFrame(dailyhedges)
        hedges_hit.append(dailyhedges)
        print('================ end intraday loop =====================')
        print('pf after intraday loop: ', pf)

        pf.remove_expired()

        # case: if mode = HBPS, update current dailypnl value to factor in
        # settlement vols.
        if mode == 'HBPS':
            print('HBPS Mode: calculating PnL')
            tmp_pf = mark_to_vols(pf, settle_vols, dup=True)
            # any change in value will be due to vol updates, since flatvols were
            # used in update loop.
            dailyvega = tmp_pf.compute_value() - init_val
            print('dailyvega: ', dailyvega)
            print('dailygamma: ', dailygamma)
            dailypnl += dailyvega
            print('dailypnl: ', dailypnl)
            print('total = gamma + vega: ', dailypnl, dailyvega + dailygamma)

    # Step 5: Apply signal
        if signals is not None:
            print("========================= SIGNALS ==========================")
            relevant_signals = signals[signals.value_date == date]
            pf, cost, sigvals = apply_signal(pf, settle_prices, settle_vols, relevant_signals, date,
                                             sigvals, strat='filo',
                                             brokerage=brokerage, slippage=slippage)
            print('signals cost: ', cost)
            dailycost += cost
            print("====================================================================")

    # Step 6: rolling over portfolio and hedges if required.
        # simple case
        if mode == 'HSPS':
            print('HSPS Mode... hedging basis settlements')
            h_vdf = settle_vols
        else:
            print(mode + ' Mode... hedging basis book')
            h_vdf = book_vols

        print("=========================== ROLLOVER ============================")
        roll_vdf = settle_vols if remark_on_roll else h_vdf
        print('auto_detect vol_ids: ', pf.get_hedger().auto_detect_volids())

        pf, cost, all_deltas = roll_over(pf, roll_vdf, settle_prices, date, brokerage=brokerage,
                                         slippage=slippage, hedges_only=roll_hedges_only, 
                                         same_month_exception=same_month_exception)
        pf.refresh()
        print('roll_over cost: ', cost)
        dailycost += cost

        if remark_on_roll:
            pdts = [x[0] for x in all_deltas]
            print('init book vols: ', book_vols)
            if pdts:
                # isolate the sections of book vols that are not contained in
                # pdts .
                unaffected = book_vols[~book_vols.pdts.isin(pdts)]
                # for each affected product, isolate the vols and concatenate them
                # to the unaffected vols
                for pdt in pdts:
                    print('remarking on rollover; remarking book vols for ' + pdt)
                    curr_settles = settle_vols[settle_vols.pdt == pdt]
                    unaffected = pd.concat([unaffected, curr_settles])
                book_vols = unaffected
            print('new book vols: ', book_vols)

        print("===================================================================")

        pf.assign_hedger_dataframes(h_vdf, settle_prices, settles=settle_vols)

    # Step 7: Hedge - bring greek levels across portfolios (and families) into
    # line with target levels using specified vols/prices.
        print("========================= REBALANCE ==========================")
        # print('settle prices before rebalance: ', settle_prices)
        pf, cost, roll_hedged = rebalance(h_vdf, settle_prices, pf, brokerage=brokerage,
                                          slippage=slippage, next_date=next_date,
                                          settlements=settle_vols, book=book)
        print('rebalance cost: ', cost)
        dailycost += cost
        print("==================================================================")
        
    # Step 8: compute market minuses, isolate if end of sim.
        mm, diff = compute_market_minus(pf, settle_vols)
        print('market minuses for the day: ', mm, diff)
        if end_date is not None:
            end_sim = (next_date == date_range[-1] or next_date ==
                       end_date or (date < end_date and next_date > end_date))
        else:
            end_sim = date == date_range[-1]

    # Step 9: timestep before computing PnL. store and compute the theta
    # debit/credit. update dailypnl and dailygamma with theta debit.
        if next_date is None:
            # step to the next business day.
            num_days = ((pd.Timestamp(date) + BDay(1)) -
                        pd.Timestamp(date)).days
        else:
            num_days = (pd.Timestamp(next_date) - pd.Timestamp(date)).days

        print('actual timestep of ' + str(num_days))

        # isolate the change due to timestep (i.e. theta debit/credit)

        pre_timestep_value = pf.compute_value()
        pf.timestep(num_days * timestep)
        theta_change = pf.compute_value() - pre_timestep_value

        # standardize to big theta for hedging optimization purposes.
        if num_days == 1:
            thetas.append(theta_change * 1.4)
        else:
            # make sure this works for usual holidays as well.
            thetas.append((theta_change/num_days) * 1.4)

        print('dailypnl before theta change: ', dailypnl)
        print('dailygamma before theta change: ', dailygamma)

        dailypnl += theta_change
        dailygamma += theta_change

        print('theta debit/credit: ', theta_change)

        if mode == 'HSPS':
            init_val = pf.compute_value()
            print('HSPS Init val: ', init_val)

        elif mode == 'HBPB':
            # last day of the simulation, or market minus exceeded, or contract
            # rolled
            print('HBPB Mode: handling PnLs')
            print('date: ', date)
            print('end_date: ', end_date)
            print('next_date: ', next_date)
            print('end_sim: ', end_sim)

            if (end_sim and remark_at_end) or (mm >= mkt_minus):
                tmp = pf.compute_value()
                pf = mark_to_vols(pf, settle_vols)
                newval = pf.compute_value()
                dailyvega += newval - tmp
                # reset book vols
                print('dailyvega: ', dailyvega)
                # temporarily skip the final remark
                dailypnl += dailyvega
                book_vols = settle_vols

            init_val = pf.compute_value()
            print('HBPB Init val: ', init_val)

        else:
            # update from book to settlement vols to compute
            # initial value for next loop. pf_settle = pf_book + market_minus
            newpf = mark_to_vols(pf, settle_vols, dup=True)
            init_val = newpf.compute_value()
            print('HBPS Init val: ', init_val)

    # Step 10: reset book vols if the market minuses are too great, or if
    # contract roll occurred. remark portfolio to book vols.
    # value differential is counted  as vega pnl
        if (mode in ('HBPB', 'HBPS')) and (mm >= mkt_minus):
            # reset the book vols
            tmp = pf.compute_value()
            book_vols = settle_vols
            print('resetting book vols on ',
                  pd.to_datetime(book_vols.value_date.unique()[0]))

            pf = mark_to_vols(pf, book_vols)
            dailyvega += pf.compute_value() - tmp

    # Step 11: Subtract brokerage/slippage costs from rebalancing. Append to
    # relevant lists.
        # gamma/vega pnls
        gamma_pnl_daily.append(dailygamma)
        vega_pnl_daily.append(dailyvega)
        gammapnl += dailygamma
        vegapnl += dailyvega
        gamma_pnl_cumul.append(gammapnl)
        vega_pnl_cumul.append(vegapnl)
        # gross/net pnls
        gross_daily_values.append(dailypnl)
        dailynet = dailypnl - dailycost
        netpnl += dailynet
        net_daily_values.append(dailynet)
        grosspnl += dailypnl
        gross_cumul_values.append(grosspnl)
        net_cumul_values.append(netpnl)

    # Step 13: Logging relevant output to csv
    ##########################################################################
        print("========================= LOG WRITING  ==========================")
        dailylog = write_log(log_pf, drawdown_limit, date, dailypnl, dailynet, grosspnl, netpnl, dailygamma, 
                             gammapnl, dailyvega, vegapnl, roll_hedged, data_order, num_days,
                             highest_value, net_cumul_values, breakevens, dailypnl, dailycost, vol_change, price_change)
        loglist.extend(dailylog)
        print("========================= END LOG WRITING ==========================")

        print('[1.0]   EOD PNL (GROSS): ', dailypnl)
        print('[1.0.1] EOD Vega PNL: ', dailyvega)
        print('[1.0.2] EOD Gamma PNL: ', dailygamma)
        print('[1.0.2.5] Daily cost: ', dailycost)
        print('[1.0.3] EOD PNL (NET) :', dailynet)
        print('[1.0.4] Cumulative PNL (GROSS): ', grosspnl)
        print('[1.0.5] Cumulative Vega PNL: ', vegapnl)
        print('[1.0.6] Cumulative Gamma PNL: ', gammapnl)
        print('[1.0.7] Cumulative PNL (net): ', netpnl)
        print('[1.1]  EOD PORTFOLIO: ', pf)

    # Step 10: Plotting results/data viz
    ######################### PRINTING OUTPUT ###########################
    log = pd.DataFrame(loglist)

    # append the open high low close if applicable
    if ohlc:
        print('merging OHLC data with logs...', end="")
        tdf = pricedata[['value_date', 'underlying_id', 'price', 'price_id']]
        opens = tdf[tdf.price_id == 'px_open']
        print('opens columns: ', opens.columns)
        opens.columns = ['px_open' if x ==
                         'price' else x for x in opens.columns]
        opens = opens.drop('price_id', axis=1)
        highs = tdf[tdf.price_id == 'px_high']
        highs.columns = ['px_high' if x ==
                         'price' else x for x in highs.columns]
        highs = highs.drop('price_id', axis=1)
        lows = tdf[tdf.price_id == 'px_low']
        lows.columns = ['px_low' if x == 'price' else x for x in lows.columns]
        lows = lows.drop('price_id', axis=1)
        # close = tdf[tdf.price_id == 'px_settle']
        # join each to the log.
        log = pd.merge(log, opens, on=['value_date', 'underlying_id'])
        log = pd.merge(log, highs, on=['value_date', 'underlying_id'])
        log = pd.merge(log, lows, on=['value_date', 'underlying_id'])
        print('done.')

    # appending 25d vol changes and price changes
    if signals is not None:
        signals.loc[signals['call/put'] == 'call', 'call_put_id'] = 'C'
        signals.loc[signals['call/put'] == 'put', 'call_put_id'] = 'P'

        df = pd.merge(signals, pricedata[['value_date', 'underlying_id',
                                          'price', 'vol_id', 'call_put_id']],
                      on=['value_date', 'vol_id', 'call_put_id'])
        df = df.drop_duplicates()

        log = pd.merge(log, df[['value_date', 'vol_id', 'price',
                                'signal']],
                       on=['value_date', 'vol_id'])

    elapsed = time.clock() - t

    print('Time elapsed: ', elapsed)

    print('################# Portfolio: ###################')
    print(pf)

    if not verbose:
        enablePrint()

    print('SIMULATION COMPLETE. PRINTING RELEVANT OUTPUT... [7/7]')

    print('##################### PNL: #####################')
    print('gross pnl: ', grosspnl)
    print('net pnl: ', netpnl)
    print('vega pnl: ', vegapnl)
    print('gamma pnl: ', gammapnl)
    print('total theta paid: ', sum(thetas))
    print('gamma money: ', grosspnl - sum(thetas))
    print('vols remarked at end: ', remark_at_end)
    print('vols remarked on contract roll: ', remark_on_roll)
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
        yrs = log.value_date.dt.year.unique()
        yr_str = '-'.join([str(x) for x in yrs])

        # plotting gross pnl values
        plt.figure()
        colors = ['c' if x >= 0 else 'r' for x in gross_daily_values]
        # xvals = list(range(1, len(gross_daily_values) + 1))
        xvals = list(pd.to_datetime(log.value_date.unique()))

        plt.bar(xvals, net_daily_values, align='center',
                color=colors, alpha=0.6, label='net daily values')
        plt.plot(xvals, gross_cumul_values, c='b',
                 alpha=0.8, label='gross cumulative pnl')
        plt.plot(xvals, net_cumul_values, c='r',
                 alpha=0.8, label='net cumulative pnl')
        plt.plot(xvals, gamma_pnl_cumul, c='g',
                 alpha=0.5, label='cu. gamma pnl')
        plt.plot(xvals, vega_pnl_cumul, c='y', alpha=0.5, label='cu. vega pnl')
        plt.title('gross/net pnl daily ' + yr_str)
        plt.legend()
        plt.show()

    days = list(pd.to_datetime(log.value_date.unique()))
    hedges_hit = pd.concat(hedges_hit)
    theta_paid = sum(thetas)
    gamma_money = grosspnl - theta_paid

    return log, net_cumul_values[-1], hedges_hit, gamma_money,\
        theta_paid, thetas, sum(gamma_pnl_daily), gamma_pnl_daily, breakevens, \
        (days, net_daily_values, gross_cumul_values,
         net_cumul_values, gamma_pnl_cumul, vega_pnl_cumul)


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
        init_val (float): the initial value of the portfolio
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
        the Portfolio.compute_value function.

    """
    barrier_futures = []
    exercise_futures = []
    exercise_profit = 0
    barrier_profit = 0
    price_change = {}
    vol_change = {}
    bvolchange = {}

    # print('pdf: ', pdf)

    if voldf.empty:
        print("Volatility Dataframe is empty!")

    print('voldf date: ', voldf.value_date.unique())

    # # 1) sanity checks
    # if pf.empty():
    #     return pf, False, 0, 0, 0, [], []

    # print('curr_date: ', curr_date)
    date = pd.to_datetime(pdf.value_date.unique()[0])

    # 2)  update prices of futures, underlying & portfolio alike.
    for ft in pf.get_all_futures():
        pdt = ft.get_product()
        uid = ft.get_uid()
        # try:
            # case: flat price.
        if flat_price:
            continue
        else:
            try:
                val = pdf[(pdf.pdt == pdt) &
                          (pdf.underlying_id == uid)].price.values[0]
            except IndexError as e:
                print('inputs: ', pdt, uid)
                print('pdf: ', pdf)
                raise ValueError("Something broke in feed price step.") from e
            pf, cost, fts = handle_barriers(
                voldf, pdf, ft, val, pf, date)
            barrier_futures.extend(fts)
            pchange = val - ft.get_price() 

            # print('updating price to ' + str(val))
            ft.update_price(val)
            if ft.get_uid() not in price_change:
                price_change[ft.get_uid()] = pchange

        # index error would occur only if data is missing.
        # except IndexError:
        #     ft.update_price(ft.get_price())

    # handling pnl arising from barrier futures.
    if barrier_futures:
        for ft in barrier_futures:
            pdt = ft.get_product()
            pnl_mult = multipliers[pdt][-1]
            try:
                uid = ft.get_product() + '  ' + ft.get_month()
                val = pdf[(pdf.pdt == pdt) &
                          (pdf.underlying_id == uid)].price.values[0]
                # calculate difference between ki price and current price
                diff = val - ft.get_price()
                pnl_diff = diff * ft.lots * pnl_mult
                barrier_profit += pnl_diff
                # update the price of each future
                ft.update_price(val)

            # index error would occur only if data is missing.
            except IndexError:
                ft.update_price(ft.get_price())
    pf.refresh()    

    print('uid price dict before exercise: ', pf.uid_price_dict())
    exercise_profit, pf, exercised, exercise_futures = handle_exercise(
        pf, brokerage, slippage)

    total_profit = exercise_profit + barrier_profit
    print('barrier + exercise profit: ', total_profit)

    # refresh portfolio after price updates.
    pf.refresh()
    pf.remove_expired()
    pf.refresh()

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
    if exercised:
        # case: portfolio is empty after data feed step (i.e. expiries and
        # associated deltas closed)
        if intermediate_val == 0:
            print('gamma pnl calc: portfolio closed out case')
            gamma_pnl = total_profit
        # case: portfolio is NOT empty.
        else:
            # print('other case')
            gamma_pnl = (intermediate_val + total_profit) - init_val
    else:
        print('pnl calc: not exercised case')
        print('intermediate_val: ', intermediate_val)
        print('init val: ', init_val)
        print('total_profit: ', total_profit)

        gamma_pnl = (intermediate_val + total_profit -
                     init_val) if intermediate_val != 0 else 0
        # print('gamma pnl: ', gamma_pnl)

    # skip feeding in vols if 1) data not present or 2) flat_vols flag is
    # triggered.
    volskip = True if flat_vols else False
    if volskip:
        print('Volskip is True')
        print('vdf datatype: ', voldf.datatype.unique())
        print('dataframe empty: ', voldf.empty)
        print('flatvols: ', flat_vols)
    if not volskip:
        # update option attributes by feeding in vol.
        for op in pf.get_all_options():
            # info reqd: strike, order, product, tau
            strike = op.K
            bvol2, b_vol, strike_vol = None, None, None
            # interpolate or round? currently rounding, interpolation easy.
            ticksize = multipliers[op.get_product()][-2]
            # get strike corresponding to closest available ticksize.
            strike = round(round(strike / ticksize) * ticksize, 2)
            vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()

            try:
                # val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                #             (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                # strike_vol = val.vol.values[0]
                strike_vol = get_vol_at_strike(voldf[voldf.vol_id == vid], strike)

            except (IndexError, ValueError):
                print('### VOLATILITY DATA MISSING ###')
                print(op)
                print('###############################')
                strike_vol = op.vol

            try:
                if op.barrier is not None:
                    barlevel = op.ki if op.ki is not None else op.ko
                    # b_val = voldf[(voldf.pdt == product) & (voldf.strike == barlevel) &
                    #               (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                    # print('inputs: ', product, barlevel, vid, cpi)
                    # b_vol = b_val.vol.values[0]
                    b_vol = get_vol_at_strike(voldf[voldf.vol_id == vid], barlevel)
                    # print('bvol: ', b_vol)
                # case to update the barrier vol of the tick-wide digital. 
                if op.barrier == 'euro':
                    assert op.dbarrier is not None 
                    bvol2 = get_vol_at_strike(voldf[voldf.vol_id == vid], op.dbarrier)

            except (IndexError, ValueError):
                print('### BARRIER VOLATILITY DATA MISSING ###')
                b_vol = op.bvol
                bvol2 = op.bvol2 
            # print('bvol: ', b_vol)
            # print('bvol2: ', bvol2)
            vol_change = save_vol_change(op, strike_vol, vol_change, 'strike')
            vol_change = save_vol_change(op, b_vol, vol_change, 'bvol')
            vol_change = save_vol_change(op, bvol2, vol_change, 'bvol2')

            op.update_greeks(vol=strike_vol, bvol=b_vol, bvol2=bvol2)

        pf.refresh()

    # print('pf after feed: ', pf)

    vega_pnl = pf.compute_value() - intermediate_val if intermediate_val != 0 else 0
    return pf, gamma_pnl, vega_pnl, total_profit, exercise_futures, barrier_futures, price_change, vol_change


def save_vol_change(op, new_vol, vol_change_dic, flag):
    volid = op.get_vol_id() 
    if flag == 'strike':
        old_vol = op.vol 
        strike = op.K 
    elif flag == 'bvol':
        old_vol = op.bvol 
        strike = op.ki if op.ki is not None else op.ko 
    elif flag == 'bvol2':
        old_vol = op.bvol2
        strike = op.ki if op.ki is not None else op.ko
    try:
        vol_change = new_vol - old_vol
    except TypeError as e:
        vol_change = 0

    if volid not in vol_change_dic:
        vol_change_dic[volid] = {} 
    vol_change_dic[volid][strike] = vol_change * 100
    return vol_change_dic


# TODO: Streamline this so that it calls comp functions and doesn't create new objects. 
def handle_barriers(vdf, pdf, ft, val, pf, date):
    """Handles delta differential and spot hedging for knockin/knockout events.

    Args:
        vdf (TYPE): dataframe of volatilities
        pdf (TYPE): dataframe of prices
        ft (TYPE): future object whose price is being updated
        val (TYPE): current price point
        pf (TYPE): portfolio being simulated.
        date (TYPE): current date

    Returns:
        tuple: unchanged portfolio, cost, futures accumulated as a result of barrier events.

    """

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

    # edge case where there is one daily op expiring today.
    if not op.is_bullet():
        if np.isclose(max(op.get_ttms()), 0):
            return pf, 0, []

    bar_op = copy.deepcopy(op)

    # create vanilla option with the same stats as the barrier option AFTER
    # knockin.
    volid = op.get_vol_id()
    # print('vdf Z8: ', vdf)

    vanop = create_vanilla_option(vdf, pdf, volid, op.char, op.shorted,
                                  lots=op.lots, vol=op.vol, strike=op.K,
                                  bullet=op.bullet)

    # case 1: knockin case - option is not currently knocked in.
    if op.ki is not None:
        # print('simulation.handle_barriers - knockin case')
        # case 1-1: di option, val is below barrier.
        if op.direc == 'down' and val <= op.ki:
            # print('simulation.handle_barriers - down-in case ' + str(op))
            step = ft_ticksize
        elif op.direc == 'up' and val >= op.ki:
            # print('simulation.handle_barriers - up-in case ' + str(op))
            step = -ft_ticksize
        # knockin case with no action
        else:
            # print('barriers handled')
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
        # print('simulation.handle_barriers - knockout case')
        if op.knockedout:
            print('simulation.handle_barriers - knockedout')
            ret = pf, 0, []

        else:
            # case 1: updating price to val will initiate a knockout
            if op.direc == 'up' and val > op.ko:
                # print('simulation.handle_barriers - up-out case ' + str(op))
                step = -ft_ticksize
            elif op.direc == 'down' and val < op.ko:
                # print('simulation.handle_barriers - down-out case ' + str(op))
                step = ft_ticksize
            else:
                # print('barriers handled')
                return pf, 0, []

            ft_price = round(
                round((op.ko + step) / ft_ticksize) * ft_ticksize, 2)
            bar_op.underlying.update_price(ft_price)
            # print('future price: ', ft_price)
            delta_diff = bar_op.delta
            # print('delta_diff: ', delta_diff)

            # creating the future object.
            ft_shorted = True if delta_diff < 0 else False
            lots_req = abs(round(delta_diff))
            fts, _ = create_underlying(pdt, ftmth, pdf, date, ftprice=ft_price,
                                       shorted=ft_shorted, lots=lots_req)

            ret = pf, 0, [fts]

    # regular vanilla option case; do nothing.
    else:
        # print('simulation.handle_barriers - final else case')
        ret = pf, 0, []

    # print('barriers handled')
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
        1) options are exercised if less than or equal to 1 days to maturity, and
        option is in the money. 
        2) Accepts both cash and future settlement

    """
    if pf.empty() or pf.ops_empty():
        return 0, pf, False, []
    exercised = False
    # t = time.clock()
    profit = 0
    tol = 1/365
    # handle options exercise
    all_ops = pf.get_all_options()
    # otc_ops = pf.OTC_options
    # hedge_ops = pf.hedge_options

    tobeadded = []
    for op in all_ops:
        # bullet case
        print('handle_exercise - future price: ', op.get_underlying().get_price())
        ttm = op.tau if op.is_bullet() else min(op.get_ttms())
        # print('ttm: ', np.array(op.get_ttms()) * 365)
        # print('exercise: ', op.exercise())
        if np.isclose(ttm, 0) or ttm < tol:
            exer = op.exercise()
            if op.is_bullet():
                op.tau = 0 
            else:
                op.get_ttms()[0] = 0
            if exer:

                print("exercising option " + str(op))
                if op.settlement == 'cash':
                    print("----- CASH SETTLEMENT ------")
                elif op.settlement == 'futures':
                    ft = op.get_underlying()
                    ft.update_lots(op.lots)
                    # print('future added - ', str(ft))
                    # print('lots: ', op.lots, ft.lots)
                    tobeadded.append(ft)

                # calculating the net profit from this exchange.
                product = op.get_product()
                pnl_mult = multipliers[product][-1]
                ftprice, strike = op.get_underlying().get_price(), op.K
                oppnl = op.lots * (ftprice - strike) * pnl_mult if op.char == 'call' else op.lots * (strike - ftprice) * pnl_mult
                if op.shorted: 
                    oppnl = -oppnl
                print('profit/loss on this exercise: ', oppnl)
                print("------------------------------")
                profit += oppnl
                exercised = True

    # debug statement:
    # print('handle_exercise - tobeadded: ', [str(x) for x in tobeadded])
    # print('handle_exercise - options exercised: ', exercised)
    # print('handle_exercise - net exercise profit: ', profit)
    # print('handle exercise time: ', time.clock() - t)
    return profit, pf, exercised, tobeadded


###############################################################################
###############################################################################
###############################################################################


###############################################################################
########### Hedging-related functions (generation and implementation) #########
###############################################################################


def roll_over(pf, vdf, pdf, date, brokerage=None, slippage=None, hedges_only=False, same_month_exception=False):
    """
    If ttm < ttm_tol, closes out that position (and all accumulated deltas),
    saves lot size/strikes, and rolls it over into the
    next month.
    
    Args:
        pf (TYPE): portfolio being hedged
        vdf (TYPE): volatility dataframe
        pdf (TYPE): price dataframe
        date (TYPE): Date to be used when accessing data from dataframes.
        brokerage (TYPE, optional): brokerage amount
        slippage (TYPE, optional): slippage amount
        hedges_only (bool, optional): boolean that indicates if only hedges are to be rolled.
        same_month_exception (bool, optional): if True, hedges are not rolled if they have the same product & month as OTC ops
    
    Returns:
        tuple: updated portfolio and cost of operations undertaken
    
    Deleted Parameters:
        ttm_tol (int, optional): tolerance level for time to maturity.
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

    any_rolled = False

    # iterate over each family.
    for fa in fa_lst:
        print('-------------- rolling options from ' +
              fa.name + ' -----------------')
        print('simulation.roll_over - fa: ', fa)
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
        dic = fa.get_hedge_options() if hedges_only else fa.get_all_options()

        # get list of all product-month combos in OTC options to address same month exception case.
        otc_params = set([(op.get_product(), op.get_month()) for op in pf.OTC_options])
        # otc_mth = set([op.get_month()] for op in pf.OTC_options)

        print('num_ops: ', len(dic))

        # get list of already processed ops for this family.
        processed_ops = processed[fa]

        # iterate over each option in this family
        for op in dic.copy():
            flag = 'hedge' if (hedges_only or op in fa.get_hedge_options()) else 'OTC'
            # case: op has already been processed since its parter was
            # processed.
            if op in processed_ops:
                print(str(op) + ' has been processed')
                continue
            composites = []

            # case: roll if ttm threshold is breached or roll_all is triggered.
            # print('op.tau: ', op.tau * 365)
            needtoroll = (((round(op.tau * 365) <= fa.ttm_tol) or
                           np.isclose(op.tau, fa.ttm_tol/365)) or
                          roll_all)

            # print('tolerance breached: ', round(op.tau * 365) <= fa.ttm_tol)
            # print('op.tau: ', op.tau * 365)
            # print('fa ttm tol: ', fa.ttm_tol)
            # print('ttm close: ', np.isclose(op.tau, fa.ttm_tol/365))

            # final check in the case of same_month_exception 
            if same_month_exception:
                if (op.get_product(), op.get_month()) in otc_params:
                    print('same month exception triggered')
                    needtoroll = False

            # print('needtoroll: ', needtoroll)
            if needtoroll:
                print('rolling option ' + str(op) + ' from ' + flag)
                toberemoved.append(op)
                rolled_vids.add(op.get_vol_id())
                # creating the underlying future object
                print('rolling op')
                fa, cost, newop, old_op, iden = contract_roll(
                    fa, op, vdf, pdf, date, flag, slippage=slippage)
                composites.append(newop)
                processed_ops.add(old_op)
                deltas_to_close.add(iden)
                total_cost += cost
                # if rolling, roll partners as well so that composite structure
                # can be maintained.
                print('handling partners...')
                deltas_to_close, composites, total_cost, processed, rolled_vids =\
                    roll_handle_partners(date, pf, fa, op, deltas_to_close,
                                         composites, total_cost, processed, vdf, pdf, rolled_vids)

                composites = create_composites(composites)
        # refresh family after iterating through its options.
        fa.refresh()
        print('-------------- ' + fa.name + ' handled -----------------')
    # refresh portfolio after first pass through options.
    pf.refresh()
    # edge case: in the case where a roll_product is specified and hedge
    # options are added, these hedge options may not be found in the above
    # loop since hedge options have no default behavior of being initialized
    # as partners with OTC options.
    edge_case_ops = set()
    print('rolled vids: ', rolled_vids)
    for vid in rolled_vids:
        ops = [op for op in pf.hedge_options if op.get_vol_id() == vid]
        # print('EDGE CASE OPS: ', ops)
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
                    fam, op, vdf, pdf, date, flag, slippage=slippage)
                # roll the partners
                deltas_to_close, composites, total_cost, processed, rolled_vids\
                    = roll_handle_partners(date, pf, fa, op, deltas_to_close,
                                           composites, total_cost, processed, vdf, pdf, rolled_vids, slippage=slippage)
                edge_case_ops.update(partners)
                create_composites(composites)

    # # refresh after handling options.
    # pf.refresh()
    # close out deltas associated with the rolled-over options
    pf, cost = close_out_deltas(pf, deltas_to_close)
    # refresh after handling the futures.
    pf.refresh()

    total_cost += cost

    print(' --- done contract rolling --- ')
    return pf, total_cost, deltas_to_close


def roll_handle_partners(date, pf, fa, op, deltas_to_close, composites, 
                         total_cost, processed, vdf, pdf, rolled_vids, slippage=None):
    """Helper method that handles partners when rolling options.
    
    Args:
        date (TYPE): current date within the simulation
        pf (TYPE): super-portfolio being handled.
        fa (TYPE): the family within pf being handled. if not pf.families, fa = pf.
        op (TYPE): the option whose partners are being handled.
        deltas_to_close (TYPE): a set of deltas to close so far, passed from roll_over function.
        composites (TYPE): a list of current composites associated with this family.
        total_cost (TYPE): total cost so far.
        processed (TYPE): dictionary mapping family -> list of processed options.
        vdf (TYPE): dataframe of volatilies.
        pdf (TYPE): dataframe of prices
        rolled_vids (TYPE): vol_ids that have already been rolled thus far.
        slippage (None, optional): Optional parameter that adds slippage risk, measured in ticks.
    
    Returns:
        tuple: updated deltas_to_close, composites, total_cost, processed and rolled_vids
    
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
                tar, opx, vdf, pdf, date, flag2, slippage=slippage)
            composites.append(new_opx)
            deltas_to_close.add(iden_2)
            processed[tar].add(old_opx)
            total_cost += cost2

    return deltas_to_close, composites, total_cost, processed, rolled_vids


def contract_roll(pf, op, vdf, pdf, date, flag, slippage=None):
    """Helper method that deals with contract rolliing if needed.
    
    Args:
        pf (TYPE): the portfolio to which this option belongs.
        op (TYPE): the option being contract-rolled.
        vdf (TYPE): dataframe of strikewise-vols
        pdf (TYPE): dataframe of prices
        date (TYPE): current date in the simulation
        flag (TYPE): OTC or hedge.
        slippage (None, optional): Slippage factor. 
    
    Returns:
        tuple: pf, cost, newop, op, iden 
        > pf = updated portfolio with the new contract option. 
        > cost = cost of rolling over.
        > newop = the new option that was added. 
        > iden = tuple of (product, future month, future price) 
    
    Raises:
        ValueError: Raised if the data for the contract being rolled to
        cannot be found in the dataset provided.
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

    # identifying deltas to close
    iden = (pdt, ftmth, ftprice)

    # TODO: figure out if this affects rolling into serial months. Guessing it does.
    # creating the new options object - getting the tau and vol
    new_vol_id = pdt + '  ' + new_ft_month + '.' + new_ft_month
    lots = op.lots

    r_delta = None
    strike = None
    # case: delta rolling value is specified.
    if d_cond is not None:
        if d_cond == 50:
            strike = 'atm'
        else:
            r_delta = d_cond
    if strike is None:
        strike = op.K

    newop = create_vanilla_option(vdf, pdf, new_vol_id, op.char, op.shorted,
                                  date, lots=lots, strike=strike, delta=r_delta)

    # cost is > 0 if newop.price > op.price
    cost = newop.get_price() - op.get_price()

    pf.remove_security([op], flag)
    pf.add_security([newop], flag)

    pf.refresh()

    # insert function to handle handling of vol_id mappings if they are manually specified. 
    pf = hedging_volid_handler(pf, op.get_vol_id(), new_vol_id)
    engine = pf.get_hedger()

    # check to see if vega slippage is a parameter.
    if engine.has_vol_slippage():
        vol_slippage = engine.get_vol_slippage() 
        if type(vol_slippage) == dict:
            pdt_ticks = vol_slippage[op.get_product()] 
            num_ticks = pdt_ticks[min([x for x in pdt_ticks], key=lambda x: abs(x - op.lots))]
        else:
            num_ticks = vol_slippage
        print('contract_roll - vega slippage: ', num_ticks)
        cost += 2 * num_ticks * abs(newop.get_greek('vega'))

    # handle tick slippage if present. 
    if slippage is not None:
        if type(slippage) == dict:
            pdt_ticks = slippage[op.get_product()] 
            num_ticks = pdt_ticks[min([x for x in pdt_ticks], key=lambda x: abs(x - op.lots))]
        else:
            num_ticks = slippage
        print('contract_roll - num_ticks of slippage: ', num_ticks)
        cost += 2 * num_ticks * op_ticksize[newop.get_product()] * multipliers[newop.get_product()][-1]

    return pf, cost, newop, op, iden


def rebalance(vdf, pdf, pf, buckets=None, brokerage=None, slippage=None,
              next_date=None, settlements=None, book=None):
    """Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.

    Args:
        vdf (pandas dataframe): Dataframe of volatilities
        pdf (pandas dataframe): Dataframe of prices
        pf (object): portfolio object
        buckets (list, optional): buckets to be used when hedging basis expiry.
        brokerage (None, optional): Description
        slippage (None, optional): Description
        next_date (pandas timestamp, optional): next date in the simulation.
        settlements (Dataframe, optional): dataframe of settlement volatilities
        book (bool, optional): if True, rebalancing is done basis book vols. False otherwise.

    Returns:
        tuple: portfolio, cost, boolean indicating if delta-roll occurred.

    """

    droll = None

    # sanity check
    if pf.empty() or pf.ops_empty():
        return pf, 0, False

    roll_hedged = check_roll_status(pf)
    droll = not roll_hedged

    cost = 0

    # first: handle roll-hedging.
    if roll_hedged:
        print('deltas within bounds. skipping roll_hedging')

    if not roll_hedged:
        print('-------- ROLL HEDGING REQUIRED ---------')

        pf, exp = hedge_delta_roll(pf, vdf, pdf, brokerage=brokerage,
                                   slippage=slippage, book=book, settlements=settlements)
        cost += exp
        print('-------- ROLL HEDGING COMPLETED ---------')

    hedge_count = 0

    # get unique date.
    date = vdf.value_date.unique()[0]

    # timestep before rebalancing but after delta rolling
    init_ops = pf.get_all_options()
    num_days = 0 if next_date is None else (
        pd.Timestamp(next_date) - pd.Timestamp(date)).days
    print('timstepping ' + str(num_days) + ' before rebalancing...')
    pf.timestep(num_days * timestep)

    print('pf post timestep pre hedge: ', pf)

    # case: no families (i.e. simple portfolio)
    if not pf.get_families():
        print('simulation.rebalance - simple portfolio hedging. ')
        hedge_engine = pf.get_hedger()
        # initial boolean check
        done_hedging = hedge_engine.satisfied()
        # hedging non-delta greeks.
        while (not done_hedging and hedge_count < 1):
            for flag in pf.hedge_params:
                if flag == 'gamma':
                    fee = hedge_engine.apply('gamma')
                    cost += fee
                    hedge_engine.refresh()
                elif flag == 'vega':
                    fee = hedge_engine.apply('vega')
                    cost += fee
                    hedge_engine.refresh()
                elif flag == 'theta':
                    fee = hedge_engine.apply('theta')
                    cost += fee
                    hedge_engine.refresh()

            # debug statements
            print('overall hedge params: ', pf.hedge_params)

            # hedging delta after non-delta greeks.
            if 'delta' in pf.hedge_params:
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
            print('---------- rebalance loop -------------')

            hedge_engine = dep.get_hedger()
            # initial boolean check
            done_hedging = hedge_engine.satisfied()

            # hedging non-delta greeks.
            while (not done_hedging and hedge_count < 1):
                # insert the actual business of hedging here.
                for flag in dep.hedge_params:
                    if flag == 'gamma':
                        fee = hedge_engine.apply('gamma')
                        cost += fee
                        hedge_engine.refresh()
                    elif flag == 'vega':
                        fee = hedge_engine.apply('vega')
                        cost += fee
                        hedge_engine.refresh()
                    elif flag == 'theta':
                        fee = hedge_engine.apply('theta')
                        cost += fee
                        hedge_engine.refresh()
                hedge_count += 1
                done_hedging = hedge_engine.satisfied()
                print('dep hedges satisfied: ', done_hedging)

        # refresh after hedging individual families
        pf.refresh()

        # debug statements
        print('overall hedge params: ', pf.hedge_params)

        # hedging delta overall after all family-specific hedges have been
        # handled.
        if 'delta' in pf.hedge_params:
            # grabbing condition that indicates zeroing condition on
            # delta
            print('hedging delta')
            ov_hedge = pf.get_hedger()
            fee = ov_hedge.apply('delta')
            cost += fee

    print('updating hedgepoints...')
    pf.hedger.update_hedgepoints()

    # un-timestep OTC options.
    print('reversing timestep...')
    pf.timestep(-num_days * timestep, ops=init_ops)
    # pf.refresh()
    print('hedging completed. ')
    print('pf after hedging: ', pf.net_greeks)
    return (pf,  cost, droll)


def hedge_delta_roll(fpf, vdf, pdf, brokerage=None, slippage=None, book=False, settlements=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta
    exceeds certain bounds.

    Args:
        fpf (portfolio): Description
        vdf (pandas df): volatility data frame containing information for current day
        pdf (pandas df): price dataframe containing information for current day
        brokerage (int, optional): brokerage fees per lot
        slippage (int, optional): slippage loss
        book (bool, optional): Indication as to whether hedging is done basis book vols or settle vols
        settlements (Dataframe, optional): dataframe of settlement vols 

    Returns:
        tuple: updated portfolio and cost of purchasing/selling options.

    Deleted Parameters:
        pf (object): Portfolio being hedged
        roll_cond (list): list of the form ['roll', value, frequency, bound]
    """
    # print('hedge_delta_roll conds: ', roll_cond)

    # initializing dictionary mapping pf -> processed options in that pf.

    if not fpf.get_families():
        print('--- delta rolls: simple portfolio case ---')
        return hedge_delta_roll_simple(fpf, vdf, pdf, brokerage=brokerage,
                                       slippage=slippage, book=book, settlements=settlements)

    else:
        print(' --- delta rolls: composite case ---')
        return hedge_delta_roll_comp(fpf, vdf, pdf, brokerage=brokerage,
                                     slippage=slippage, book=book, settlements=settlements)


def hedge_delta_roll_simple(pf, vdf, pdf, brokerage=None, slippage=None, book=False, settlements=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta exceeds certain bounds.

    Args:
        pf (object): Portfolio being hedged
        vdf (pandas df): volatility data frame containing information for current day
        pdf (pandas df): price dataframe containing information for current day
        brokerage (int, optional): brokerage fees per lot
        slippage (int, optional): slippage loss
        book (bool, optional): True if hedging is being done basis book vols, false otherwise. 
        settlements (None, optional): dataframe of settlement vols. 

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
                                              slippage=slippage, brokerage=brokerage,
                                              book=book, settlements=settlements)
            toberemoved.append(old_op)
            composites.append(newop)
            cost += rcost
            # if rolling option, roll all partners as well.
            for opx in op.partners:
                new_opx, old_opx, rcost = delta_roll(pf, opx, roll_val, vdf, pdf, flag,
                                                     slippage=slippage, brokerage=brokerage,
                                                     book=book, settlements=settlements)
                composites.append(new_opx)
                toberemoved.append(old_opx)
                cost += rcost
        composites = create_composites(composites)
        print('composites: ', [str(x) for x in composites])

    print('number of ops rolled: ', len(toberemoved))

    return pf, cost


def hedge_delta_roll_comp(fpf, vdf, pdf, brokerage=None, slippage=None, book=False, settlements=None):
    """Helper function that handles delta hedge rolling for portfolios consisting
    of multiple families.

    Args:
        fpf (object): full portfolio
        vdf (dataframe): dataframe of vols
        pdf (dataframe): dataframe of prices
        brokerage (None, optional): brokerage value. 
        slippage (None, optional): slippage value. 
        book (bool, optional): True if hedging basis book vols else False
        settlements (dataframe, optional): dataframe of settlement vols

    Returns:
        tuple: portfolio with contracts rolled, and cost.

    Raises:
        ValueError: Raised if something breaks while rolling over option composites from
        two different families.
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
                # print(' --- finished rolling for family ' + str(pf.name) + '
                # ---')
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
                                                  slippage=slippage, brokerage=brokerage,
                                                  book=book, settlements=settlements)
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
                                                         slippage=slippage, brokerage=brokerage,
                                                         book=book, settlements=settlements)
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
        print('processed: ', [str(i) for i in processed[x]])

    fpf.refresh()
    return fpf, cost


def delta_roll(pf, op, roll_val, vdf, pdf, flag, slippage=None,
               brokerage=None, book=False, settlements=None):
    """Helper function that deals with delta-rolling options.

    Args:
        pf (object): portfolio being evaluated
        op (object): option being contract rolled.
        roll_val (float): delta value we are rolling this option to. 
        vdf (dataframe): dataframe of vols
        pdf (dataframe): dataframe of prices
        flag (str): OTC or hedge
        slippage (float, optional): 
        brokerage (float, optional): 
        book (bool, optional): True if hedging is done basis book vols else False
        settlements (dataframe, optional): dataframe of settlement vols. 

    Returns:
        tuple: new option, old option, cost of rolling from old to new. 
    """
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

    if slippage is not None:
        if type(slippage) == dict:
            pdt_ticks = slippage[op.get_product()] 
            num_ticks = pdt_ticks[min([x for x in pdt_ticks], key=lambda x: abs(x - op.lots))]
        else:
            num_ticks = slippage
        print('delta_roll - num_ticks of slippage: ', num_ticks)
        cost += num_ticks * op_ticksize[newop.get_product()] * multipliers[newop.get_product()][-1]

    # case: hedging is being done basis book vols. need to calculate premium difference when
    # considering this option basis book vols and basis settle vols, and add the difference to the
    # cost.
    if book:
        print('volid, book vol: ', op.get_vol_id(), op.vol)

        try:
            cpi = 'C' if op.char == 'call' else 'P'
            df = settlements
            settle_vol = df[(df.vol_id == op.get_vol_id()) &
                            (df.call_put_id == cpi) &
                            (df.strike == op.K)].vol.values[0]
        except IndexError as e:
            print('scripts.simulaton.delta_roll - book vol case: cannot find vol: ',
                  op.get_vol_id(), cpi, op.K)
            settle_vol = op.vol

        print('volid, settle vol: ', op.get_vol_id(), settle_vol)
        true_value = _compute_value(newop.char, newop.tau, settle_vol, newop.K,
                                    newop.underlying.get_price(), 0, 'amer', ki=newop.ki,
                                    ko=newop.ko, barrier=newop.barrier, d=newop.direc,
                                    product=newop.get_product(), bvol=newop.bvol)

        print('op value basis settlements: ', true_value)
        pnl_mult = multipliers[newop.get_product()][-1]
        diff = (true_value - newop.get_price()) * newop.lots * pnl_mult
        print('diff: ', diff)
        cost += diff

    pf.remove_security([op], flag)
    pf.add_security([newop], flag)
    pf.refresh()

    return newop, op, cost


def check_roll_status(pf):
    """Checks to see if delta-roll conditions, if they exist, are satisfied.

    Args:
        pf (portfolio): Portfolio object

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


def write_log(pf, drawdown_limit, date, dailpnl, dailynet, grosspnl, netpnl, dailygamma, 
              gammapnl, dailyvega, vegapnl, roll_hedged, data_order, num_days,
              highest_value, net_cumul_values, breakevens, dailypnl, dailycost, vol_change, price_change):
    """Summary
    
    Args:
        log_pf (TYPE): Description
        pf (TYPE): Description
        drawdown_limit (TYPE): Description
        date (TYPE): Description
        dailpnl (TYPE): Description
        dailynet (TYPE): Description
        grosspnl (TYPE): Description
        netpnl (TYPE): Description
        dailygamma (TYPE): Description
        gammapnl (TYPE): Description
        dailyvega (TYPE): Description
        vegapnl (TYPE): Description
        roll_hedged (TYPE): Description
        data_order (TYPE): Description
        num_days (TYPE): Description
        highest_value (TYPE): Description
        net_cumul_values (TYPE): Description
        breakevens (TYPE): Description
        dailypnl (TYPE): Description
        dailycost (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    loglist = []

    if drawdown_limit is not None:
        drawdown_val = net_cumul_values[-1] - highest_value
        drawdown_pct = (
            highest_value - net_cumul_values[-1])/(drawdown_limit)

    settlement_prices = pf.uid_price_dict()
    # option specific information
    for op in pf.get_all_options():
        op_value = op.get_price()
        # pos = 'long' if not op.shorted else 'short'
        oplots = -op.lots if op.shorted else op.lots
        opvol = op.vol
        strike = op.K
        vol_id = op.get_vol_id()
        tau = round(op.tau * 365)
        
        underlying_id = op.get_uid()
        ftprice = settlement_prices[underlying_id]
        dvol = vol_change[op.get_vol_id()][op.K] if vol_change else 0
        dprice = price_change[op.get_uid()]

        dic = pf.get_aggregated_greeks()
        d, g, t, v = dic[op.get_product()]

        # breakeven = pf.breakeven()[pdt][ftmth]
        # breakevens.append(breakeven)

        # fix logging to take into account BOD to BOD convention.
        lst = [date, vol_id, tau, op_value, oplots, ftprice, strike, opvol, 
               dailypnl, dailynet, grosspnl, netpnl, dailygamma, gammapnl, 
               dailyvega, vegapnl, d, g, t, v, num_days, dvol, dprice]

        cols = ['value_date', 'vol_id', 'ttm', 'option_value', 'option_lottage', 
                'px_settle', 'strike', 'vol', 'eod_pnl_gross', 'eod_pnl_net', 
                'cu_pnl_gross','cu_pnl_net', 'eod_gamma_pnl', 'cu_gamma_pnl', 
                'eod_vega_pnl','cu_vega_pnl', 'net_delta', 'net_gamma', 'net_theta',
                'net_vega', 'timestep', 'vol_change', 'price_change']

        adcols = ['op_delta', 'op_gamma', 'op_theta',
                  'op_vega', 'txn_costs']

        if op.barrier is not None:
            barlevel = op.ki if op.ki is not None else op.ko
            knockedin = op.knockedin
            knockedout = op.knockedout
            bvol_change = vol_change[op.get_vol_id()][barlevel]

            lst.extend([barlevel, bvol_change, knockedin, knockedout])
            cols.extend(['barlevel', 'bvol_change','knockedin', 'knockedout'])
            
        cols.extend(adcols)

        # getting net greeks
        delta, gamma, theta, vega = op.greeks()

        lst.extend([delta, gamma, theta, vega, dailycost])

        l_dic = OrderedDict(zip(cols, lst))
        loglist.append(l_dic)
    
    return loglist

#######################################################################
#######################################################################
#######################################################################
