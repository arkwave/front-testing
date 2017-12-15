# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2017-03-07 21:31:13
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-15 23:24:33

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
from .util import create_underlying, create_vanilla_option, close_out_deltas, create_composites, assign_hedge_objects, compute_market_minus, mark_to_vols
from .prep_data import reorder_ohlc_data, granularize
from .calc import get_barrier_vol, _compute_value
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
                   ohlc=False, remark_on_roll=False, remark_at_end=False, hinge=False):
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
        hinge (bool, optional): flag indicating if hinge analysis is in place (i.e. 1 intrady hedge and 1 settlement hedge. )

    Returns:
        dataframe: dataframe consisting of the logs on each day of the simulation.


    Raises:
        AssertionError: Description


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
    date_range = sorted(voldata.value_date.unique())

    ########################################
    # constructing/assigning hedge objects #
    book = True if mode in ('HBPS', 'HBPB') else False
    pf = assign_hedge_objects(pf, book=book)
    # initial timestep, because previous day's settlements were used to construct
    # the portfolio.
    # get the init diff and timestep
    # init_val = pf.compute_value()
    init_diff = (pd.to_datetime(
        date_range[1]) - pd.to_datetime(date_range[0])).days - 1
    print('init diff: ', init_diff)
    pf.timestep(init_diff * timestep)
    init_val = pf.compute_value()
    print('sim_start BOD init_value: ', init_val)
    # assign book vols; defaults to initialization date.
    book_vols = voldata[voldata.value_date ==
                        pd.to_datetime(date_range[0])]
    print('book vols initialized at ' +
          pd.to_datetime(date_range[0]).strftime('%Y-%m-%d'))
    # reassign the dates since the first datapoint is the init date.
    date_range = date_range[1:]
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

    print('sigvals: ', sigvals)
    # boolean flag indicating missing data
    # Note: [partially depreciated]
    broken = False
    hedges_hit = []
    ##################################################

    ########### identifying simulation mode ###########
    if mode in ('HBPS', 'HBPB'):
        flat_vols = True
    else:
        flat_vols = False
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
        print("==========================================================")

        log_pf = copy.deepcopy(pf)
        dailypnl = 0
        dailygamma = 0
        dailyvega = 0
        # price_changes = {}
        latest_price = {}
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
        # can be granularized with no problems
        settle_vols = vdf_1[vdf_1.datatype == 'settlement'].copy()
        settle_prices = pdf_1[pdf_1.datatype == 'settlement'].copy()

        print('settle_prices: ', settle_prices)

        # filter dataframes to get only pertinent UIDs data.
        pdf_1 = pdf_1[pdf_1.underlying_id.isin(
            pf.get_unique_uids())].reset_index(drop=True)
        vdf_1 = vdf_1[vdf_1.vol_id.isin(
            pf.get_unique_volids())].reset_index(drop=True)
        data_order = None

        print('vdf unique times: ', vdf_1.time.unique())
        # need this check because for intraday/settlement to settlement, no reordering
        # is required; granularize is called in OHLC case only after an order
        # has been determined
        if not ohlc:
            print('@@@@@@@@@@@@@@@@@ Granularizing: Intraday Case @@@@@@@@@@@@@@@@')
            pdf_1 = granularize(pdf_1, pf, intraday=True)
            if pdf_1.empty:
                continue
            print('pdf_1: ', pdf_1)
            try:
                assert len(pdf_1.underlying_id.unique()) == 1
            except AssertionError as e:
                raise AssertionError("dataset not filtered for UIDS on " + str(date) + " : ",
                                     pdf_1.underlying_id.unique()) from e
        print('================ beginning intraday loop =====================')
        unique_ts = pdf_1.time.unique()
        dailyhedges = []
        for ts in unique_ts:
            pdf = pdf_1[pdf_1.time == ts]
            print('ts: ', ts)
            if ohlc:
                print('@@@@@@@@@@@@@@@ OHLC STEP GRANULARIZING @@@@@@@@@@@@@@@@')
                init_pdf, pdf, data_order = reorder_ohlc_data(pdf, pf)
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            # currently, vdf only exists for settlement anyway.
            vdf = vdf_1[vdf_1.time == ts]
            for index in pdf.index:

                # get the current row and variables
                pdf_ts = pdf[pdf.index == index]
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
                print('index: ', index)

                if datatype == 'intraday':
                    dailyhedges.append(
                        {'date': date, 'time': ts, 'uid': uid, 'hedge point': val})

                    print('valid price move to ' + str(val) +
                          ' for uid last hedged at ' + str(lp))
                elif datatype == 'settlement':
                    print('settlement price move to ' + str(val) +
                          ' for uid last hedged at ' + str(lp))

                print('timestep init val: ', init_val)

                # for prices: filter and use exclusively the intraday data. assign
                # to hedger objects.
                print('pf at start: ', pf)
                print('price datatype: ', pdf_ts.datatype.unique())
                print('vol datatype: ', vdf.datatype.unique())

                pf.assign_hedger_dataframes(vdf, pdf_ts)

                print('last price before update: ', latest_price)

            # Step 3: Feed data into the portfolio.

                print("========================= FEED DATA ==========================")
                # NOTE: currently, exercising happens as soon as moneyness is triggered.
                # This should not be much of an issue since exercise is never
                # actually reached.
                pf, broken, gamma_pnl, vega_pnl, exercise_profit, exercise_futures, barrier_futures \
                    = feed_data(vdf, pdf_ts, pf, init_val, flat_vols=flat_vols, flat_price=flat_price)

                print(
                    "========================= PNL & BARR/EX ==========================")

            # Step 4: Compute pnl for the this timestep.
                updated_val = pf.compute_value()
                # sanity check: if the portfolio is closed out during this
                # timestep, pnl = exercise proft.
                if pf.empty():
                    pnl = exercise_profit
                else:
                    pnl = (updated_val - init_val) + exercise_profit

                print('timestamp pnl: ', pnl)
                print('timestamp gamma pnl: ', gamma_pnl)
                print('timestamp vega pnl: ', vega_pnl)

                # update the daily variables.
                dailypnl += pnl
                dailygamma += gamma_pnl
                dailyvega += vega_pnl

                # Detour: add in exercise & barrier futures if required.
                if exercise_futures:
                    print('adding exercise futures')
                    pf.add_security(exercise_futures, 'OTC')

                if barrier_futures:
                    print('adding barrier futures')
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
        print("========================= SIGNALS ==========================")
        if signals is not None:
            print('signals not none, applying. ')
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

        pf, cost, all_deltas = roll_over(pf, roll_vdf, settle_prices, date, brokerage=brokerage,
                                         slippage=slippage)
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
        print('settle prices before rebalance: ', settle_prices)
        pf, cost, roll_hedged = rebalance(h_vdf, settle_prices, pf, brokerage=brokerage,
                                          slippage=slippage, next_date=next_date,
                                          settlements=settle_vols, book=book)
        print('rebalance cost: ', cost)
        dailycost += cost
        print("==================================================================")

    # Step 8: Update highest_value so that the next
    # loop can check for drawdown.

    # store a copy for log writing purposes.

        # compute market minuses, isolate if end of sim.
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
        # Portfolio-wide information
        # dic = log_pf.get_net_greeks()
        call_vega = sum([op.vega for op in log_pf.get_all_options()
                         if op.K >= op.underlying.get_price()])

        put_vega = sum([op.vega for op in log_pf.get_all_options()
                        if op.K < op.underlying.get_price()])

        if drawdown_limit is not None:
            drawdown_val = net_cumul_values[-1] - highest_value
            drawdown_pct = (
                highest_value - net_cumul_values[-1])/(drawdown_limit)

        settlement_prices = pf.uid_price_dict()
        # option specific information
        for op in log_pf.get_all_options():
            op_value = op.get_price()
            # pos = 'long' if not op.shorted else 'short'
            char = op.char
            oplots = -op.lots if op.shorted else op.lots
            opvol = op.vol
            strike = op.K
            pdt, ftmth, opmth = op.get_product(), op.get_month(), op.get_op_month()
            vol_id = op.get_vol_id()
            tau = round(op.tau * 365)
            where = 'OTC' if op in pf.OTC_options else 'hedge'
            underlying_id = op.get_uid()
            ftprice = settlement_prices[underlying_id]

            dic = log_pf.get_net_greeks()
            d, g, t, v = dic[pdt][ftmth]

            breakeven = log_pf.hedger.breakeven[pdt][ftmth]
            breakevens.append(breakeven)

            # fix logging to take into account BOD to BOD convention.
            lst = [date, vol_id, underlying_id, char, where, tau, op_value, oplots,
                   ftprice, strike, opvol, dailypnl, dailynet, grosspnl, netpnl,
                   dailygamma, gammapnl, dailyvega, vegapnl, roll_hedged, d, g, t, v,
                   data_order, num_days, breakeven]

            cols = ['value_date', 'vol_id', 'underlying_id', 'call/put', 'otc/hedge',
                    'ttm', 'option_value', 'option_lottage', 'px_settle',
                    'strike', 'vol', 'eod_pnl_gross', 'eod_pnl_net', 'cu_pnl_gross',
                    'cu_pnl_net', 'eod_gamma_pnl', 'cu_gamma_pnl', 'eod_vega_pnl',
                    'cu_vega_pnl', 'delta_rolled', 'net_delta', 'net_gamma', 'net_theta',
                    'net_vega', 'data order', 'timestep', 'breakeven']

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

    # print('curr_date: ', curr_date)
    date = pd.to_datetime(pdf.value_date.unique()[0])

    # 2)  update prices of futures, underlying & portfolio alike.
    if not broken:
        # price_updated = False
        for ft in pf.get_all_futures():
            pdt = ft.get_product()
            uid = ft.get_uid()
            try:
                # case: flat price.
                if flat_price:
                    continue
                else:
                    val = pdf[(pdf.pdt == pdt) &
                              (pdf.underlying_id == uid)].price.values[0]
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
                              (pdf.underlying_id == uid)].price.values[0]
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

    total_profit = exercise_profit + barrier_profit
    print('total_profit: ', total_profit)

    # refresh portfolio after price updates.
    # if price_updated:
    pf.refresh()
    # removing expiries
    pf.remove_expired()
    # refresh after handling expiries.
    pf.refresh()

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
    print('intermediate value: ', intermediate_val)
    if exercised:
        # case: portfolio is empty after data feed step (i.e. expiries and
        # associated deltas closed)
        if intermediate_val == 0:
            print('gamma pnl calc: portfolio closed out case')
            gamma_pnl = total_profit
        # case: portfolio is NOT empty.
        else:
            print('other case')
            gamma_pnl = (intermediate_val + total_profit) - init_val
    else:
        print('pnl calc: not exercised case')
        gamma_pnl = (intermediate_val + total_profit -
                     init_val) if intermediate_val != 0 else 0
        print('gamma pnl: ', gamma_pnl)

    # skip feeding in vols if 1) data not present or 2) flat_vols flag is
    # triggered.
    volskip = True if (voldf.empty or flat_vols) else False
    if volskip:
        print('Volskip is True')
        print('vdf datatype: ', voldf.datatype.unique())
        print('dataframe empty: ', voldf.empty)
        print('flatvols: ', flat_vols)
    if not volskip:
        # update option attributes by feeding in vol.
        for op in pf.get_all_options():
            # info reqd: strike, order, product, tau
            strike, product, tau = op.K, op.product, op.tau
            b_vol, strike_vol = None, None
            cpi = 'C' if op.char == 'call' else 'P'
            # interpolate or round? currently rounding, interpolation easy.
            ticksize = multipliers[op.get_product()][-2]
            # get strike corresponding to closest available ticksize.
            strike = round(round(strike / ticksize) * ticksize, 2)
            vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()

            try:
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) &
                            (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                df_tau = min(val.tau, key=lambda x: abs(x - tau))
                strike_vol = val[val.tau == df_tau].vol.values[0]
            except (IndexError, ValueError):
                print('### VOLATILITY DATA MISSING ###')
                strike_vol = op.vol

            try:
                if op.barrier is not None:
                    barlevel = op.ki if op.ki is not None else op.ko
                    b_val = voldf[(voldf.pdt == product) & (voldf.strike == barlevel) &
                                  (voldf.vol_id == vid) & (voldf.call_put_id == cpi)]
                    df_tau = min(b_val.tau, key=lambda x: abs(x - tau))
                    b_vol = val[val.tau == df_tau].vol.values[0]
            except (IndexError, ValueError):
                print('### BARRIER VOLATILITY DATA MISSING ###')
                b_vol = op.bvol

            op.update_greeks(vol=strike_vol, bvol=b_vol)

        pf.refresh()

    vega_pnl = pf.compute_value() - intermediate_val if intermediate_val != 0 else 0
    return pf, broken, gamma_pnl, vega_pnl, total_profit, exercise_futures, barrier_futures


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
        if np.isclose(op.tau, tol) or op.tau <= tol:
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
        if np.isclose(op.tau, tol) or op.tau <= tol:
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
        dic = fa.get_all_options()

        # print('dic: ', [str(x) for x in dic])
        print('num_ops: ', len(dic))

        # get list of already processed ops for this family.
        processed_ops = processed[fa]

        # iterate over each option in this family
        for op in dic.copy():
            flag = 'OTC' if op in fa.OTC_options else 'hedge'
            # case: op has already been processed since its parter was
            # processed.
            if op in processed_ops:
                print(str(op) + ' has been processed')
                continue
            composites = []

            # case: roll if ttm threshold is breached or roll_all is triggered.
            print('op.tau: ', op.tau * 365)
            needtoroll = (((round(op.tau * 365) <= fa.ttm_tol) or
                           np.isclose(op.tau, fa.ttm_tol/365)) or
                          roll_all)
            print('needtoroll: ', needtoroll)
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
                    fam, op, vdf, pdf, date, flag)
                # roll the partners
                deltas_to_close, composites, total_cost, processed, rolled_vids\
                    = roll_handle_partners(date, pf, fa, op, deltas_to_close,
                                           composites, total_cost, processed, vdf, pdf, rolled_vids)
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


def roll_handle_partners(date, pf, fa, op, deltas_to_close, composites, total_cost, processed, vdf, pdf, rolled_vids):
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
                tar, opx, vdf, pdf, date, flag2)
            composites.append(new_opx)
            deltas_to_close.add(iden_2)
            processed[tar].add(old_opx)
            total_cost += cost2

    return deltas_to_close, composites, total_cost, processed, rolled_vids


def contract_roll(pf, op, vdf, pdf, date, flag):
    """Helper method that deals with contract rolliing if needed.

    Args:
        pf (TYPE): the portfolio to which this option belongs.
        op (TYPE): the option being contract-rolled.
        vdf (TYPE): dataframe of strikewise-vols
        pdf (TYPE): dataframe of prices
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

    Deleted Parameters:
        hedges (dict): Dictionary of hedging conditions
        counters (TYPE): Description

    """

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
        while (not done_hedging and hedge_count < 3):
            # insert the actual business of hedging here.
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
            while (not done_hedging and hedge_count < 3):
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

    Deleted Parameters:
        roll_cond (list): list of the form ['roll', value, frequency, bound]
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
        fpf (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description
        book (bool, optional): Description
        settlements (None, optional): Description

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
        pf (TYPE): Description
        op (TYPE): Description
        roll_val (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
        flag (TYPE): Description
        slippage (None, optional): Description
        brokerage (None, optional): Description
        book (bool, optional): Description
        settlements (None, optional): Description

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
