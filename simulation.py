"""
File Name      : simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 11/4/2017
Python version : 3.5
Description    : Overall script that runs the simulation

"""

import numpy as np
import pandas as pd
from scripts.classes import Option, Future
from scripts.prep_data import read_data, prep_portfolio, get_rollover_dates
from math import ceil
import copy
import time

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
hedges = {'delta': 'zero', 'gamma': (-5000, 5000), 'vega': (-5000, 5000)}

# slippage/brokerage
# slippage = 1
# brokerage = 1

# passage of time
timestep = 1/365


def run_simulation(voldata, pricedata, expdata, pf, hedges=hedges):
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

    date_range = sorted(voldata.value_date.unique())  # [1:]
    # print('date range: ', date_range)
    # Step 1 & 2
    init_val = 0
    # next_date = None
    for i in range(len(date_range)):
        date = date_range[i]
        try:
            next_date = date_range[i+1]
        except IndexError:
            next_date = None
        # init_val = pf.compute_value()
        # pf.timestep(timestep)
        # isolate data relevant for this day.
        print('##################### date: ', date, '################')
        # init_val = pf.compute_value()
        print('INITIAL VALUE: ', init_val)
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]
        # getting data pertinent to that day.
        # raw_change to be the difference between old and new value per
        # iteration.
        # print(str(date) + ' feeding data [1/3]')
        raw_change, pf, broken = feed_data(vdf, pdf, pf, rollover_dates)
        # pnl += raw_change
        if broken:
            break
    # Step 3
        # print(str(date) + ' handling exercise [2/3]')
        expenditure, pf = handle_exercise(pf)
        pnl += expenditure

        # compute value after updating greeks
        updated_val = pf.compute_value()
        dailypnl = updated_val - init_val if init_val != 0 else 0
        pnl += dailypnl
        print('[10]   EOD PNL: ', dailypnl)
        print('[10.5] Cumulative PNL: ', pnl)
    # Step 4
        # print(str(date) + ' rebalancing [3/3')
        pf = rebalance(vdf, pdf, pf, hedges)
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
    # print('Portfolio: ', pf)
    return pnl, pf


def feed_data(voldf, pdf, pf, dic):
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

    # debugging
    x = list(pf.OTC['C']['N7'][0])
    # print('HEDGES: ', pf.hedges)
    y = list(pf.hedges['C']['N7'][0]) if pf.hedges else []

    broken = False
    if voldf.empty:
        raise ValueError('vol df is empty!')
    date = voldf.value_date.unique()[0]
    raw_diff = 0
    # 1) initial value of the portfolio before updates.
    prev_val = pf.compute_value()
    # print('[0]    PREVIOUS VALUE: ', prev_val)
    # print('[0.1]  PRICE BEFORE VOL UPDATE: ', x[0].get_price())
    # print('[0.2]  VOL BEFORE VOL UPDATE: ', x[0].vol)
    # print('[0.3]  GREEKS BEFORE VOL UPDATE: ', pf.OTC['C']['N7'][2:])
    # print('[0.4]  PORFOLIO BEFORE VOL UPDATE: ', pf)
    # 2) Check for rollovers and expiries
    # rollovers
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
    priceList = []

    for ft in pf.get_all_futures():

        pdt, ordering = ft.get_product(), ft.get_ordering()
        # print(pf.get_all_futures()) if ordering is None else print(
        # 'ordering not none')
        # print('ordering: ', ordering)
        try:
            val = pdf[(pdf.pdt == pdt) & (
                pdf.order == ordering)].settle_value.values[0]
            ft.update_price(val)
            # print('UPDATED - new price: ', val)
        # index error would occur only if data is missing.
        except IndexError:
            # print('Date, ordering, product', pdt,
                  # ordering, pdf.value_date.unique())
            print('###### DATA MISSING #######')
            # print(pdt, ordering)
            broken = True
            break

    # update option attributes by feeding in vol.
    all_options = pf.get_all_options()
    if not broken:
        for op in all_options:
            # info reqd: strike, order, product.
            strike, order, product = op.K, op.ordering, op.product
            cpi = 'C' if op.char == 'call' else 'P'
            # interpolate or round? currently rounding, interpolation easy.
            strike = round(strike/10) * 10
            try:
                val = voldf[(voldf.pdt == product) & (voldf.strike == strike) & (
                    voldf.order == order) & (voldf.call_put_id == cpi)].settle_vol.values[0]
                op.update_greeks(vol=val)
                # print('UPDATED - new vol: ', val)
            except IndexError:
                print('### DATA MISSING ###')
                broken = True
                break

    # updating portfolio after modifying underlying objects
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)

    # print('[1]  TAU: ', x[0].tau)
    # print('[2]  TTM: ', x[0].tau * 365)
    # print('[3]  VOL AFTER UPDATE: ', x[0].vol)
    # print('[4]  PRICE AFTER UPADTE: ', x[0].compute_price())
    # print('[5]  GREEKS AFTER UPDATE: ', pf.OTC['C']['N7'][2:])

    # if y:
    #     print('[5.1] GREEKS HEDGE: ', y[0].greeks())
    #     print('[5.2] GREEKS HEDGE: ', y[1].greeks())

    # print('[6]  PORFOLIO AFTER UPDATE: ', pf)
    # print('[7]  NET GREEKS: ', pf.net_greeks)

    # 5) computing new value
    new_val = pf.compute_value()
    raw_diff = new_val - prev_val
    # print('[8]  NEW VALUE AFTER FEED: ', new_val)
    return raw_diff, pf, broken


def handle_exercise(pf):
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
    all_ops = pf.get_all_options()

    for op in all_ops:

        if op.tau <= tol and op.exercise():
            print("----- EXERCISING CASE ------")
            print(op.tau, op.exercise())
            op.update_tau(op.tau)
            # once for exercise, another for selling/buying to cover the
            # future obtained.
            product = op.get_product()
            pnl_mult = multipliers[product][-1]
            # fees = 2*brokerage if not op.shorted else 0
            expenditure += op.lots * op.get_price()*pnl_mult  # - fees

    return expenditure, pf


def rebalance(vdf, pdf, pf, hedges):
    """ Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.
    Notes:
    1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed.
    2) hedging delta done by shorting/buying -delta * lots futures.
    3)

    Args:
        vdf (TYPE): Dataframe of volatilities
        pdf (TYPE): Dataframe of prices
        pf (TYPE): portfolio object
        hedges (TYPE): Dictionary of hedging conditions 

    Returns:
        TYPE: Description
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    # hedging delta, gamma, vega.
    dic = copy.deepcopy(pf.get_net_greeks())
    for product in dic:
        for month in dic[product]:
            ordering = pf.compute_ordering(product, month)
            ginputs = gen_hedge_inputs(
                hedges, vdf, pdf, month, pf, product, ordering, 'gamma')
            vinputs = gen_hedge_inputs(
                hedges, vdf, pdf, month, pf, product, ordering, 'vega')
            pf = hedge(pf, ginputs, product, month, 'gamma')
            pf = hedge(pf, vinputs, product, month, 'vega')
            pf, dhedges = hedge_delta(hedges['delta'], vdf, pdf,
                                      pf, month, product, ordering)
    return pf


# TODO: update this with new objects in mind.
def gen_hedge_inputs(hedges, vdf, pdf, month, pf, product, ordering, flag):
    """Helper function that generates the inputs required to construct atm
    straddles for hedging, based on the flag.

    Args:
        hedges (TYPE): hedging rules.
        vdf (TYPE): volatility dataframe
        pdf (TYPE): price dataframe
        month (TYPE): month being hedged
        pf (TYPE): portfolio being hedged
        product (TYPE): product being hedged
        ordering (TYPE): ordering corresponding to month being hedged
        flag (TYPE): gamma or vega

    Returns:
        list : inputs required to construct atm straddles.
    """
    net_greeks = pf.get_net_greeks()
    greeks = net_greeks[product][month]
    # naming variables for clarity.
    gamma = greeks[1]
    vega = greeks[3]
    greek = gamma if flag == 'gamma' else vega
    gamma_bound = hedges['gamma']
    vega_bound = hedges['vega']
    bound = gamma_bound if flag == 'gamma' else vega_bound

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
        pf (TYPE): portfolio object
        inputs (TYPE): list of inputs reqd to construct straddle objects
        product (TYPE): the product being hedged
        month (TYPE): month being hedged

    Returns:
        tuple: cost of the hedge, and the updated portfolio
    """

    price, k, cvol, pvol, tau, underlying, greek, bound, ordering = inputs

    # creating straddle components.
    callop = Option(k, tau, 'call', cvol, underlying,
                    'euro', month=month, ordering=ordering, shorted=None)

    # print('[9]  CVOL: ', cvol)
    # print('[10] PVOL: ', pvol)

    putop = Option(k, tau, 'put', pvol, underlying,
                   'euro', month=month, ordering=ordering, shorted=None)

    # print('delta diff ' + str(flag), cd + pd)

    # straddle_val = callop.compute_price() + putop.compute_price()
    lm, dm = multipliers[product][1], multipliers[product][0]

    if flag == 'gamma':
        greek_c = (list(callop.greeks())[1] * dm) / (callop.lots * lm)
        greek_p = (list(putop.greeks())[1] * dm) / (putop.lots * lm)

    else:
        greek_c = list(callop.greeks())[3] * 100 / (callop.lots * lm * dm)
        greek_p = list(putop.greeks())[3] * 100 / (putop.lots * lm * dm)

    # gamma and vega hedging.
    cdelta, cgamma, ctheta, cvega = callop.greeks()
    pdelta, pgamma, ptheta, pvega = putop.greeks()

    if flag == 'gamma':
        pgreek, cgreek = pgamma, cgamma
        # print('gamma hedging straddle: ', pgamma, cgamma)
    else:
        pgreek, cgreek = pvega, cvega

    # checking if gamma exceeds bounds
    upper = bound[1]
    lower = bound[0]
    if greek > upper or greek < lower:
        # gamma hedging logic.
        # print(flag + ' hedging')
        if greek < lower:
            # print('lower!')
            # need to buy straddles. expenditure is positive.
            callop.shorted = False
            putop.shorted = False
            # print('greek_c: ', greek_c)
            # print('greek_p: ', greek_p)
            if flag == 'gamma':
                lots_req = round((abs(greek) * dm)/((greek_c + greek_p) * lm))
            elif flag == 'vega':
                lots_req = round((abs(greek) * 100) /
                                 ((greek_c + greek_p) * lm * dm))
            callop.lots, putop.lots = lots_req, lots_req
            # print('lots req: ', lots_req)
            num_required = ceil(abs(greek)/(pgreek + cgreek))
            # print('num_req: ', num_required)

        elif greek > upper:
            # need to short straddles. expenditure is negative.
            # print('upper')
            callop.shorted = True
            putop.shorted = True
            if flag == 'gamma':
                lots_req = round((greek * dm)/((greek_c + greek_p) * lm))
            elif flag == 'vega':
                # print('vega')
                lots_req = round((greek * 100)/((greek_c + greek_p) * lm * dm))
            # print('lots req: ', lots_req)
            callop.lots, putop.lots = lots_req, lots_req
            num_required = ceil((greek)/(pgreek + cgreek))
            # print('num_required: ', num_required)

        callops = [callop] * num_required
        putops = [putop] * num_required
        cd, cg, ct, cv = callop.greeks()
        # print('CALLOP GREEKS: ', cd, cg, ct, cv)
        pd, pg, pt, pv = putop.greeks()
        # print('PUTOP GREEKS: ', pd, pg, pt, pv)
        # print('DEBUG - adding')
        pf.add_security(callops, 'hedge')
        pf.add_security(putops, 'hedge')

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

    future_price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    net_greeks = pf.get_net_greeks()
    # curr_delta_hedged = 0
    # print('[11]  NG DH: ', net_greeks)
    if cond == 'zero':
        # flag that indicates delta hedging.
        vals = net_greeks[product][month]
        delta = vals[0]
        # check if hedges already exist for this product/month
        # if (product in pf.hedges) and (month in pf.hedges[product]):
        #     hedge_futures = pf.hedges[product][month][1]
        #     curr_delta_hedged = sum([x.lots for x in hedge_futures])
        #     num_lots_needed = abs(round(delta)) - curr_delta_hedged
        #     shorted = True if num_lots_needed > 0 else False
        #     num_lots_needed = abs(num_lots_needed)
        # print('[12]  DELTA: ', delta)
        shorted = True if delta > 0 else False
        num_lots_needed = abs(round(delta))
        if num_lots_needed == 0:
            print('delta is already zeroed!')
            return pf, None
        else:
            ft = Future(month, future_price, product,
                        shorted=shorted, ordering=ordering, lots=num_lots_needed)
            pf.add_security([ft], 'hedge')
    return pf, ft


if __name__ == '__main__':
    filepath = 'portfolio_specs.txt'
    vdf, pdf, edf = read_data(filepath)
    # check sanity of data
    vdates = pd.to_datetime(vdf.value_date.unique())
    # print(vdates)

    pdates = pd.to_datetime(pdf.value_date.unique())
    # print(pdates)
    if not np.array_equal(vdates, pdates):
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range.')

    # generate portfolio
    pf = prep_portfolio(vdf, pdf, filepath)
    # print(pf)
    # proceed to run simulation
    run_simulation(vdf, pdf, edf, pf)
