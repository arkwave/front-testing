"""
File Name      : simulation.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 3/4/2017
Python version : 3.5
Description    : Overall script that runs the simulation

"""

import numpy as np
import pandas as pd
from scripts.classes import Option, Future
from scripts.prep_data import read_data, prep_portfolio, get_rollover_dates
from math import ceil
import copy

"""
TODO:
> Step 3     : handle_options
> Step 4     : pnl accumulation
"""

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
slippage = 1
brokerage = 1

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
    rollover_dates = get_rollover_dates(pricedata)
    pnl = 0
    start = min(min(voldata.value_date), min(pricedata.value_date))
    end = min(max(voldata.value_date), max(pricedata.value_date))
    date_range = pd.bdate_range(start, end)
    # Step 1 & 2
    for date in date_range:
        # isolate data relevant for this day.
        vdf = voldata[voldata.value_date == date]
        pdf = pricedata[pricedata.value_date == date]
        # getting data pertinent to that day.
        # raw_change to be the difference between old and new value per
        # iteration.
        raw_change, pf = feed_data(vdf, pdf, pf, rollover_dates)
        pnl += raw_change
    # Step 3
        cost, pf = handle_options(pf)
        pnl += cost
    # Step 4
        cost, pf = rebalance(vdf, pdf, pf, hedges)
        pnl += cost
    # Step 6: Plotting results/data viz


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
    date = voldf.value_date.unique()[0]
    raw_diff = 0
    # 1) initial value of the portfolio before updates.
    prev_val = pf.compute_value()
    # decrement tau
    pf.timestep(timestep)
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
    for ft in pf.get_all_futures():
        pdt, ordering = ft.get_product(), ft.get_ordering()
        val = pdf[(pdf.pdt == pdt) & (
            pdf.order == ordering)].settle_val.values[0]
        ft.update_price(val)

    # update option attributes by feeding in vol.
    all_options = pf.get_all_options()
    for op in all_options:
        # info reqd: strike, order, product.
        strike, order, product = op.K, op.ordering, op.product
        cpi = 'C' if op.char == 'call' else 'P'
        # interpolate or round? currently rounding, interpolation easy.
        strike = round(strike/10) * 10
        val = voldf[(voldf.pdt == product) & (voldf.strike == strike) & (
            voldf.order == order) & (voldf.call_put_id == cpi)]
        op.update_greeks(val)

    # updating portfolio after modifying underlying objects
    pf.update_sec_by_month(None, 'OTC', update=True)
    pf.update_sec_by_month(None, 'hedge', update=True)

    # 5) computing new value
    new_val = pf.compute_value()
    raw_diff = new_val - prev_val

    return raw_diff, pf


def handle_options(pf):
    """
    Inputs:
    1) pf      : portfolio object.

    Outputs: None.
    """
    pass


def rebalance(vdf, pdf, pf, hedges):
    """ Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega. 
    Notes:
    1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed. 
    2) hedging delta done by shorting/buying -delta * lots futures. 
    3) 

    Args:
        vdf (TYPE): Description
        pdf (TYPE): Description
        pf (TYPE): Description
        hedges (TYPE): Description

    Returns:
        TYPE: Description
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    expenditure = 0
    # hedging delta, gamma, vega.
    dic = copy.deepcopy(pf.get_net_greeks())
    for product in dic:
        for month in dic[product]:
            ordering = pf.compute_ordering(product, month)
            cost, pf = hedge_gamma_vega(
                hedges, vdf, pdf, month, pf, product, ordering)
            expenditure += cost
            cost, pf = hedge_delta(hedges['delta'], vdf, pdf,
                                   month, pf, product, ordering)
            expenditure += cost

    return expenditure, pf


# TODO: update this with new objects in mind.
def hedge_gamma_vega(hedges, vdf, pdf, month, pf, product, ordering):
    """Helper function that deals with vega/gamma hedging   

    Args:
        hedges (dictionary): dictionary that contains hedging requirements for the different greeks
        vdf (pandas dataframe): dataframe of volatilities
        pdf (pandas dataframe): dataframe of prices
        greeks (list): list of greeks corresponding to this month and product. 
        month (string): the month of the option's underlying. 
        pf (portfolio): portfolio specified by portfolio_specs.txt 

    Returns:
        tuple: expenditure on this hedge, and updated portfolio object. 

    """
    expenditure = 0
    net_greeks = pf.get_net_greeks()
    greeks = net_greeks[product][month]
    # naming variables for clarity.
    gamma = greeks[1]
    vega = greeks[3]

    gamma_bound = hedges['gamma']
    vega_bound = hedges['vega']

    # relevant data for constructing Option and Future objects.
    price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    straddle_strike = round(price/10) * 10
    cvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'C') & (vdf.order == ordering) & (vdf.strike == straddle_strike)].settle_vol.values[0]
    pvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == straddle_strike)].settle_vol.values[0]
    tau = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == straddle_strike)].tau.values[0]
    underlying = Future(month, price, product)

    # check to see if gamma and vega are being hedged.
    if 'gamma' and 'vega' in hedges:
        # creating straddle components.
        callop = Option(price, tau, 'call', cvol, underlying, 'euro')
        putop = Option(price, tau, 'put', pvol, underlying, 'euro')
        straddle_val = callop.compute_price() + putop.compute_price()

    # gamma and vega hedging.
    cdelta, cgamma, ctheta, cvega = callop.greeks()
    pdelta, pgamma, ptheta, pvega = putop.greeks()

    # checking if gamma exceeds bounds
    if gamma not in range(*gamma_bound):
        lower = gamma_bound[0]
        upper = gamma_bound[1]
        # gamma hedging logic.
        if gamma < lower:
            # need to buy straddles. expenditure is positive.
            callop.shorted = False
            putop.shorted = False
            num_required = ceil((lower-gamma)/(pgamma + cgamma))
        elif gamma > upper:
            # need to short straddles. expenditure is negative.
            callop.shorted = True
            putop.shorted = True
            num_required = ceil((upper-gamma)/(pgamma + cgamma))
        expenditure += num_required * straddle_val
        for i in range(num_required):
            pf.add_security(callop, 'hedge')
            pf.add_security(putop, 'hedge')

    if vega not in range(*vega_bound):
        lower = vega_bound[0]
        upper = vega_bound[1]
        # gamma hedging logic.
        if vega < lower:
            # need to buy straddles. expenditure is positive.
            # flag = 'long'
            callop.shorted = False
            putop.shorted = False
            num_required = ceil((lower-vega)/(pvega + cvega))
        elif vega > upper:
            # need to short straddles. expenditure is negative.
            num_required = ceil((upper-vega)/(pvega + cvega))
            callop.shorted = True
            putop.shorted = True

        expenditure += num_required * straddle_val
        for i in range(num_required):
            pf.add_security(callop, 'hedge')
            pf.add_security(putop, 'hedge')

    return expenditure, pf


# TODO: delta hedging
def hedge_delta(cond, vdf, pdf, month, pf, product, ordering):
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
    expenditure = 0
    net_greeks = pf.get_net_greeks()
    if cond == 'zero':
        # flag that indicates delta hedging.
        for product in net_greeks:
            for month in net_greeks[product]:
                vals = net_greeks[product][month]
                delta = vals[0]
                num_lots_needed = delta * 100
                num_futures = ceil(num_lots_needed / lots)
                shorted = True if delta > 0 else False
                ft = Future(month, future_price, product,
                            shorted=shorted, ordering=ordering)
                for i in range(num_futures):
                    pf.add_security(ft, 'hedge')
                expenditure = (expenditure - num_futures*future_price) if shorted else (
                    expenditure + num_futures*future_price)
    return expenditure, pf


if __name__ == '__main__':
    filepath = 'portfolio_specs.txt'
    vdf, pdf, edf = read_data(filepath)
    # check sanity of data
    vdates = pd.to_datetime(vdf.value_date.unique())
    pdates = pd.to_datetime(pdf.value_date.unique())
    if not np.array_equal(vdates, pdates):
        raise ValueError(
            'Invalid data sets passed in; vol and price data must have the same date range.')
    # generate portfolio
    pf = prep_portfolio(vdf, pdf, filepath)
    # proceed to run simulation
    run_simulation(vdf, pdf, edf, pf)
