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

"""
TODO:
> step 0     : conceptualize every step. 
> step 1     : feed_data
> Step 2     : delta/gamma/vega hedging
> Step 3     : handle_options
> Step 4     : pnl accumulation
> Step 5     : rebalancing - delta hedging.

"""

# TODO: Figure out how to pass in lots efficiently.

# list of hedging conditions.
hedges = {'delta': 'zero', 'gamma': (-5000, 5000), 'vega': (-5000, 5000)}

# slippage/brokerage
slippage = 1
lots = 10

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


# TODO: Finish implementation.
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
    raw_diff = 0
    # 1) initial value of the portfolio before updates.
    prev_val = pf.compute_value()
    # decrement tau
    pf.timestep(timestep)

    # 2) Check for rollovers and expiries
    # TODO

    # 3)  update prices of futures, underlying & portfolio alike.
    for future in pf.get_all_futures:
        name = future.get_name()
        # TODO: Figure out specifics of names after knowing dataset.
        pricename = name + '_' + 'price'
        val = pdf[pricename]       # placeholder
        future.update_price(val)

    # update option attributes by feeding in vol.
    all_options = pf.get_securities()[0]
    for option in all_options:
        name = option.get_product()
        volname = name + '_' + 'vol'
        volvalue = vdf[volname]    # placeholder
        option.update_greeks(volvalue)

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
    1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints      assumed. 
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
    dic = pf.get_net_greeks()
    for product in dic:
        for month in dic[product]:
            net = dic[product][month]
            cost, pf = hedge_gamma_vega(hedges, vdf, pdf, net, month, pf)
            expenditure += cost
            cost, pf = hedge_delta(hedges['delta'], vdf, pdf, net, month, pf)
            expenditure += cost
    return expenditure, pf


# TODO: update this with new objects in mind.
def hedge_gamma_vega(hedges, vdf, pdf, greeks, month, pf):
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
    # naming variables for clarity.
    delta = greeks[0]
    gamma = greeks[1]
    theta = greeks[2]
    vega = greeks[3]

    gamma_bound = hedges['gamma']
    delta_cond = hedges['delta']
    vega_bound = hedges['vega']

    # relevant data for constructing Option and Future objects.
    price = data[price_name]       # placeholder
    vol = data[vol_name]           # placeholder
    tau = calc_tau()               # placeholder.
    product = get_product          # placeholder
    underlying = Future(month, price, product)

    # check to see if gamma and vega are being hedged.
    if 'gamma' and 'vega' in hedges:
        # creating straddle components.
        callop = Option(price, tau, 'call', vol, underlying, 'euro')
        putop = Option(price, tau, 'put', vol, underlying, 'euro')
        straddle_val = callop.get_value() + putop.get_value()

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
            flag = 'long'
            num_required = ceil((lower-gamma)/(pgamma + cgamma))
        elif gamma > upper:
            # need to short straddles. expenditure is negative.
            flag = 'short'
            num_required = ceil((upper-gamma)/(pgamma + cgamma))
        expenditure += num_required * straddle_val
        for i in range(num_required):
            pf.add_security(callop, flag)
            pf.add_security(putop, flag)

    if vega not in range(*vega_bound):
        lower = vega_bound[0]
        upper = vega_bound[1]
        # gamma hedging logic.
        if vega < lower:
            # need to buy straddles. expenditure is positive.
            flag = 'long'
            num_required = ceil((lower-vega)/(pvega + cvega))
        elif vega > upper:
            # need to short straddles. expenditure is negative.
            flag = 'short'
            num_required = ceil((upper-vega)/(pvega + cvega))
        expenditure += num_required * straddle_val
        for i in range(num_required):
            pf.add_security(callop, flag)
            pf.add_security(putop, flag)

    return expenditure, pf


# TODO: delta hedging
def hedge_delta(cond, vdf, pdf, net, month, pf):
    """Helper function that implements delta hedging. General idea is to zero out delta at the end of the day by buying/selling -delta * lots futures. 

    Args:
        cond (string): condition for delta hedging
        vdf (dataframe): Dataframe of volatilities
        pdf (dataframe): Dataframe of prices 
        net (list): greeks associated with net_greeks[product][month]
        month (str): month of underlying future. 
        pf (portfolio): portfolio object specified by portfolio_specs.txt 

    Returns:
        TYPE: Description

    Deleted Parameters:
        data (TYPE): Description
        greeks (TYPE): Description
    """
    future_price = data[price]  # placeholder
    expenditure = 0
    if cond == 'zero':
        # flag that indicates delta hedging.
        for product in greeks:
            for month in greeks[product]:
                vals = greeks[product][month]
                delta = vals[0]
                flag = 'short' if delta > 0 else 'long'
                # long delta; need to short futures.
                # TODO: math.ceil isn't the right thing. figure out how
                # lots play into price.
                num_lots_needed = delta * 100
                num_futures = ceil(num_lots_needed / lots)
                for i in range(len(num_futures)):
                    pf.add_security(ft, flag)
                    if flag == 'short':
                        expenditure -= future_price
                    else:
                        expenditure += future_price
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
