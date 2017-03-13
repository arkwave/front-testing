""" Overall script that runs the simulation """

from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
from scripts import calc as clc
from scripts import prep_data as dat
from math import ceil, floor
import pandas as pd
import numpy as np

"""
TODO:
> Step 3     : handle_options
> Step 4     : pnl accumulation
> Step 5     : rebalancing - delta hedging.

"""

# TODO: Product specific information.

# {product : [futures_multiplier - (dollar_mult, lot_mult),
# options_multiplier - (dollar_mult, lot_mult), futures_tick,
# options_tick, brokerage]}

# list of hedging conditions.
hedges = {'delta': 'zero', 'gamma': (-5000, 5000), 'vega': (-5000, 5000)}

# slippage/brokerage
slippage = 1


def run_simulation(df, pf, hedges):
    """Each run of the simulation consists of 6 steps:

    1) Feed data into the portfolio.
    2) Compute:
            > change in greeks from price and vol update
            > change in overall value of portfolio from price and vol update.
    3) Handle the options component:
            > Check if option is bullet or daily.
            > Check for expiry/exercise. Expiry can be due to barriers or tau = 0. Record changes to:
                    - futures bought/sold as the result of exercise. [PnL]
                    - changes in monthly greeks from options expiring. [PnL]
                    - total number of securities in the portfolio; remove expired options.
    4) PnL calculation. Components include:
            > PnL contribution from changes in price/vols.
            > PnL Contribution from Options
            > PnL from shorting straddles (gamma/vega hedging)

    5) Rebalance the Greeks
            > buy/sell options to hedge gamma/vega according to conditions
            > buy/sell futures to zero delta (if required)

    Process then repeats from step 1 for the next input.

    Inputs:
    1) df             : dataframe containing price and vol series for all futures.
    2) pf             : Portfolio object.
    3) gamma_bound    : gamma limits
    4) vega_bound     : vega limits
    5) delta_cond     : delta hedging strategy

    Outputs:
    1) Graph of daily PnL
    2) Graph of cumulative PnL
    3) Various summary statistics.

    """
    pnl = 0
    # Step 1 & 2
    for i in list(df.Index):
        # getting data pertinent to that day.
        data = df.iloc[[i]]
        # raw_change to be the difference between old and new value per
        # iteration.
        raw_change, pf = feed_data(data, pf)
        pnl += raw_change
    # Step 3
        cost, pf = handle_options(pf)
        pnl += cost
    # Step 4
        cost, pf = rebalance(data, pf, hedges)
        pnl += cost
    # Step 6: Plotting results/data viz


def feed_data(data, pf):
    """
    This function does the following:
            0) Store old value of the portfolio.
            1) given a one-row dataframe, feed the relevant entries into each security within the portfolio, where the string associated with sec.underlying() is the same as title of column in dataframe.
            2) update the value of the portfolio according to the info fed in. 
    Inputs: 
            1) data : the data being fed into the portfolio.
            2) pf   : an object of type Portfolio. Refer to scripts\classes.py for class documentation.
    Outputs:
            1) raw_diff: the change in the portfolio's value solely due to new price/vols.
            2) pf      : the updated portfolio object.
    """
    raw_diff = 0
    # initial value of the portfolio before updates.
    prev_val = pf.compute_value()
    time_passed = 1/365
    # decrement tau
    pf.timestep(time_passed)
    # update prices of futures, underlying & portfolio alike.
    for future in pf.get_all_futures:
        name = future.get_name()
        # TODO: Figure out specifics of names after knowing dataset.
        pricename = name + '_' + 'price'
        val = df[pricename]
        future.update_price(val)

    # update option attributes by feeding in vol.
    all_options = pf.get_securities()[0]
    for option in all_options:
        name = option.get_product()
        volname = name + '_' + 'vol'
        volvalue = df[volname]
        option.update_greeks(vol)

    # computing new value
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


def rebalance(data, pf, hedges):
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    expenditure = 0
    # hedging delta, gamma, vega.
    dic = pf.net_greeks()
    for month in dic:
        net = dic[month]
        cost, pf = hedge_gamma_vega(hedges, data, net, month)
        expenditure += cost
        cost, pf = hedge_delta(hedges['delta'], data, net, month)
        expenditure += cost
    return expenditure, pf


# TODO: delta hedging
def hedge_gamma_vega(hedges, data, greeks, month):
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
    price = data[price_name]    # placeholder
    vol = data[vol_name]        # placeholder
    tau = calc_tau()            # placeholder.
    product = get_product       # placeholder
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


def hedge_delta(cond, data, greeks, month):
    pass


if __name__ == '__main__':
    pf = dat.read_portfolio()
    df = dat.prep_data()
    run_simulation(df, pf)
