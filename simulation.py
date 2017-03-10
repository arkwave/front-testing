""" Overall script that runs the simulation """

from scripts import classes as cs
from scripts import calc as clc
from scripts import prep_data as dat
import pandas as pd
import numpy as np

"""
TODO:
> Step 1 & 2 : feed_data
> Step 3     : handle_options
> Step 5     : pnl accumulation
> Step 6     : rebalancing.

"""


def run_simulation(df, pf, gamma_cond, vega_cond, delta_cond):
    """Each run of the simulation consists of 6 steps:
    1) Feed data into the portfolio.
    2) Compute:
            > change in greeks from price and vol update
            > change in overall value of portfolio from price and vol update.
    3) Handle the options component:
            > Check if option is bullet or daily. [PnL]
            > Check for expiry/exercise. Expiry can be due to barriers or tau = 0. Record changes to:
                    - futures bought/sold as the result of exercise. [PnL]
                    - changes in monthly greeks from options expiring. [PnL]
                    - total number of securities in the portfolio; remove expired options.
    4) PnL calculation. Components include:
            > PnL Contribution from Options
            > PnL contribution from Futures.
    5) Rebalance the Greeks
            > buy/sell options to hedge gamma/vega according to conditions
            > buy/sell futures to zero delta (if required)
    Process then repeats from step 1 for the next input.

    Inputs:
    1) df            : dataframe containing price series for all futures (portfolio and underlying), and vol series for all futures.
    2) pf            : Portfolio object.
    3) gamma_cond    : gamma limits
    4) vega_cond     : vega limits
    5) delta_cond    : delta hedging strategy

    Outputs:
    1) Graph of daily PnL
    2) Graph of cumulative PnL
    3) Various summary statistics.

    """
    # Step 1 & 2
    for i in list(df.Index):
        # getting data pertinent to that day.
        data = df.iloc[[i]]
        # raw_change to be the difference between old and new value per
        # iteration.
        raw_change, pf = feed_data(data, pf)
    # Step 3
        pf = handle_options(pf)
    # Step 4
        pf = handle_futures(pf)
    # Step 5
        pf = rebalance(data, pf, delta_cond, gamma_cond, vega_cond)

    # Step 6: Plotting results/data viz


def feed_data(data, pf):
    """This function should:
            0) Store old value of the portfolio.
            1) given a one-row dataframe, feed the relevant entries into each security within
            the portfolio. Hope is that string associated with sec.underlying is the same as title of column in dataframe.
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
        name = option.get_underlying.get_name()
        volname = name + '_' + 'vol'
        volvalue = df[volname]
        option.update_greeks(vol)

    # computing new value
    new_val = pf.compute_value()
    raw_diff = new_val - prev_val

    return raw_diff, pf

    return raw_diff, pf


def handle_options(pf):
    """
    Inputs: 
            1) pf  : an instance of a Portfolio object. 

    Outputs:
            1) 
    """
    pass


def rebalance(pf, delta_cond, gamma_cond, vega_cond):
    # delta hedging mandated
    if delta_cond == 'zero':
        # purchase delta * underlying price
        pass
    pass


if __name__ == '__main__':
    pf = None  # TODO: devise clean way to import a portfolio.
    df = dat.prep_data()
    run_simulation(df, pf)
