""" Overall script that runs the simulation """

from scripts import classes as cs
from scripts import calc as clc
from scripts import prep_data as dat
import pandas as pd
import numpy as np

"""
TODO:
> Step 1 : feed_data
> Step 2 : feed_data
> Step 3 : handle_options
> Step 4 : handle_futures
> Step 5 : pnl accumulation
> Step 6 : rebalancing.
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
	4) Handle the futures component:
		> record overall change in value of the futures.
	5) PnL calculation. Components include:
		> PnL Contribution from Options
		> PnL contribution from Futures.
	6) Rebalance the Greeks
		> buy/sell options to hedge gamma/vega according to conditions
		> buy/sell futures to zero delta (if required)
	Process then repeats from step 1 for the next input.

	Inputs:
	1) df            :
	2) pf            :
	3) gamma_cond    :
	4) vega_cond     :
	5) delta_cond    :

	Outputs:
	1) Graph of daily PnL
	2) Graph of cumulative PnL
	3) Various summary statistics.


	"""

    # Step 1 & 2
    for i in list(df.Index):
        # getting data pertinent to that day.
        data = df.iloc[[i]]
        # raw_change to be the difference between old and new value per iteration.
        raw_change, pf = feed_data(data, pf)

    # Step 3
    	pf = handle_options(pf)

    # Step 4
    	pf = handle_futures(pf)

    # Step 5


    # Step 6


    # Step 7



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

	"""
	raw_diff = 0
	prev_val = pf.compute_value()
	# decrement tau
	# feed in new values of price to :
	# 	1) futures in portfolio (i.e. pf.futures)
	#	2) futures that are underlying (i.e. pf.options.get_future())
	#   result: updates prices of all futures, including underlying.
	# feed in new values of vols to:
	#	1) options in portfolio (i.e. pf.options)
	# 	result: updates self.vol, greeks and value for each option.
	# compute value again, store difference.
	# return difference and new portfolio.


	return raw_diff, pf
def handle_options(pf):
	"""
	Inputs: 
		1) pf  : an instance of a Portfolio object. 
	
	Outputs:
		1) 
	"""



def handle_futures(pf):
	"""
	Inputs: 
		1) pf  : an instance of a Portfolio object. 
	
	Outputs:
		1) 
	"""
	pass


if __name__ == '__main__':
    pf =  # TODO: devise clean way to import a portfolio.
    df = dat.prep_data()
    run_simulation(df, pf)
