# PortfolioSimulator
General-purpose module that simulates the behavior of a portfolio subject to a variety of constraints and strategies.

## Overview of this Module:
1. This module allows arbitrarily-designated portfolios to be run through historical price/vol series of variable length. 


## Running the simulation:

### Required inputs 
1. The simulation requires 4 base data files to run:
	> Daily strikewise volatility surface for options being run.
	> Daily underlying price series for all options being run. 
	> DataFrame of all option expiries. 
	> Dataframe of portfolio specifications
	> Dataframe of signals (if applicable) 

2. Relative path (i.e. path from working directory) to each of the three files can be specified in scripts/global_vars.py. This file currently automates the filepaths which are searched according to some simple logic, so following this logic is recommended. This logic also dictates where the files prepared during the simulation run will be saved/stored. 

3. An end date can be stipulated if desired. Otherwise, end date is set to be the largest date common to the vol data, price data, and signals (if applicable). 

## Requirements:
1. Python > 3.x
2. Packages: Pandas, Numpy, Scipy, matplotlib/seaborn for visualization. Using the Anaconda distribution recommended.
