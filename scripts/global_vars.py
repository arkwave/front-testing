# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-12 13:32:49
# @Last Modified by:   Ananth
# @Last Modified time: 2017-05-18 20:17:52
import pandas as pd


################ CHANGE THESE #################
pdt = 'S'
size = 'small'
signals = False
start_date = pd.Timestamp('2016-05-01')
end_date = pd.Timestamp('2016-07-30')
brokerage = 0
slippage = None  # [0.05, 0.10, 0.15]
###############################################


######### automate path selection based on inputs ##########
folder = 'datasets/' + size + '_' + pdt.lower() + '/'

# final datasets
final_vol_path = folder + 'final_vols.csv'
final_price_path = folder + 'final_price.csv'
final_exp_path = folder + 'final_expdata.csv'


# cleaned datasets (penultimate processing step)
cleaned_vol_path = folder + 'cleaned_vol.csv'
cleaned_price_path = folder + 'cleaned_price.csv'
cleaned_exp_path = folder + 'cleaned_exp.csv'


# raw datasets
# raw_vol_path = folder + 'raw_vols.csv'
# raw_price_path = folder + 'raw_prices.csv'
raw_exp_path = 'datasets/option_expiry.csv'


# hedge path
hedge_path = 'hedging.csv'

# signal path
signal_path = folder + 'signals.csv' if signals else None


# portfolio location
portfolio_path = 'datasets/corn_portfolio_specs.csv'
# portfolio_path = 'specs.csv'

#########################################################

# path composites for easy checking
final_paths = [final_vol_path, final_price_path,
               final_exp_path, cleaned_price_path, signal_path]

# raw_paths = [raw_vol_path, raw_price_path, raw_exp_path]


# separate variables for testing-related files.
test_start_date = pd.Timestamp('2017-01-15')
test_vol_data = 'datasets/small_c/c_vols.csv'
test_price_data = 'datasets/small_c/c_prices.cs v'
test_exp_data = 'datasets/option_expiry.csv'
