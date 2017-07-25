# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-12 13:32:49
# @Last Modified by:   Ananth
# @Last Modified time: 2017-07-13 16:20:19
import pandas as pd
import os


################ CHANGE THESE #################
pdt = 'W'
signals = True
start_date = pd.Timestamp('2016-05-23')
end_date = pd.Timestamp('2016-07-01')
brokerage = 0
slippage = None  # [0.05, 0.10, 0.15]
write_path = 'results/' + pdt.lower() + '/logs/'
###############################################


######### automate path selection based on inputs ##########
main_direc = 'anant_000'
# main_direc = 'Ananth'

folder = 'C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/datasets/data_dump/'

if not os.path.isdir(folder):
    os.mkdir(folder)

# data dumps that are filtered for price/vol data
vol_dump = folder + pdt.lower() + '_vol_dump.csv'
price_dump = folder + pdt.lower() + '_price_dump.csv'


# # final datasets
final_vol_path = 'datasets/debug/' + pdt.lower() + '_final_vols.csv'

# final_expiries = 'datasets/debug/' + pdt.lower() + '_option_expiry.csv'

# raw datasets
raw_exp_path = 'datasets/option_expiry.csv'


# hedge path
hedge_path = 'hedging.csv'

# signal path
# signal_path = folder + 'signals.csv' if signals else None
signal_path = '../skew_lib/results/test_signal.csv' if signals else None


# portfolio location
# portfolio_path = 'datasets/corn_portfolio_specs.csv'
portfolio_path = 'specs.csv'

#########################################################

# path composites for easy checking
final_paths = [vol_dump, price_dump, raw_exp_path, signal_path]


# separate variables for testing-related files.
test_start_date = pd.Timestamp('2017-01-15')
test_vol_data = 'datasets/small_c/c_vols.csv'
test_price_data = 'datasets/small_c/c_prices.cs v'
test_exp_data = 'datasets/option_expiry.csv'
