# defines paths to all relevant datasets
from pandas import Timestamp
import os

# full datasets - processed
final_vol_path = 'datasets/full_data/filtered_2_vols.csv'
final_price_path = 'datasets/full_data/filtered_2_prices.csv'
final_exp_path = 'datasets/full_data/final_expdata.csv'

# full datasets - raw
raw_vol_path = 'datasets/full_data/corn_full_vols.csv'
raw_price_path = 'datasets/full_data/corn_full_prices.csv'
raw_exp_path = 'datasets/option_expiry.csv'


# full datasets - cleaned
cleaned_vol = 'datasets/full_data/cleaned_vol.csv'
cleaned_price = 'datasets/full_data/cleaned_price.csv'
cleaned_exp = 'datasets/full_data/cleaned_exp.csv'


# small datasets - raw
small_vol_path = 'datasets/small_data/corn_vols.csv'
small_price_path = 'datasets/small_data/corn_prices.csv'


# small datasets - processed
small_final_vol_path = 'datasets/small_data/final_vols.csv'
small_final_price_path = 'datasets/small_data/final_price.csv'
small_final_exp_path = 'datasets/small_data/final_expdata.csv'


# small datasets - cleaned
small_cleaned_vol = 'datasets/small_data/cleaned_vol.csv'
small_cleaned_price = 'datasets/small_data/cleaned_price.csv'
small_cleaned_exp = 'datasets/small_data/cleaned_exp.csv'


# hedge path
hedge_path = 'hedging.csv'

# signal path
signal_path = 'datasets/signals.csv'

# portfolio location
portfolio_path = 'datasets/corn_portfolio_specs.csv'


# simulation settings
start_date = Timestamp('2014-08-07')
# start_date = Timestamp('2017-01-01')
# end_date =

brokerage = 0
slippage = None


# separate variables for testing-related files.
test_start_date = Timestamp('2017-01-15')
test_vol_data = 'datasets/small_data/corn_vols.csv'
test_price_data = 'datasets/small_data/corn_prices.csv'
test_exp_data = 'datasets/option_expiry.csv'
