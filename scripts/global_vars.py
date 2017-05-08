# defines paths to all relevant datasets
from pandas import Timestamp
import os


# change these accordingly.
pdt = 'CT'
size = 'small'
folder = None

############ folders by product ##############
# corn
corn_folder_full = 'datasets/full_c/'
corn_folder_small = 'datasets/small_c/'
# cotton
ct_folder_small = 'datasets/small_ct/'
ct_folder_full = 'datasets/full_ct/'
# bean oil
bo_folder_small = 'datasets/small_bo/'
bo_folder_full = 'datasets/full_bo/'
##############################################

# assigning folder based on pdt and size flags
if pdt == 'C':
    folder = corn_folder_full if size == 'full' else corn_folder_small
elif pdt == 'CT':
    folder = ct_folder_full if size == 'full' else ct_folder_small
elif pdt == 'BO':
    folder = bo_folder_full if size == 'full' else bo_folder_small


# full datasets - processed
final_vol_path = folder + 'filtered_2_vols.csv'
final_price_path = folder + 'filtered_2_prices.csv'
final_exp_path = folder + 'final_expdata.csv'

# full datasets - raw
raw_vol_path = folder + 'corn_full_vols.csv'
raw_price_path = folder + 'corn_full_prices.csv'
raw_exp_path = 'datasets/option_expiry.csv'


# full datasets - cleaned
cleaned_vol = folder + 'cleaned_vol.csv'
cleaned_price = folder + 'cleaned_price.csv'
cleaned_exp = folder + 'cleaned_exp.csv'


# small datasets - raw
small_vol_path = folder + pdt.lower() + '_vols.csv'
small_price_path = folder + pdt.lower() + '_prices.csv'


# small datasets - processed
small_final_vol_path = folder + 'final_vols.csv'
small_final_price_path = folder + 'final_price.csv'
small_final_exp_path = folder + 'final_expdata.csv'


# small datasets - cleaned
small_cleaned_vol = folder + 'cleaned_vol.csv'
small_cleaned_price = folder + 'cleaned_price.csv'
small_cleaned_exp = folder + 'cleaned_exp.csv'


# hedge path
hedge_path = 'hedging.csv'

# signal path
signal_path = folder + 'signals.csv'

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
