# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-06-30 19:46:51
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-03 21:28:33

################### Imports ###################
import pandas as pd
###############################################


############################## Main Function ################################

def generate_skew_signals(pdf, vdf, init_pos=0, strategy=None, strat_str=None):
    """
    #########################################
    Main function that generates the signals based on the dataframe and the dictionary of strategies passed in. This function does the following things:
        1) initializes the position to be init_pos (defaults to 0)
        2) filters skew data for that given day. 
        3) selects the appropriate strategy to apply, given:
            > 1. the current position 
            > 2. the direction of the move. 
            > 3. the current level of the variable of interest. 
           Strategy applies a change to the portfolio based on logic. 
        4) repeat from step 1. 


    Args:
        dataframe (TYPE): Description
        init_pos (int, optional): Description
        strategy (None, optional): Description
        #########################################

            Notes: 

            > attempting to keep this as general as possible so that signals generated on the basis of machine learning methods can be easily applied. 

            > These signals could be categorical or numeric, depending on the kind of method used to generate the signal. 

            > general formulation:
            - dataframe has a target column
            - strat_dic is multi-indexed according to:
                 1) direction and 
                 2) magnitude of change of value of target_column. 
            - function mapped to by strat_dic should accept 1) current_input and 2) previous input. use these two to:
                1) derive direction
                2) derive current 'bucket' or bracket. 
                3) apply the appropriate strategy. 

    Deleted Parameters:
        strat_dic (TYPE): dictionary containing references to strategy-applying functions on the basis of: 1) direction and 2) current position. 


    Returns:
        TYPE: Dataframe of signals 
    """
    # dictionary mapping vol_id to prev value of metric.
    prev_vals = {}
    # dictionary mapping vol_id to prev position held on that vol_id
    pos_dic = {}
    # assigning important variables.
    pdf.value_date = pd.to_datetime(pdf.value_date)
    vdf.value_date = pd.to_datetime(vdf.value_date)

    dates = pd.to_datetime(vdf.value_date.unique())

    # dictionary of the form date : {vol_id: {callvol, putvol, atmvol, signal}}
    final_dic = {}

    # main loop
    for date in dates:
        final_dic[date] = {}
        relevant_prices = pdf[pdf.value_date == date]
        relevant_vols = vdf[vdf.value_date == date]
        vids = relevant_prices.vol_id.unique()
        # filter each unique vol_id per given day.
        for vid in vids:
            # preallocate a dictionary for each vid pertaining to each date.
            if vid not in final_dic[date]:
                final_dic[date][(vid, 'call')] = {}
                final_dic[date][(vid, 'put')] = {}

            # grab the current position of this vol_id, add to dictionary if it
            # is not present. TODO: reconcile this with rolling over.
            if vid not in pos_dic:
                pos_dic[vid] = 0

            # at this point, should technically be filtered down to one row,
            # i.e. one particular vol_id at on one particular day.
            v_prices = relevant_prices[relevant_prices.vol_id == vid]
            v_vols = relevant_vols[relevant_vols.vol_id == vid]

            prev_vals, pos_dic, sig_values = strategy(v_prices, v_vols,
                                                      prev_vals, pos_dic, vid)

            # sig values should be a list of lists, with each list containing:
            # strike, call/put, lots, greek, greekval, signal, product.

            # print(vid_daily_data.columns)

            # # grab the previous percentile value, if it exists.
            # pval = prev_vals[vid] if vid in prev_vals else 0
            # curr_val = vid_daily_data.pct.values[0]
            # prev_vals[vid] = curr_val

            # # set current position of this vid
            # curr_pos = pos_dic[vid]

            # # assigning relevant columns to final_dic.
            # final_dic[date][vid]['pct'] = curr_val
            # final_dic[date][vid]['skew'] = vid_daily_data['skew'].values[0]
            # final_dic[date][vid]['call_vol'] = \
            #     vid_daily_data.call_vol.values[0]

            # final_dic[date][vid]['put_vol'] = \
            #     vid_daily_data.put_vol.values[0]

            # if 'atm_vol' in vid_daily_data.columns:
            #     final_dic[date][vid]['atm_vol'] = \
            #         vid_daily_data.atm_vol.values[0]

            # TODO: can try to replace this with a more general formulation.
            magnitude = curr_val - pval
            # direction = 'down' if magnitude < 0 else 'up'
            # select function based on direction in strat_dic.

            sig = strategy(curr_val, curr_pos, magnitude=magnitude)
            final_dic[date][vid]['signal'] = sig

    final_df = process_dic(final_dic, 'band_strat_simple')

    return final_df


def process_dic(dic, strat_str):
    """Processes the dictionary returne d by computation steps of _strategy_ into a dataframe. 

    Args:
        dic (TYPE): dictionary to convert into a dataframe.
        strat_str (TYPE): descriptor of the strategy used, will determine the method in which dataframe is processed. 

    Returns:
        TYPE: Description
    """
    if strat_str == 'band_strat_simple':
        # unstacking the first two keys in nested dict.
        t1 = pd.DataFrame.from_records(dic).stack().reset_index()
        t1.columns = ['vol_id', 'value_date', 'dic']
        # unstacking the rest.
        t2 = pd.DataFrame.from_records(list(t1.dic.values))
        # concatenating end results, dropping useless columns.
        final = pd.concat([t1, t2], axis=1)
        final = final.drop('dic', axis=1)
        final['vega'] = 10000
        final['pdt'] = final.vol_id.str.split().str[0]
        final['opmth'] = final.vol_id.str.split().str[1].str[:2]
        final['ftmth'] = final.vol_id.str.split('.').str[1]

    return final
