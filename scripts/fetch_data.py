# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-17 15:34:51
# @Last Modified by:   arkwave
# @Last Modified time: 2017-07-14 14:18:02

import pandas as pd
from sqlalchemy import create_engine
import time
import os
import numpy as np
from .prep_data import match_to_signals, get_min_start_date, clean_data, vol_by_delta, ciprice

contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'LCC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'V', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V' 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


def pull_relevant_data(pf_path=None, sigpath=None, signals=None, start_date=None, end_date=None, pdt=None, opmth=None, ftmth=None):
    """
    Pulls relevant data based on one of two conditions:
            1. Securities specified in the portfolio, or 
            2. Securities specified in the signal file to be applied. 
        Portfolio takes precedence over signals if both are supplied. Data is pulled according to the following steps:
            1. get all unique underlying ids and vol ids
            2. get all relevant underlying id information from price table
            3. get all relevant settlement vol surfaces from vol table 

    Args:
        pf_path (str, optional): path to a csv specifiying portfolio
        sigpath (str, optional): path to a csv specifying signals to be applied. 
        signals (None, optional): Description
        start_date (None, optional): Description
        end_date (None, optional): Description
        pdt (None, optional): Description
        opmth (None, optional): Description
        ftmth (None, optional): Description

    Returns:
        tuple: price and volatility dataframes. 

    Raises:
        ValueError: Raised if neither pf_path nor sigpath are specified. 
    """
    uids, volids = None, None

    # FIXME: make sure this works for len(pdt) > 1
    if not any(i is None for i in [pdt, opmth, ftmth]):
        print('pulling based on product, opmth and ftmth')
        # NOTE: Currently defaults to pulling ALL contract-month options prior
        # to the current contract month in the same year.
        ft_month = ftmth[0]
        ft_yr = ftmth[1]
        index = contract_mths[pdt].index(ft_month)
        relevant_uids = contract_mths[pdt][:(index + 1)]

        uids = [pdt + '  ' + x + ft_yr for x in relevant_uids]
        volids = [pdt + '  ' + x + ft_yr + '.' +
                  x + ft_yr for x in relevant_uids]

    elif pf_path is not None and not pd.read_csv(pf_path).empty:
        pf = pd.read_csv(pf_path)
        if not pf.empty:
            print('pulling from pf_path')
            # getting unique vol_ids from portfolio
            ops = pf[pf.Type == 'Option']
            fts = set(pf[pf.Type == 'Future'].vol_id)
            op_uids = set(ops.vol_id.str.split('.').str[0])
            uids = list(op_uids.union(fts))
            volids = list(set(ops.vol_id.unique()))
            # get pdt len for future ops

    elif sigpath is not None:
        print('pulling from sigpath')
        df = pd.read_csv(sigpath)
        df['length'] = df.pdt.str.len()
        df['vidspace'] = ''
        df.loc[df.length == 2, 'vidspace'] = 1
        df.loc[df.length == 1, 'vidspace'] = 2
        # handling spaces. irritating.
        placeholder = pd.Series([' '] * len(df.length))
        df['underlying_id'] = df.pdt + placeholder*(2-df.length) + df.ftmth

        df['vol_id'] = df.pdt + placeholder * \
            df.vidspace + df.opmth + '.' + df.ftmth
        uids = df.underlying_id.unique()
        volids = df.vol_id.unique()

    elif signals is not None:
        print('pulling from signals')
        df = signals.copy()
        df['length'] = df.pdt.str.len()
        df['vidspace'] = ''
        df.loc[df.length == 2, 'vidspace'] = 1
        df.loc[df.length == 1, 'vidspace'] = 2
        # handling spaces. irritating.
        placeholder = pd.Series([' '] * len(df.length))
        df['underlying_id'] = df.pdt + placeholder*(2-df.length) + df.ftmth

        df['vol_id'] = df.pdt + placeholder * \
            df.vidspace + df.opmth + '.' + df.ftmth
        uids = df.underlying_id.unique()
        volids = df.vol_id.unique()

    else:
        raise ValueError('no valid inputs; cannot draw data')

    # creating SQL engine, drawing data
    user = input('DB Username: ')
    password = input('DB Password: ')

    engine = create_engine(
        'postgresql://' + user + ':' + password + '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    # construct queries
    price_query, vol_query = construct_queries(uids, volids,
                                               start_date=start_date, end_date=end_date)

    # reading in the query
    pdf = pd.read_sql(price_query, connection, columns=[
        'ticker', 'valuedate', 'px_settle'])

    vdf = pd.read_sql(vol_query, connection)

    connection.close()

    # handling column names etc
    pdf['underlying_id'] = pdf.ticker.str[:2].str.strip() + '  ' + \
        pdf.ticker.str[2:4]
    # print('pdf after draw: ', pdf)

    vdf.volid = vdf.volid.str.split().str[
        0] + '  ' + vdf.volid.str.split().str[1]

    pdf = pdf[['underlying_id', 'valuedate', 'px_settle']]

    pdf.columns = ['underlying_id', 'value_date', 'settle_value']
    vdf.columns = ['value_date', 'vol_id',
                   'strike', 'call_put_id', 'settle_vol']

    # sorting data types
    pdf.value_date = pd.to_datetime(pdf.value_date)
    vdf.value_date = pd.to_datetime(vdf.value_date)

    pdf = pdf.sort_values('value_date')
    vdf = vdf.sort_values('value_date')

    # resetting indices
    pdf.reset_index(drop=True, inplace=True)
    vdf.reset_index(drop=True, inplace=True)

    return pdf, vdf


def construct_queries(uids, volids, start_date=None, end_date=None):
    """Dynamically constructs SQL query to draw relevant information from the database
    based on input flags. 

    Args:
        uids (TYPE): list of unique underlying IDs
        volids (TYPE): list of unique vol_ids
        start_date (None, optional): simulation start date
        end_date (None, optional): simulation end date 

    Returns:
        TYPE: Description
    """
    uids = [" ".join(x.split()) + ' Comdty'for x in uids]
    print('uids: ', uids)
    print('volids: ', volids)

    # constructing baseline queries
    price_query = 'select ticker, valuedate, px_settle from view_future_data where ticker '
    vol_query = 'select value_date, volid, strike, call_put_id, settle_vol from public.table_opera_option_vol_surface where volid '

    # constructing filter statements
    price_filter = '= ' + "'" + \
        uids[0] + "'" if len(uids) == 1 else ' in ' + \
        str(tuple(map(str, uids)))

    vol_filter = 'like ' + \
        str("'" + volids[0] + "'") if len(volids) == 1 else ' in ' + \
        str(tuple(map(str, volids)))

    # updating filter statement based on start/end dates inputted
    start = "".join(start_date.strftime('%Y-%m-%d').split('-')
                    ) if start_date is not None else ''
    end = "".join(end_date.strftime('%Y-%m-%d').split('-')
                  ) if end_date is not None else ''

    if start_date is not None:
        price_filter += ' and valuedate >= ' + "'" + start + "'"
        vol_filter += ' and value_date >= ' + "'" + start + "'"
    if end_date is not None:
        price_filter += ' and valuedate <= ' + "'" + end + "'"
        vol_filter += ' and value_date <= ' + "'" + end + "'"

    price_query = price_query + price_filter
    vol_query = vol_query + vol_filter

    print('price query: ', price_query)
    print('vol query: ', vol_query)

    return price_query, vol_query


def pull_alt_data(pdt):
    """Utility function that draws/cleans data from the alternate data table. 

    Args:
        pdt (string): The product being drawn from the database

    Returns:
        tuple: vol dataframe, price dataframe, raw dataframe. 


    """
    print('starting clock..')
    t = time.clock()

    user = input('DB Username: ')
    password = input('DB Password: ')

    engine = create_engine(
        'postgresql://' + user + ':' + password + '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    query = "select security_id, settlement_date, future_settlement_value, option_expiry_date,implied_vol \
            FROM table_option_settlement_data_all where security_id like '" + pdt.upper() + " %%' and extract(YEAR from settlement_date) > 2009"
    print('query: ', query)
    df = pd.read_sql_query(query, connection)

    df.to_csv('datasets/data_dump/' + pdt.lower() +
              '_raw_data.csv', index=False)

    print('finished pulling data')
    print('elapsed: ', time.clock() - t)

    add = 1 if len(pdt) == 2 else 0

    # cleans up data
    df['vol_id'] = df.security_id.str[:9+add]
    df['call_put_id'] = df.security_id.str[9+add:11+add].str.strip()
    # getting 'S' from 'S Q17.Q17'
    df['pdt'] = df.vol_id.str[0:len(pdt)]
    # df['pdt'] = df.vol_id.str.split().str[0].str.strip()

    # getting opmth
    df['opmth'] = df.vol_id.str.split(
        '.').str[0].str.split().str[1].str.strip()
    df.opmth = df.opmth.str[0] + \
        (df.opmth.str[1:].astype(int) % 10).astype(str)

    df['strike'] = df.security_id.str[10+add:].astype(float)

    df['implied_vol'] = df.implied_vol / 100

    # Q17 -> Q7
    df['ftmth'] = df.vol_id.str.split('.').str[1].str.strip()
    df.ftmth = df.ftmth.str[0] + \
        (df.ftmth.str[1:].astype(int) % 10).astype(str)

    # creating underlying_id and vol_id
    df.vol_id = df.pdt + '  ' + df.opmth + '.' + df.ftmth
    df['underlying_id'] = df.pdt + '  ' + df.ftmth

    # selecting
    vdf = df[['settlement_date', 'vol_id',
              'call_put_id', 'strike', 'implied_vol']]
    pdf = df[['settlement_date', 'underlying_id', 'future_settlement_value']]

    vdf.columns = ['value_date', 'vol_id',
                   'call_put_id', 'strike', 'settle_vol']
    pdf.columns = ['value_date', 'underlying_id', 'settle_value']

    # removing duplicates, resetting indices
    pdf = pdf.drop_duplicates()
    vdf = vdf.drop_duplicates()

    pdf.reset_index(drop=True, inplace=True)
    vdf.reset_index(drop=True, inplace=True)

    if not os.path.isdir('datasets/data_dump'):
        os.mkdir('datasets/data_dump')

    vdf.to_csv('datasets/data_dump/' + pdt.lower() +
               '_vol_dump.csv', index=False)
    pdf.to_csv('datasets/data_dump/' + pdt.lower() +
               '_price_dump.csv', index=False)

    return vdf, pdf, df


def prep_datasets(vdf, pdf, edf, start_date, end_date, pdt, specpath='',
                  signals=None, test=False, write=False, writepath=None, direc='C:/Users/Ananth/Desktop/Modules/HistoricSimulator/'):
    """Utility function that does everything prep_data does, but to full datasets rather than things drawn from the database. Made because i was lazy. 

    Args:
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Datafrane of prices
        edf (dataframe): Dataframe of option expiries 
        start_date (pd Timestamp): start date of the simulation
        end_date (pd Timestamp): end date of the simulation
        specpath (str, optional): path to a portfolio specs csv
        signals (None, optional): path to a signals dataframe
        test (bool, optional): flag that indicates if dataframes should be written.
        write (bool, optional): Descriptio

    Returns:
        Tuple: vol, price, expiry, cleaned_price and start date. 

    Raises:
        ValueError: Description
    """
    # edf = pd.read_csv(epath).dropna()

    # sanity checking
    print('start_date: ', start_date)
    print('end_date: ', end_date)

    # fixing datetimes
    vdf.value_date = pd.to_datetime(vdf.value_date)
    pdf.value_date = pd.to_datetime(pdf.value_date)

    vdf = vdf.sort_values('value_date')
    pdf = pdf.sort_values('value_date')

    # filtering relevant dates
    vdf = vdf[(vdf.value_date >= start_date) &
              (vdf.value_date <= end_date)]

    pdf = pdf[(pdf.value_date >= start_date) &
              (pdf.value_date <= end_date)]

    assert not vdf.empty
    assert not pdf.empty

    vid_list = vdf.vol_id.unique()
    p_list = pdf.underlying_id.unique()

    # print('util.prep_datasets - vid_list: ', vid_list)
    # print('util.prep_datasets - p_list: ', p_list)

    if os.path.exists(specpath):
        specs = pd.read_csv(specpath)
        vid_list = specs[specs.Type == 'Option'].vol_id.unique()

    # case 1: drawing based on portfolio.
    if signals is not None:
        signals.value_date = pd.to_datetime(signals.value_date)
        vdf, pdf = match_to_signals(vdf, pdf, signals)

    # print('vid list: ', vid_list)

    # get effective start date, pick whichever is max

    # case 2: drawing based on pdt, ft and opmth
    dataset_start_date = get_min_start_date(
        vdf, pdf, vid_list, signals=signals)
    print('datasets start date: ', dataset_start_date)

    # dataset_start_date = pd.to_datetime(dataset_start_date)

    # start_date = dataset_start_date if (start_date is None) or \
    #     ((start_date is not None) and (dataset_start_date > start_date)) else start_date

    print('prep_data start_date: ', start_date)

    # catch errors
    if (vdf.empty or pdf.empty):
        raise ValueError(
            '[scripts/prep_data.read_data] : Improper start date entered; resultant dataframes are empty')
    # print('pdf: ', pdf)
    # print('vdf: ', vdf)
    # clean dataframes
    edf = clean_data(edf, 'exp')
    vdf = clean_data(vdf, 'vol', date=start_date,
                     edf=edf)
    pdf = clean_data(pdf, 'price', date=start_date,
                     edf=edf)

    # reassigning variables
    final_vol = vdf
    final_price = pdf

    # final preprocessing steps
    final_price = ciprice(pdf)
    # print('final price: ', final_price)
    # final_vol = civols(vdf, final_price)

    print('sanity checking date ranges')
    if not np.array_equal(pd.to_datetime(final_vol.value_date.unique()),
                          pd.to_datetime(final_price.value_date.unique())):
        vmask = final_vol.value_date.isin([x for x in final_vol.value_date.unique()
                                           if x not in final_price.value_date.unique()])
        pmask = final_price.value_date.isin([x for x in final_price.value_date.unique()
                                             if x not in final_vol.value_date.unique()])
        final_vol = final_vol[~vmask]
        final_price = final_price[~pmask]

    if not test:
        vbd = vol_by_delta(final_vol, final_price)

        # merging vol_by_delta and price dataframes on product, underlying_id,
        # value_date and order
        vbd.underlying_id = vbd.underlying_id.str.split().str[0]\
            + '  ' + vbd.underlying_id.str.split().str[1]
        final_price.underlying_id = final_price.underlying_id.str.split().str[0]\
            + '  ' + final_price.underlying_id.str.split().str[1]
        merged = pd.merge(vbd, final_price, on=[
                          'pdt', 'value_date', 'underlying_id'])
        final_price = merged

        # handle conventions for vol_id in price/vol data.
        final_vol.vol_id = final_vol.vol_id.str.split().str[0]\
            + '  ' + final_vol.vol_id.str.split().str[1]
        final_price.vol_id = final_price.vol_id.str.split().str[0]\
            + '  ' + final_price.vol_id.str.split().str[1]

    if write:
        desired_path = writepath if writepath is not None else direc + 'datasets/debug/'
        if not os.path.isdir(desired_path):
            os.mkdir(desired_path)

        # write datasets into the debug folder.
        final_vol.to_csv(desired_path + pdt.lower() +
                         '_final_vols.csv', index=False)

        final_price.to_csv(desired_path + pdt.lower() +
                           '_final_price.csv', index=False)

        pdf.to_csv(desired_path + pdt.lower() +
                   '_roll_df.csv', index=False)

        edf.to_csv(desired_path + 'final_option_expiry.csv', index=False)

    return final_vol, final_price, edf, pdf, start_date


def grab_data(pdts, start_date, end_date, ftmth=None, opmth=None, sigpath=None,
              writepath=None, direc='C:/Users/Ananth/Desktop/Modules/HistoricSimulator/',
              write=True, test=False, volids=None):
    """Utility function that allows the user to easily grab a dataset by specifying just the product,
    start_date and end_date. Used to small datasets for the purposes of testing new functions/modules.
DO NOT USE to generate datasets to be passed into simulation.py; use
pull_alt_data + prepare_datasets for that.

    Args:
        pdt (TYPE): the product being evaluated.
        opmth (TYPE): option month
        ftmth (TYPE): future month
        start_date (TYPE): start date of the dataset desired.
        end_date (TYPE): end date of the dataset desired.

    return:
        pandas dataframe: the data particular to that commodity between start and end dates.
    """
    print('### RUNNING GRAB_DATA ###')
    print('start_date: ', start_date)
    print('end_date: ', end_date)

    final_pdf = pd.DataFrame()
    final_vols = pd.DataFrame()

    pdts = set(pdts)

    for pdt in pdts:
        volpath = direc + 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'

        price_path = direc + 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'

        print('volpath: ', volpath)
        print('pricepath: ', price_path)

        # handling signals
        signals = pd.read_csv(sigpath) if sigpath is not None else None
        if signals is not None:
            signals.value_date = pd.to_datetime(signals.value_date)

        # handling prices and vos
        if not os.path.exists(volpath) or not os.path.exists(price_path):
            print('dumps dont exist, pulling raw data')
            vdf, pdf, raw_df = pull_alt_data(pdt)
        else:
            print('dumps exist, reading in')
            vdf = pd.read_csv(volpath)
            pdf = pd.read_csv(price_path)

        # handling datetime formats.
        edf = pd.read_csv(direc + 'datasets/option_expiry.csv')
        vdf.value_date = pd.to_datetime(vdf.value_date)
        pdf.value_date = pd.to_datetime(pdf.value_date)
        edf.expiry_date = pd.to_datetime(edf.expiry_date)

        # filter according to start/end dates
        vdf = vdf[(vdf.value_date >= start_date) &
                  (vdf.value_date <= end_date)]

        pdf = pdf[(pdf.value_date >= start_date) &
                  (pdf.value_date <= end_date)]

        print('pdf columns: ', pdf.columns)
        print('vdf columns: ', vdf.columns)

        if volids is not None:
            relevant_volids = [x for x in volids if x[:2].strip() == pdt]
            uids = [x.split()[0] + '  ' + x.split('.')[1]
                    for x in relevant_volids]
            print('relevant volids: ', relevant_volids)
            print('relevant uids: ', uids)
            pdf = pdf[pdf.underlying_id.isin(uids)]
            vdf = vdf[vdf.vol_id.isin(relevant_volids)]

        # try filtering just by uid and vol_id
        if ftmth is not None and opmth is not None:
            u_id = pdt + '  ' + ftmth
            print('uid: ', u_id)
            vol_id = pdt + '  ' + opmth + '.' + ftmth
            print('vid: ', vol_id)

            vdf = vdf[(vdf.vol_id == vol_id)]
            pdf = pdf[(pdf.underlying_id == u_id)]

        vdf, pdf, edf, roll_df, start_date = prep_datasets(vdf, pdf, edf, start_date,
                                                           end_date, pdt, signals=signals,
                                                           test=False, write=write, writepath=writepath,
                                                           direc=direc)
        final_pdf = pd.concat([final_pdf, pdf])
        final_vols = pd.concat([final_vols, vdf])

    print('### GRAB DATA COMPLETED ###')
    return vdf, pdf, edf
