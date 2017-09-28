# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-17 15:34:51
# @Last Modified by:   Ananth
# @Last Modified time: 2017-09-28 19:18:56

# import time
import pandas as pd
from sqlalchemy import create_engine
import time
import os
import numpy as np
from .prep_data import match_to_signals, get_min_start_date, clean_data, vol_by_delta, sanity_check, handle_intraday_conventions
from .global_vars import main_direc

contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'QC': ['H', 'K', 'N', 'U', 'Z'],
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


def pull_alt_data(pdt, start_date=None, end_date=None, write_dump=False,
                  direc='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/'):
    """Utility function that draws/cleans data from the alternate data table.

    Args:
        pdt (string): The product being drawn from the database
        start_date (None, optional): Description
        end_date (None, optional): Description
        write_dump (bool, optional): Description

    Returns:
        tuple: vol dataframe, price dataframe, raw dataframe.


    """

    assert (start_date is not None or end_date is not None) or write_dump

    print('starting clock..')
    t = time.clock()

    user = 'sumit'
    password = 'Olam1234'

    # user = input('DB Username: ')
    # password = input('DB Password: ')

    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')

    engine = create_engine('postgresql://' + user + ':' + password +
                           '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    query = "select security_id, settlement_date, future_settlement_value, option_expiry_date,implied_vol \
            FROM table_option_settlement_data_all where security_id like '" + pdt.upper() + " %%' "

    if start_date is not None:
        query += "and settlement_date >= " + "'" + \
            ''.join(start_date.split('-')) + "'"

    if end_date is not None:
        query += " and settlement_date <= " + \
            "'" + ''.join(end_date.split('-')) + "'"

    if start_date is None and end_date is None and write_dump:
        query += "and extract(YEAR from settlement_date) > 2009"
    print('query: ', query)
    df = pd.read_sql_query(query, connection)

    # df.to_csv('datasets/data_dump/' + pdt.lower() +
    #           '_raw_data.csv', index=False)

    print('finished pulling data')
    print('elapsed: ', time.clock() - t)

    add = 1 if len(pdt) == 2 else 0

    # cleans up data
    df['vol_id'] = df.security_id.str[:9 + add]
    df['call_put_id'] = df.security_id.str[9 + add:11 + add].str.strip()
    # getting 'S' from 'S Q17.Q17'
    df['pdt'] = df.vol_id.str[0:len(pdt)]
    # df['pdt'] = df.vol_id.str.split().str[0].str.strip()

    # getting opmth
    df['opmth'] = df.vol_id.str.split(
        '.').str[0].str.split().str[1].str.strip()
    df.opmth = df.opmth.str[0] + \
        (df.opmth.str[1:].astype(int) % 10).astype(str)

    df['strike'] = df.security_id.str[10 + add:].astype(float)

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
                   'call_put_id', 'strike', 'vol']
    pdf.columns = ['value_date', 'underlying_id', 'price']

    # removing duplicates, resetting indices
    pdf = pdf.drop_duplicates()
    vdf = vdf.drop_duplicates()

    pdf.reset_index(drop=True, inplace=True)
    vdf.reset_index(drop=True, inplace=True)

    if write_dump:
        if not os.path.isdir(direc + 'datasets/data_dump'):
            os.mkdir(direc + 'datasets/data_dump')
        vdf.to_csv(direc + 'datasets/data_dump/' + pdt.lower() +
                   '_vol_dump.csv', index=False)
        pdf.to_csv(direc + 'datasets/data_dump/' + pdt.lower() +
                   '_price_dump.csv', index=False)

    return vdf, pdf, df


def prep_datasets(vdf, pdf, edf, start_date, end_date, pdt, specpath='',
                  signals=None, test=False, write=False, writepath=None,
                  direc='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/'):
    """Utility function that does everything prep_data does, but to full 
        datasets rather than things drawn from the database.

    Args:
        vdf (dataframe): Dataframe of vols
        pdf (dataframe): Datafrane of prices
        edf (dataframe): Dataframe of option expiries
        start_date (pd Timestamp): start date of the simulation
        end_date (pd Timestamp): end date of the simulation
        pdt (TYPE): Description
        specpath (str, optional): path to a portfolio specs csv
        signals (None, optional): path to a signals dataframe
        test (bool, optional): flag that indicates if dataframes should be written.
        write (bool, optional): Descriptio
        writepath (None, optional): Description
        direc (TYPE, optional): Description

    Returns:
        Tuple: vol, price, expiry, cleaned_price and start date.

    Raises:
        ValueError: Description
    """
    # edf = pd.read_csv(epath).dropna()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # sanity checking
    vdf.value_date = pd.to_datetime(vdf.value_date)
    pdf.value_date = pd.to_datetime(pdf.value_date)

    vdf = vdf.sort_values('value_date')
    pdf = pdf.sort_values('value_date')

    assert not vdf.empty
    assert not pdf.empty

    vid_list = vdf.vol_id.unique()

    if os.path.exists(specpath):
        specs = pd.read_csv(specpath)
        vid_list = specs[specs.Type == 'Option'].vol_id.unique()

    # case 1: drawing based on portfolio.
    if signals is not None:
        signals.value_date = pd.to_datetime(signals.value_date)
        vdf, pdf = match_to_signals(vdf, pdf, signals)

    # get effective start date, pick whichever is max

    # case 2: drawing based on pdt, ft and opmth
    dataset_start_date = get_min_start_date(
        vdf, pdf, vid_list, signals=signals)
    print('datasets start date: ', dataset_start_date)

    print('prep_data start_date: ', start_date)

    # catch errors
    if (vdf.empty or pdf.empty):
        raise ValueError(
            '[scripts/prep_data.read_data] : ' +
            'Improper start date entered; resultant dataframes are empty')

    print('vdf.columns: ', vdf.columns)
    print('pdf.columns: ', pdf.columns)

    vdf.columns = ['value_date', 'vol_id', 'call_put_id', 'strike', 'vol']
    pdf.columns = ['value_date', 'underlying_id', 'price']

    # clean dataframes
    edf = clean_data(edf, 'exp')
    vdf = clean_data(vdf, 'vol', date=start_date,
                     edf=edf)
    pdf = clean_data(pdf, 'price', date=start_date,
                     edf=edf)

    # reassigning variables
    final_vol = vdf
    final_price = pdf

    assert not vdf.empty
    assert not pdf.empty

    # final preprocessing steps
    # final_price = ciprice(pdf)
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

    assert not vdf.empty
    assert not pdf.empty

    if write:
        desired_path = writepath if writepath is not None else direc + 'datasets/debug/'
        if not os.path.isdir(desired_path):
            os.mkdir(desired_path)

        sd, ed = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
        # write datasets into the debug folder.
        final_vol.to_csv(desired_path + pdt.lower() +
                         '_final_vols_' + sd + '_' + ed + '.csv', index=False)

        final_price.to_csv(desired_path + pdt.lower() +
                           '_final_price_' + sd + '_' + ed + '.csv', index=False)

        pdf.to_csv(desired_path + pdt.lower() +
                   '_roll_df_' + sd + '_' + ed + '.csv', index=False)

        edf.to_csv(desired_path + 'final_option_expiry.csv', index=False)

    return final_vol, final_price, edf, pdf, start_date


def grab_data(pdts, start_date, end_date, ftmth=None, opmth=None, sigpath=None,
              writepath=None,
              direc='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/',
              write=True, test=False, volids=None, write_dump=False):
    """
    Utility function that allows the user to easily grab a dataset by specifying just the product,
    start_date and end_date.


    Args:
        pdts (TYPE): Description
        start_date (TYPE): start date of the dataset desired.
        end_date (TYPE): end date of the dataset desired.
        ftmth (TYPE): future month
        opmth (TYPE): option month
        sigpath (None, optional): Description
        writepath (None, optional): Description
        direc (TYPE, optional): Description
        write (bool, optional): Description
        test (bool, optional): Description
        volids (None, optional): Description
        write_dump (bool, optional): Description

    return:
        pandas dataframe: the data particular to that commodity between start and end dates.

    Deleted Parameters:
        pdt (TYPE): the product being evaluated.
    """
    print('### RUNNING GRAB_DATA ###')
    # print('start_date: ', start_date)
    # print('end_date: ', end_date)

    final_pdf = pd.DataFrame()
    final_vols = pd.DataFrame()
    edf = pd.DataFrame()

    sd = ''.join(start_date.split('-'))
    ed = ''.join(end_date.split('-'))

    desired_path = direc + 'datasets/debug/' if writepath is None else writepath

    pdts = set(pdts)

    for pdt in pdts:
        final_volpath = desired_path + pdt.lower() + '_final_vols_' + \
            sd + '_' + ed + '.csv'
        final_pricepath = desired_path + pdt.lower() + '_final_price_' + \
            sd + '_' + ed + '.csv'
        final_exppath = desired_path + 'final_option_expiry.csv'

        # print('final_volpath: ', final_volpath)
        # print('final_pricepath: ', final_pricepath)
        # print('final exppath: ', final_exppath)

        if (os.path.exists(final_volpath) and
                os.path.exists(final_pricepath) and
                os.path.exists(final_exppath)):
            print('cleaned data found, reading in and returning...')
            vdf = pd.read_csv(final_volpath)
            pdf = pd.read_csv(final_pricepath)
            edf = pd.read_csv(final_exppath)
            # handling datetimes
            vdf.value_date = pd.to_datetime(vdf.value_date)
            pdf.value_date = pd.to_datetime(pdf.value_date)
            edf.expiry_date = pd.to_datetime(edf.expiry_date)

            final_pdf = pd.concat([final_pdf, pdf])
            final_vols = pd.concat([final_vols, vdf])

        else:
            print('cleaned data not found; preparing from dumps')
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
                s = pd.Timestamp(start_date)
                e = pd.Timestamp(end_date)
                diff = (e-s).days
                if diff > 300:
                    print('Pulling data might take a few minutes.')

                if write_dump:
                    print('pulling and saving dumps')
                    vdf, pdf, raw_df = pull_alt_data(
                        pdt, write_dump=True, direc=direc)
                else:
                    print('pulling relevant data; not saving dumps')
                    vdf, pdf, raw_df = pull_alt_data(
                        pdt, start_date, end_date, write_dump=False, direc=direc)
            else:
                print('dumps exist, reading in')
                vdf = pd.read_csv(volpath)
                pdf = pd.read_csv(price_path)

            # handling datetime formats.
            edf = pull_expdata() if edf.empty else edf
            vdf.value_date = pd.to_datetime(vdf.value_date)
            pdf.value_date = pd.to_datetime(pdf.value_date)
            edf.expiry_date = pd.to_datetime(edf.expiry_date)

            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            # filter according to start/end dates
            vdf = vdf[(vdf.value_date >= start_date) &
                      (vdf.value_date <= end_date)]

            pdf = pdf[(pdf.value_date >= start_date) &
                      (pdf.value_date <= end_date)]

            # print('pdf columns: ', pdf.columns)
            # print('vdf columns: ', vdf.columns)

            if volids is not None:
                relevant_volids = [x for x in volids if x[:2].strip() == pdt]
                uids = [x.split()[0] + '  ' + x.split('.')[1]
                        for x in relevant_volids]
                # print('relevant volids: ', relevant_volids)
                # print('relevant uids: ', uids)
                pdf = pdf[pdf.underlying_id.isin(uids)]
                vdf = vdf[vdf.vol_id.isin(relevant_volids)]

            # try filtering just by uid and vol_id
            if ftmth is not None and opmth is not None:
                u_id = pdt + '  ' + ftmth
                vol_id = pdt + '  ' + opmth + '.' + ftmth
                vdf = vdf[(vdf.vol_id == vol_id)]
                pdf = pdf[(pdf.underlying_id == u_id)]

            vdf, pdf, edf, roll_df, start_date = prep_datasets(vdf, pdf, edf, start_date,
                                                               end_date, pdt, signals=signals,
                                                               test=test, write=write,
                                                               writepath=writepath,
                                                               direc=direc)
            final_pdf = pd.concat([final_pdf, pdf])
            final_vols = pd.concat([final_vols, vdf])

    if volids is not None:
        print('volids: ', volids)
        relevant_volids = volids
        uids = [x.split()[0] + '  ' + x.split('.')[1]
                for x in relevant_volids]
        print('relevant volids: ', relevant_volids)
        print('relevant uids: ', uids)
        final_pdf = final_pdf[final_pdf.underlying_id.isin(uids)]
        final_vols = final_vols[final_vols.vol_id.isin(relevant_volids)]

    # last step: sanity check dates etc.
    sanity_check(final_vols.value_date.unique(),
                 final_pdf.value_date.unique(),
                 pd.to_datetime(start_date),
                 pd.to_datetime(end_date))

    print('### GRAB DATA COMPLETED ###')
    return final_vols, final_pdf, edf


def pull_expdata():
    """Helper method that pulls the relevant products' expiry dates from the database.

    Args:
        pdts (TYPE): Description
    """
    user = 'sumit'
    password = 'Olam1234'
    engine = create_engine('postgresql://' + user + ':' + password +
                           '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()

    init_query = 'select vol_id, expiry_date from public.table_opera_option_expiry'

    df = pd.read_sql_query(init_query, connection)

    # initial processing.
    df.expiry_date = pd.to_datetime(df.expiry_date)

    # assigning year variable
    df['year'] = (df.expiry_date.dt.year % 2000)
    df = df[df.year >= 10]
    # assign product.
    df['product'] = df.vol_id.str[:2].str.strip()
    # filter out USDBRL entries.
    df = df[df['product'] != 'US']
    # assign opmth and ftmth variables.
    df['opmth'] = df.vol_id.str.split().str[1].str.split('.').str[0]
    df['ftmth'] = df.vol_id.str.split().str[1].str.split('.').str[1]
    # filter out weekly options
    df.loc[:, 'weekly'] = df.opmth.str.contains('W')
    df = df[df.weekly == False]

    # convert datetime formats; N11.N11 -> N1.N1
    s = df['opmth'].copy()
    df.ix[:, 'opmth'] = s.str[0] + (pd.to_numeric(s.str[1:]) % 10).astype(str)

    return df


def pull_intraday_data(pdts, start_date=None, end_date=None):
    """Helper method that pulls intraday data from the DB. 

    Args:
        pdts (TYPE): Products for which intraday data is required
        start_date (None, optional): start date
        end_date (None, optional): end date

    Returns:
        TYPE: Dataframe
    """
    user = 'sumit'
    password = 'Olam1234'
    engine = create_engine('postgresql://' + user + ':' + password +
                           '@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
    connection = engine.connect()
    print('constructing query...')
    query = construct_intraday_query(
        pdts, start_date=start_date, end_date=end_date)
    print('query: ', query)

    # fetch the dataframe from the db.
    t = time.clock()
    print('fetching intraday data...')
    df = pd.read_sql_query(query, connection)
    print('fetch completed. elapsed: ', time.clock() - t)

    df = handle_intraday_conventions(df)

    # cleaning up the columns in preparation for clean_data call
    # df.date_time = pd.to_datetime(df.date_time)
    # df['time'] = df.date_time.dt.time
    # df['value_date'] = df.date_time.dt.date
    # df['pdt'] = df.commodity.str[:2].str.strip()
    # df['ftmth'] = df.commodity.str[2:-6].str.strip()
    # df['underlying_id'] = df.pdt + '  ' + df.ftmth

    # # setting datatype: intraday vs settlements
    # df['datatype'] = 'intraday'

    cols = ['value_date', 'time', 'underlying_id',
            'pdt', 'ftmth', 'price', 'datatype']
    df = df[cols]

    df.value_date = pd.to_datetime(df.value_date)
    df.time = df.time.astype(pd.Timestamp)
    df.sort_values(by=['value_date', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def construct_intraday_query(pdts, start_date=None, end_date=None):
    """Helper method that generates the SQL query for pulling from the intraday table. 

    Args:
        pdts (TYPE): Description
        start_date (None, optional): Description
        end_date (None, optional): Description
    """
    init_query = 'select commodity, date_time, trade_type, price from public.table_intra_day_trade_cumulative_volume_group where ('

    # generate product query
    pdt_str = ''
    for pdt in pdts:
        space = 1 if len(pdt) == 1 else 0
        pdt_str += ' commodity like ' + "'" + pdt + ' '*space + '%%' + "'"
        if pdt != pdts[-1]:
            pdt_str += ' or '
        else:
            pdt_str += ' ) '

    init_query += pdt_str

    date_str = ''

    # add in date conditions
    if start_date is not None or end_date is not None:
        date_str += ' and ('
        if start_date is not None:
            date_str += ' date(date_time) >= ' + "'" + start_date + "'"
            if end_date is None:
                date_str += ' )'
            else:
                date_str += ' and '

        if end_date is not None:
            date_str += ' date(date_time) <= ' + "'" + end_date + "'" + ')'

    init_query += ' ' + date_str

    # print('init_query: ', init_query)
    return init_query
