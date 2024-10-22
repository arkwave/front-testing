# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-17 15:34:51
# @Last Modified by:   arkwave
# @Last Modified time: 2018-01-03 19:34:01

# import time
import datetime as dt
import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import create_engine
import time
import os
import numpy as np
from .prep_data import match_to_signals, get_min_start_date, clean_data, sanity_check
from .global_vars import main_direc
from joblib import Parallel, delayed
import getpass

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
    'CA': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


def pull_settlement_data(pdt, start_date=None, end_date=None, write_dump=False,
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

    user = getpass.getuser()
    password = getpass.getpass()
    engine_url = input("DB Url: ")
    
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')

    engine = create_engine('postgresql://' + user + ':' + password + '@' + engine_url)
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
    connection.close()

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
              'call_put_id', 'strike', 'implied_vol', 'option_expiry_date']]
    pdf = df[['settlement_date', 'underlying_id', 'future_settlement_value']]

    vdf.columns = ['value_date', 'vol_id',
                   'call_put_id', 'strike', 'vol', 'expdate']
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
                  direc='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/', volids=None):
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
    # print('datasets start date: ', dataset_start_date)
    # print('prep_data start_date: ', start_date)

    # catch errors
    if (vdf.empty or pdf.empty):
        raise ValueError(
            '[scripts/prep_data.read_data] : ' +
            'Improper start date entered; resultant dataframes are empty')

    # print('vdf.columns: ', vdf.columns)
    # print('pdf.columns: ', pdf.columns)

    vdf.columns = ['value_date', 'vol_id',
                   'call_put_id', 'strike', 'vol', 'expdate']
    pdf.columns = ['value_date', 'underlying_id', 'price']

    # clean dataframes
    print('==== cleaning dataframes ====')
    edf = clean_data(edf, 'exp')
    vdf = clean_data(vdf, 'vol',
                     edf=edf)
    pdf = clean_data(pdf, 'price',
                     edf=edf)
    print('==== done cleaning dataframes ====')
    # reassigning variables
    final_vol = vdf
    final_price = pdf

    assert not vdf.empty
    assert not pdf.empty

    # final preprocessing steps
    # final_price = ciprice(pdf)
    # print('final price: ', final_price)
    # final_vol = civols(vdf, final_price)

    print('sanitizing date ranges')
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
        print('writing cleaned/prepped data')
        desired_path = writepath if writepath is not None else direc + 'datasets/debug/'
        if not os.path.isdir(desired_path):
            os.mkdir(desired_path)

        sd, ed = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
        # write datasets into the debug folder.
        # volids = final_vol.vol_id.unique()
        vid_str = '_'.join([x for x in volids]) if volids is not None else ''
        if vid_str != '':
            vid_str = '_' + vid_str
        final_vol.to_csv(desired_path + pdt.lower() +
                         '_final_vols_' + sd + '_' + ed + vid_str + '.csv', index=False)
        final_price.to_csv(desired_path + pdt.lower() +
                           '_final_price_' + sd + '_' + ed + vid_str + '.csv', index=False)

        pdf.to_csv(desired_path + pdt.lower() +
                   '_roll_df_' + sd + '_' + ed + vid_str + '.csv', index=False)

        edf.to_csv(desired_path + 'final_option_expiry.csv', index=False)

    return final_vol, final_price, edf, pdf, start_date


def construct_data_paths(pdt, sd, ed, desired_path, volid_str=None):
    """Helper method that constructs the paths to the datasets.

    Args:
        pdt (TYPE): product we're interested in
        start_date (TYPE): dataset start
        end_date (TYPE): dataset end
        desired_path (TYPE): dataset write-to location
        volid_str (None, optional): str describing the volids contained.

    Returns:
        tuple: paths to vol data, price data and expiry data respectively.
    """
    volid_str = '_' + volid_str if volid_str != '' else volid_str
    final_volpath = desired_path + pdt.lower() + '_final_vols_' + \
        sd + '_' + ed + volid_str + '.csv'

    final_pricepath = desired_path + pdt.lower() + '_final_price_' + \
        sd + '_' + ed + volid_str + '.csv'

    final_exppath = desired_path + 'final_option_expiry.csv'

    return final_volpath, final_pricepath, final_exppath


def grab_data(pdts, start_date, end_date, ftmth=None, opmth=None, sigpath=None,
              writepath=None,
              direc='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/',
              write=True, test=False, volids=None, write_dump=False, last_pulled=''):
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
        pdt_volids = [x for x in volids if x[
            :2].strip() == pdt] if volids is not None else None
        volid_str = '_'.join([x for x in pdt_volids]
                             ) if pdt_volids is not None else ''

        final_volpath, final_pricepath, final_exppath = \
            construct_data_paths(pdt.lower(), sd, ed,
                                 desired_path, volid_str=volid_str)

        # print('volpath: ', final_volpath)
        # print('pricepath: ', final_pricepath)
        # print('exppath: ', final_exppath)

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
            vdf.time = dt.time.max
            pdf.time = pdf.time.astype(pd.Timestamp)

            final_pdf = pd.concat([final_pdf, pdf])
            final_vols = pd.concat([final_vols, vdf])

        else:
            print('cleaned data not found; preparing from dumps')
            volpath = direc + 'datasets/data_dump/' + pdt.lower() + '_vol_dump.csv'
            price_path = direc + 'datasets/data_dump/' + pdt.lower() + '_price_dump.csv'
            # print('volpath: ', volpath)
            # print('pricepath: ', price_path)

            # check if dataset can be extended.
            if last_pulled != '':
                print(pdt + ' data last pulled at %s. extending datasets.' %
                      last_pulled)
                # construct the paths
                vdf, pdf, edf = extend_data(
                    pdt, start_date, end_date, last_pulled)
                if write:
                    vdf.to_csv(final_volpath, index=False)
                    pdf.to_csv(final_pricepath, index=False)
                    edf.to_csv(final_exppath, index=False)

            # handling signals
            else:
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
                        vdf, pdf, raw_df = pull_settlement_data(
                            pdt, write_dump=True, direc=direc)
                    else:
                        print('pulling relevant data; not saving dumps')
                        vdf, pdf, raw_df = pull_settlement_data(
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
                    relevant_volids = [
                        x for x in volids if x[:2].strip() == pdt]
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
                                                                   direc=direc, volids=volids)
            final_pdf = pd.concat([final_pdf, pdf])
            final_vols = pd.concat([final_vols, vdf])

    if volids is not None:
        # print('volids: ', volids)
        relevant_volids = volids
        uids = [x.split()[0] + '  ' + x.split('.')[1]
                for x in relevant_volids]
        # print('relevant volids: ', relevant_volids)
        # print('relevant uids: ', uids)
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
    user = getpass.getuser()
    password = getpass.getpass()
    engine_url = input("DB Url: ")
    engine = create_engine('postgresql://' + user + ':' + password + '@' + engine_url)
    connection = engine.connect()

    init_query = 'select vol_id, expiry_date from public.table_opera_option_expiry'

    df = pd.read_sql_query(init_query, connection)
    connection.close()

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
    s2 = df['ftmth'].copy()
    df.ix[:, 'opmth'] = s.str[0] + (pd.to_numeric(s.str[1:]) % 10).astype(str)
    df.ix[:, 'ftmth'] = s2.str[0] + \
        (pd.to_numeric(s2.str[1:]) % 10).astype(str)

    df.vol_id = df['product'] + '  ' + df.opmth + '.' + df.ftmth

    return df


def pull_intraday_data(pdts, start_date=None, end_date=None, filepath='', contracts=None):
    """Helper method that pulls intraday data from the DB.

    Args:
        pdts (TYPE): Products for which intraday data is required
        start_date (None, optional): start date
        end_date (None, optional): end date
        filepath (str, optional): Description
        contracts (None, optional): Description

    Returns:
        TYPE: Dataframe
    """
    overnight_pdts = {'BO', 'C', 'KW', 'S', 'SM', 'W', 'CT', 'MW'}

    t = time.clock()

    par = True if len(pdts) > 1 else False

    if not par:
        df = _pull_intraday_data(pdts[0], start_date=start_date, end_date=end_date,
                                 filepath=filepath, contracts=contracts,
                                 overnight_pdts=overnight_pdts)
    else:
        res = Parallel(n_jobs=len(pdts))(delayed(_pull_intraday_data)(pdt, start_date=start_date, end_date=end_date,
                                                                      filepath=filepath, contracts=contracts,
                                                                      overnight_pdts=overnight_pdts) for pdt in pdts)
        df = pd.concat(res)

    print('elapsed: ', time.clock() - t)

    return df


def _pull_intraday_data(pdt, start_date=None, end_date=None, filepath='', contracts=None, overnight_pdts=None):
    """Helper method that is called in pull_intraday_data in parallel if necessary.

    Args:
        pdt (str): product.
        start_date (str, optional): start date
        end_date (str, optional): end date
        filepath (str, optional): path to write the data to.
        contracts (list, optional): contracts to be pulled in particular
        overnight_pdts (set, optional): set of overnight products.

    Returns:
        dataframe: dataframe of cleaned intraday data.
    """
    vid_str = '_'.join([pdt + '  ' + x + '.' + x for x in contracts]
                       ) if contracts is not None else ''

    if filepath == '':
        filepath = 'C:/Users/RMS08/Desktop/Modules/HistoricSimulator/datasets/debug/'

    cleaned = filepath + pdt + '_' + start_date + '_' + \
        end_date + '_intraday_cleaned' + '_' + vid_str + '.csv'

    print('cleaned filepath: ', cleaned)

    if os.path.exists(cleaned):
        df = pd.read_csv(cleaned)
        df.value_date = pd.to_datetime(df.value_date)
        df.time = pd.to_datetime(df.time).dt.time

    # filename = pdt + '_' + start_date + '_' + end_date + '_ ' + vid_str + \
    #     '_intraday_data.csv'
    else:
        print('cleaned file for ' + pdt + ' does not exist, pulling.')
        user = getpass.getuser()
        password = getpass.getpass()
        engine_url = input("DB Url: ")
        engine = create_engine('postgresql://' + user + ':' + password + '@' + engine_url)
        connection = engine.connect()
        print('constructing query...')
        # offset = 1 if pdt in overnight_pdts else 0
        query = construct_intraday_query(
            pdt, start_date, end_date, contracts=contracts)
        print('query: ', query)

        # fetch the dataframe from the db.

        print('fetching intraday data...')
        df = pd.read_sql_query(query, connection)
        connection.close()
        df.date_time = pd.to_datetime(df.date_time, format='%Y-%m-%d %H:%M:%S')
        df.date_time = df.date_time.dt.tz_convert('Asia/Dubai')
        df['value_date'] = df.date_time.dt.date
        df['time'] = df.date_time.dt.time
        df.sort_values(by=['index'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(cleaned, index=False)
    return df


def construct_intraday_query(pdt, start_date, end_date, contracts=None):
    """Summary

    Args:
        pdt (TYPE): product we want to pull data from. 
        start (TYPE): start date
        end (TYPE): Description
        contracts (None, optional): Description

    Returns:
        TYPE: Description
    """
    init_query = 'select * from public.cleaned_intraday_data where ('

    # generate product query
    pdt_str = ''

    if contracts is None:
        pdt_str = " pdt = '%s'" % pdt

    else:
        uids = [pdt + '  ' + mth for mth in contracts]
        uid_str = ", ".join(["'{0}'".format(i) for i in uids])
        pdt_str = ' underlying_id in (%s) ' % uid_str

    init_query += pdt_str + ")"

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
                date_str += ' date(date_time) <= ' + "'" + end_date + "'" + ')'
    init_query += ' ' + date_str + ';'

    # print('init_query: ', init_query)
    return init_query


def construct_ohlc_query(pdts, start_date, end_date):
    """Helper function that constructs the sql query used to pull OHLC data
    from the database.

    Args:
        pdts (TYPE): the products being pulled. 
        start_date (TYPE): simulation start date
        end_date (TYPE): simulation end date
    """
    start_date = pd.to_datetime(start_date)

    # change start date to one business day earlier.
    real_start = (start_date - BDay(1)).strftime('%Y-%m-%d')

    init_query = "select * from view_future_data_for_ananth where valuedate >= " + \
        "'" + real_start + "'" + " and valuedate <= " + "'" + end_date + "'" + ' and '

    for pdt in pdts:
        space = 1 if len(pdt) == 1 else 0
        if pdts.index(pdt) > 0:
            init_query += ' or'
        init_query += "  ticker like " + "'" + pdt + ' '*space + '%%' + "'"

    print('OHLC query: ', init_query)
    return init_query


def pull_ohlc_data(pdts, start_date, end_date, writepath='C:/Users/' + main_direc + '/Desktop/Modules/HistoricSimulator/datasets/debug'):
    """Helper function that pulls/cleans open-high-low-close data from the
    relevant table. 

    Args:
        pdts (TYPE): products being pulled
        start_date (TYPE): startimulation start_date
        end_date (TYPE): simulation end date 
    """

    # check if the data has been saved.
    init_path = writepath + '/' + \
        '_'.join([x for x in pdts]) + '_' + \
        start_date + '_' + end_date + '_ohlc'
    pricepath = init_path + '_price.csv'

    if os.path.exists(pricepath):
        df = pd.read_csv(pricepath)
        df.value_date = pd.to_datetime(df.value_date)
        df.time = pd.to_datetime(df.time).dt.time

    else:
        user = getpass.getuser()
        password = getpass.getpass()
        engine_url = input("DB Url: ")
        engine = create_engine('postgresql://' + user + ':' + password + '@' + engine_url)
        connection = engine.connect()
        print('constructing query...')
        query = construct_ohlc_query(
            pdts, start_date=start_date, end_date=end_date)
        # print('query: ', query)
        df = pd.read_sql_query(query, connection)
        connection.close()

        # rename the columns
        df.columns = ['ticker', 'value_date', 'px_open',
                      'px_high', 'px_low', 'px_settle', 'agg_vol', 'agg_oi']
        df['underlying_id'] = df.ticker.str[:5].str.strip()

        # more tidying up
        df['pdt'] = df.underlying_id.str[:2].str.strip()

        df['ftmth'] = df.underlying_id.str[2:5].str.strip()

        df['underlying_id'] = df.pdt + '  ' + df.ftmth

        cols = ['underlying_id',  'pdt', 'ftmth', 'value_date',
                'px_high', 'px_low', 'px_settle']

        df = df[cols]

        df.sort_values(by='value_date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = handle_open_to_prev_close(df)
        # df = df.groupby('underlying_id').transform(handle_open_to_prev_close)

        # df['px_open'] = df['px_settle'].shift()

        # df.to_csv('prevclose_debug.csv', index=False)

        # melt the dataframe to get identifier/price.
        df = pd.melt(df, id_vars=['underlying_id', 'pdt', 'ftmth', 'value_date'],
                     value_vars=['px_open', 'px_high', 'px_low', 'px_settle'])
        assert not df.empty
        print('OHLC pre-processing: ', df.head(5))

        # assign the datatype
        df.ix[df.variable.isin(['px_open', 'px_high', 'px_low']),
              'datatype'] = 'intraday'
        df.ix[df.variable == 'px_settle', 'datatype'] = 'settlement'

        # assign times. the order of the high and low have yet to be determined, so
        # do nothing yet.
        # df.ix[df.variable == 'px_open', 'time'] = dt.time(20, 59, 59, 999999)
        # df.ix[df.variable == 'px_settle', 'time'] = dt.time.max

        # replace open with prev close.

        df['time'] = dt.time(23, 59, 59, 999999)

        df.columns = ['underlying_id', 'pdt', 'ftmth', 'value_date',
                      'price_id', 'price', 'datatype', 'time']

        df.sort_values(by=['value_date', 'time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # df = df.dropna()

        # handle H13 -> H3
        df.underlying_id = df.pdt + '  ' + \
            df.ftmth.str[0] + (df.ftmth.str[1:].astype(int) % 10).astype(str)

        df = clean_data(df, 'price')
        df.to_csv(pricepath, index=False)

    return df


def handle_open_to_prev_close(df):
    """iterates through the underlying_ids and replaces 
    open with previous day's close price. 

    Args:
        df (TYPE): Dataframe of prices. 
    """
    for uid in df.underlying_id.unique():
        tdf = df[df.underlying_id == uid]
        df.ix[df.underlying_id == uid, 'px_open'] = tdf[
            'px_settle'].shift().fillna(0)

    return df


def extend_data(pdt, start_date, end_date, last_pulled):
    """Utility function that extends settlement datasets from last_pulled to end_date

    Args:
        pdt (TYPE): list of products. 
        start_date (TYPE): start date
        end_date (TYPE): end date desired.
        last_pulled (TYPE): latest date for which data exists. 
    """

    # get the initially existing data.
    # print('fetch_data.extend_data - start, end, last: ',
    #       start_date, end_date, last_pulled)

    ivdf, ipdf, iedf = grab_data([pdt], start_date, last_pulled)

    # pull the new data.
    nvdf, npdf, nedf = grab_data([pdt], last_pulled, end_date, write=False)

    vdf = pd.concat([ivdf, nvdf])
    pdf = pd.concat([ipdf, npdf])

    return vdf, pdf, iedf


def fetch_exchange_timings():
    user = getpass.getuser()
    password = getpass.getpass()
    engine_url = input("DB Url: ")
    engine = create_engine('postgresql://' + user + ':' + password + '@' + engine_url)
    connection = engine.connect()
    query = 'select * from public.bbg_exchange_timings'
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df
