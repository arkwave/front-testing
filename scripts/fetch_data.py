# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-05-17 15:34:51
# @Last Modified by:   Ananth
# @Last Modified time: 2017-05-22 17:40:13

import pandas as pd
from sqlalchemy import create_engine


contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'LCC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
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
    engine = create_engine(
        'postgresql://sumit:Olam1234@gmoscluster.cpmqxvu2gckx.us-west-2.redshift.amazonaws.com:5439/analyticsdb')
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

    # filter dates that are not common to both dataframes
    # diff_pdf = [x for x in pdf.value_date.unique(
    # ) if x not in vdf.value_date.unique()]
    # diff_vdf = [x for x in vdf.value_date.unique(
    # ) if x not in pdf.value_date.unique()]

    # vmask = vdf.value_date.isin(diff_vdf)
    # pmask = pdf.value_date.isin(diff_pdf)

    # vdf = vdf[~vmask]
    # pdf = pdf[~pmask]

    # sorting by date
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


def fetch_alt_data_table():
    pass
