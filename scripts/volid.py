import pandas as pd

edf = pd.read_csv('../datasets/option_expiry_from 2008.csv')
volid = 'C  Z7.Z7'
decade = 10
df = pd.read_csv('../datasets/settlement_vol_surf_table.csv')

s = df['vol_id']
# print('s: ', s)

def get_expiry_date(volid, edf):
    # handle differences in format
    target = volid.str.split()
    op_yr = pd.to_numeric(target.str[1].str[1]) + decade
    op_yr = op_yr.to_string()
    un_yr = pd.to_numeric(target.str[1].str[-1]) + decade
    un_yr = un_yr.to_string()
    
    op_mth = target.str[1].str[0]
    un_mth = target.str[1].str[3]
    prod = target.str[0]
    
    # print(prod)
    overall = op_mth.str.cat(op_yr) #.str.cat('.').str.cat(un_mth).str.cat(un_yr)
    print(overall)  
    # print(prod)
    expdate = edf[(edf['vol_id'] == overall) & (edf['product'] == prod) ]['expiry_date']
    expdate = pd.to_datetime(expdate)
    return expdate


def ttm(df, s):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
    # print(type(sim_start))
    # print(sim_start)
    expdate = get_expiry_date(s, edf)
    currdate = pd.to_datetime(df[(df['vol_id'] == s)]['value_date'])
    ttm = currdate - expdate
    print(ttm)
    return ttm


def clean_data(df, flag):
    """Summary

    Args:
        df (TYPE): Description
        flag (TYPE): Description

    Returns:
        TYPE: Description
    """

    df = df.dropna()
    # adjusting for datetime stuff
    df['value_date'] = pd.to_datetime(df['value_date'])
    if flag == 'vol':
        # cleaning volatility data
        df = df.dropna()
        # df['Product'] = df['vol_id'].str.split().str[0]
        # df['Month'] = df['vol_id'].str.split().str[1].str.split('.').str[1]
        # calculating time to expiry
        opmth = ttm(
            df, df['vol_id'])
        df['tau'] = opmth

    return df

