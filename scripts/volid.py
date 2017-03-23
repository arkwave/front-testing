import pandas as pd

edf = pd.read_csv('../datasets/option_expiry_from 2008.csv')
volid = 'C  Z7.Z7'
decade = 10
df = pd.read_csv('../datasets/settlement_vol_surf_table.csv')

s = df['vol_id']
overall = pd.Series(['Z17.Z17' for _ in range(5426)])
prod = pd.Series(['C' for _ in range(5426)])
# print('s: ', s)

def get_expiry_date(volid, edf):
    # handle differences in format

    target = volid.split()
    op_yr = pd.to_numeric(target[1][1]) + decade
    op_yr = op_yr.astype(str)
    un_yr = pd.to_numeric(target[1][-1]) + decade
    un_yr = un_yr.astype(str)
    
    op_mth = target[1][0]
    un_mth = target[1][3]
    prod = target[0]
    # print(prod)

    overall = op_mth + op_yr + '.' + un_mth + un_yr
    # t1 = op_mth.str.cat(op_yr.values)
    # t2 = un_mth.str.cat(un_yr.values)
    # overall = t1.str.cat(t2, sep='.')
    # print(overall)
    # print(len(overall))
    # prod.name = 'product'
    # print(prod.name)
    expdate = edf[(edf['vol_id'] == overall) & (edf['product'] == prod)]['expiry_date']
    expdate = pd.to_datetime(expdate)
    # print(expdate)
    return expdate


def ttm(df, s):
    """Takes in a vol_id (for example C Z7.Z7) and outputs the time to expiry for the option in years """
    # print(type(sim_start))
    # print(sim_start)
    s = s.unique()
    df['tau'] = ''
    for iden in s:
        # print(iden)
        # full = 
        expdate = get_expiry_date(iden, edf)
        currdate = pd.to_datetime(df[(df['vol_id'] == iden)]['value_date'])
        print('currdate: ', currdate)
        print('expdate: ', expdate)
        timedelta = currdate - expdate
        print('diff: ', timedelta)
        df[(df['vol_id'] == iden)]['tau']  = timedelta
    return df


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
        # calculating time to expiry
        df = ttm(df, df['vol_id'])
    return df

