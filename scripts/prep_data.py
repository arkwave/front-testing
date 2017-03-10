'''
TODO: 1) price/vol series transformation
      2) identify structure of the data.
'''

filepath = '..\portfolio_specs.txt'
from portfolio import Portfolio
from classes import Option, Future


def read_data():
    # read in the appropriate data. Finish when dataset is concrete.
    pass


def preprocess_data(portfolio, df):
    df = clean_data(df)
    s_list = portfolio.get_underlying_names()
    tdf = df.copy()
    # TODO reading in the vols. specify this once the dataset is concrete.
    tdf = tdf[s_list]
    return tdf


def prep_data(portfolio):
    df = read_data()
    df = preprocess_data(portfolio, df)
    return df


def prep_portfolio():
    """Reads in portfolio specifications from portfolio_specs.txt"""
    pf = Portfolio()
    with open(filepath) as f:
        for line in f:
            if "%%" in line or line in ['\n', '\r\n']:
                continue
            else:
                inputs = line.split(',')
                print(inputs)
                # input specifies an option
                if inputs[0] == 'Option':
                    strike = float(inputs[1])
                    tau = float(inputs[2])
                    char = inputs[3][1:-1]
                    vol = float(inputs[4])
                    payoff = inputs[5][1:-1]
                    flag = inputs[10][1:-2]
                    # handle underlying construction
                    f_mth = inputs[7][1:-1]
                    f_name = inputs[8][1:-1]
                    f_price = float(inputs[9])
                    underlying = Future(f_mth, f_name, f_price)
                    opt = Option(strike, tau, char, vol, underlying, payoff)
                    pf.add_security(opt, flag)
                # input specifies a future
                elif inputs[0] == 'Future':
                    mth = inputs[1][1:-1]
                    name = inputs[2][1:-1]
                    price = float(inputs[3])
                    flag = inputs[4][1:-1]
                    ft = Future(mth, name, price)
                    pf.add_security(ft, flag)
    print('Shorts: ', pf.short_pos)
    print('Longs: ', pf.long_pos)
    print('Net Greeks: ', pf.net_greeks)
    return pf


def clean_data(df):
    # function that cleans data (i.e. standardizes names, all that boring
    # stuff).
    pass
