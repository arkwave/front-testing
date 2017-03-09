'''
TODO: 1) price/vol series transformation
      2) identify structure of the data.
'''


def read_data():
    # read in the appropriate data. Finish when dataset is concrete.


def preprocess_data(portfolio, df):
    df = clean_data(df)
    s_list = portfolio.get_underlying()
    tdf = df.copy()
    # TODO reading in the vols. specify this once the dataset is concrete.
    tdf = tdf[s_list]
    return tdf


def prep_data(portfolio):
    df = read_data()
    df = preprocess_data(portfolio, df)
    return df


def clean_data(df):
    # function that cleans data (i.e. standardizes names, all that boring
    # stuff).
    pass
