""" Tools """

import numpy as np
from pandas import read_csv


def download_stock_data(symbol, interval='daily'):

    if interval == 'daily':
        interval_identifier = '1d'
    elif interval == 'weekly':
        interval_identifier = '5d'
    elif interval == 'monthly':
        interval_identifier = '1mo'
    else:
        raise ValueError("Invalid interval. Possible values are: 'daily', 'weekly', 'monthly'")

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval={interval_identifier}&events=history"
    stock_data = read_csv(url)

    return stock_data


def get_indicator_indices(dates, start_date=None, end_date=None):
    """ Get indices """

    if start_date is not None:
        start_index = np.where(dates == start_date)[0][0]
    else:
        start_index = 0

    if end_date is not None:
        end_index = np.where(dates == end_date)[0][0]
    else:
        end_index = len(dates)

    return start_index, end_index


def split_dataframe(stock_df, start_date=None, end_date=None):
    """ Split stock df """
    dates = stock_df['Date'].values
    opens = stock_df['Open'].values
    highs = stock_df['High'].values
    lows = stock_df['Low'].values
    closes = stock_df['Close'].values
    volumes = stock_df['Volume'].values

    if start_date is not None and end_date is not None:
        opens = slice_data_by_date(dates, opens, start_date, end_date)
        highs = slice_data_by_date(dates, highs, start_date, end_date)
        lows = slice_data_by_date(dates, lows, start_date, end_date)
        closes = slice_data_by_date(dates, closes, start_date, end_date)
        volumes = slice_data_by_date(dates, volumes, start_date, end_date)
        dates = select_date_range(dates, start_date, end_date)
        return dates, opens, highs, lows, closes, volumes
    else:
        return dates, opens, highs, lows, closes, volumes


def select_date_range(dates, start_date, end_date):
    """ Slice dates """

    dates = np.array(dates, dtype='datetime64[D]')
    start_date = np.array([start_date], dtype='datetime64[D]')[0]
    end_date = np.array([end_date], dtype='datetime64[D]')[0]

    start_date_index = np.where(dates == start_date)[0]
    if len(start_date_index) == 0:
        raise ValueError(f"{start_date} is not in the array of dates, try a different date.")
    end_date_index = np.where(dates == end_date)[0]
    if len(end_date_index) == 0:
        raise ValueError(f"{end_date} is not in the array of dates, try a different date.")

    dates = dates[start_date_index[0]:end_date_index[0]+1]

    return dates


def slice_data_by_date(dates, data, start_date, end_date):
    """ Slice data by date """

    dates = np.array(dates, dtype='datetime64[D]')
    start_date = np.array([start_date], dtype='datetime64[D]')[0]
    end_date = np.array([end_date], dtype='datetime64[D]')[0]

    start_date_index = np.where(dates == start_date)[0]
    if len(start_date_index) == 0:
        raise ValueError(f"{start_date} is not in the array of dates, try a different date.")
    end_date_index = np.where(dates == end_date)[0]
    if len(end_date_index) == 0:
        raise ValueError(f"{end_date} is not in the array of dates, try a different date.")

    data = data[start_date_index[0]:end_date_index[0]+1]

    return data
