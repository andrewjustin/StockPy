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


class Stock(object):
    def __init__(self, symbol, start_date=None, end_date=None, interval='daily'):
        self.stock_df = download_stock_data(str(symbol), interval=str(interval))
        if start_date is not None:
            start_index = np.where(self.dates() == str(start_date))[0]
            if len(start_index) == 0:
                raise ValueError(f"{str(start_date)} is not in the array of dates, try a different date.")
            else:
                self.start_index = start_index[0]
        else:
            self.start_index = 0

        if end_date is not None:
            end_index = np.where(self.dates() == str(end_date))[0]
            if len(end_index) == 0:
                raise ValueError(f"{str(end_date)} is not in the array of dates, try a different date.")
            else:
                self.end_index = end_index[0]
        else:
            self.end_index = len(self.dates())-1

    def dates(self):
        return self.stock_df['Date'].values

    def opens(self):
        return self.stock_df['Open'].values

    def highs(self):
        return self.stock_df['High'].values

    def lows(self):
        return self.stock_df['Low'].values

    def closes(self):
        return self.stock_df['Close'].values

    def volumes(self):
        return self.stock_df['Volume'].values
