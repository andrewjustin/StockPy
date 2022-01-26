""" Technical """

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
