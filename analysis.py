""" Analyze stock """
import numpy as np
import matplotlib.pyplot as plt


def support_resistance(stock_obj, ax=None, period=10, price_range=0.50, **kwargs):
    """
    Find support and resistance levels
    """

    highs = stock_obj.highs()[stock_obj.start_index:stock_obj.end_index+1]
    lows = stock_obj.lows()[stock_obj.start_index:stock_obj.end_index+1]

    critical_levels = np.array([])

    for day in range(period, stock_obj.end_index - stock_obj.start_index):
        max_high = np.max(highs[day-period:day])
        min_high = np.min(highs[day-period:day])
        average_high = np.mean(highs[day-period:day])
        max_low = np.max(lows[day-period:day])
        min_low = np.min(lows[day-period:day])
        average_low = np.mean(lows[day-period:day])
        if max_high - min_high < price_range:
            if ax is None:
                if len(critical_levels) == 0:
                    critical_levels = np.append(critical_levels, max_high)
                    plt.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=max_high+(0.2*(max_high-average_high)), color='black', **kwargs)

                elif np.min(np.abs(critical_levels - max_high)) > price_range:
                    critical_levels = np.append(critical_levels, max_high)
                    plt.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=max_high+(0.2*(max_high-average_high)), color='black', **kwargs)
            else:
                if len(critical_levels) == 0:
                    critical_levels = np.append(critical_levels, max_high)
                    ax.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=max_high+(0.2*(max_high-average_high)), color='black', **kwargs)
                elif np.min(np.abs(critical_levels - max_high)) > price_range:
                    critical_levels = np.append(critical_levels, max_high)
                    ax.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=max_high+(0.2*(max_high-average_high)), color='black', **kwargs)
        if max_low - min_low < price_range:
            if ax is None:
                if len(critical_levels) == 0:
                    critical_levels = np.append(critical_levels, min_low)
                    plt.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=min_low+(0.2*(average_low-min_low)), color='black', **kwargs)
                elif np.min(np.abs(critical_levels - min_low)) > price_range:
                    critical_levels = np.append(critical_levels, min_low)
                    plt.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=min_low+(0.2*(average_low-min_low)), color='black', **kwargs)
            else:
                if len(critical_levels) == 0:
                    critical_levels = np.append(critical_levels, min_low)
                    ax.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=min_low+(0.2*(average_low-min_low)), color='black', **kwargs)
                elif np.min(np.abs(critical_levels - min_low)) > price_range:
                    critical_levels = np.append(critical_levels, min_low)
                    ax.hlines(xmin=0, xmax=stock_obj.end_index - stock_obj.start_index, y=min_low+(0.2*(average_low-min_low)), color='black', **kwargs)
