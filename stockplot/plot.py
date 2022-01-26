import numpy as np
import matplotlib.pyplot as plt
import indicators

""" Accumulative swing index is currently not working"""
# def accumulative_swing_index(opens, highs, lows, closes, T=300, ax=None, **kwargs):
#     """ ASI """
#     asi = indicators.accumulative_swing_index(opens, highs, lows, closes, T=T)
#
#     if ax is None:
#         plt.plot(asi, **kwargs)
#     else:
#         ax.plot(asi, **kwargs)


def average_directional_index(highs, lows, closes, start_index=0, end_index=None, ax=None, period=14, **kwargs):
    """ ADX """

    if end_index is None:
        if start_index - period > 0:
            highs = highs[start_index - period:]
            lows = lows[start_index - period:]
            closes = closes[start_index - period:]
    else:
        if start_index - period < 0:
            highs = highs[:end_index+1]
            lows = lows[:end_index+1]
            closes = closes[:end_index+1]
        else:
            highs = highs[start_index - period:end_index+1]
            lows = lows[start_index - period:end_index+1]
            closes = closes[start_index - period:end_index+1]

    adx = indicators.average_directional_index(highs, lows, closes, period=period)[period:]

    if ax is None:
        plt.plot(adx, **kwargs)
    else:
        ax.plot(adx, **kwargs)


def average_true_range(highs, lows, closes, ax=None, start_index=0, end_index=None, period=14, method='simple', **kwargs):
    """ Calculate average true range (ATR) """

    if end_index is None:
        if start_index - period > 0:
            highs = highs[start_index - period:]
            lows = lows[start_index - period:]
            closes = closes[start_index - period:]
    else:
        if start_index - period < 0:
            highs = highs[:end_index+1]
            lows = lows[:end_index+1]
            closes = closes[:end_index+1]
        else:
            highs = highs[start_index - period:end_index+1]
            lows = lows[start_index - period:end_index+1]
            closes = closes[start_index - period:end_index+1]

    atr = indicators.average_true_range(highs, lows, closes, period=period, method=method)[period:]

    if ax is None:
        plt.plot(atr, **kwargs)
    else:
        ax.plot(atr, **kwargs)


def bollinger_bands(prices, ax=None, start_index=0, end_index=None, period=20, std_devs=2, upper_band_color='blue',
    lower_band_color='orange', middle_band_color='black', **kwargs):
    """ Bollinger Bands """

    if end_index is None:
        if start_index - period > 0:
            prices = prices[start_index-period:]
    else:
        if start_index - period < 0:
            prices = prices[:end_index+1]
        else:
            prices = prices[start_index-period:end_index+1]

    bands = indicators.bollinger_bands(prices, period=period, std_devs=std_devs)[:,period:]

    if ax is None:
        plt.plot(bands[0], color=upper_band_color, label=f'Boll. Band - Upper ({period},{std_devs})', **kwargs)
        plt.plot(bands[1], color=middle_band_color, label=f'SMA ({period})', **kwargs)
        plt.plot(bands[2], color=lower_band_color, label=f'Boll. Band - Lower ({period},{std_devs})', **kwargs)
    else:
        ax.plot(bands[0], color=upper_band_color, label=f'Boll. Band - Upper ({period},{std_devs})', **kwargs)
        ax.plot(bands[1], color=middle_band_color, label=f'SMA ({period})', **kwargs)
        ax.plot(bands[2], color=lower_band_color, label=f'Boll. Band - Lower ({period},{std_devs})', **kwargs)


def candles(opens, highs, lows, closes, candle_scale=1.0, ax=None, start_index=0, end_index=None, wick_color='black',
    up_color='green', down_color='red', **kwargs):
    """ Plot candles """
    if len(np.unique([len(opens),len(highs),len(lows),len(closes)])) > 1:
        raise ValueError("Arrays containing opens, highs, lows, and closes must have the same length")

    if end_index is not None:
        opens = opens[start_index:end_index+1]
        highs = highs[start_index:end_index+1]
        lows = lows[start_index:end_index+1]
        closes = closes[start_index:end_index+1]

    if ax is None:
        for timestep in range(len(opens)):
            plt.vlines(x=timestep, ymin=lows[timestep], ymax=highs[timestep], colors=wick_color, linewidth=1.5*candle_scale, **kwargs)
            if opens[timestep] > closes[timestep]:
                plt.vlines(x=timestep, ymin=opens[timestep], ymax=closes[timestep], colors=down_color, linewidth=4*candle_scale, **kwargs)
            else:
                plt.vlines(x=timestep, ymin=closes[timestep], ymax=opens[timestep], colors=up_color, linewidth=4*candle_scale, **kwargs)
        plt.ylabel("Price ($)")
    else:
        for timestep in range(len(opens)):
            ax.vlines(x=timestep, ymin=lows[timestep], ymax=highs[timestep], colors=wick_color, linewidth=1.5*candle_scale, **kwargs)
            if opens[timestep] > closes[timestep]:
                ax.vlines(x=timestep, ymin=opens[timestep], ymax=closes[timestep], colors=down_color, linewidth=4*candle_scale, **kwargs)
            else:
                ax.vlines(x=timestep, ymin=closes[timestep], ymax=opens[timestep], colors=up_color, linewidth=4*candle_scale, **kwargs)
        ax.set_ylabel("Price ($)")


def chaikin_volatility(highs, lows, closes, ax=None, start_index=0, end_index=None, ema_period=10, look_back_period=10,
    hlines_kwargs=dict({}), **kwargs):
    """ Chaikin Volatility """
    if end_index is None:
        if start_index - ema_period - look_back_period > 0:
            highs = highs[start_index - ema_period - look_back_period:]
            lows = lows[start_index - ema_period - look_back_period:]
            closes = closes[start_index - ema_period - look_back_period:]
    else:
        if start_index - ema_period - look_back_period < 0:
            highs = highs[:end_index+1]
            lows = lows[:end_index+1]
            closes = closes[:end_index+1]
        else:
            highs = highs[start_index - ema_period - look_back_period:end_index+1]
            lows = lows[start_index - ema_period - look_back_period:end_index+1]
            closes = closes[start_index - ema_period - look_back_period:end_index+1]

    chaik_vol = indicators.chaikin_volatility(highs, lows, closes, ema_period=ema_period, look_back_period=look_back_period)[ema_period+look_back_period:]

    if ax is None:
        plt.plot(chaik_vol, **kwargs)
        plt.hlines(xmin=0, xmax=len(chaik_vol)-1, y=0, color='black', **hlines_kwargs)
    else:
        ax.plot(chaik_vol, **kwargs)
        ax.hlines(xmin=0, xmax=len(chaik_vol)-1, y=0, color='black', **hlines_kwargs)


def exponential_moving_average(prices, ax=None, start_index=0, end_index=None, period=50, multiplier_numerator=2, **kwargs):
    """ EMA """
    if end_index is None:
        if start_index - period > 0:
            prices = prices[start_index-period:]
    else:
        if start_index - period < 0:
            prices = prices[:end_index+1]
        else:
            prices = prices[start_index-period:end_index+1]

    ema = indicators.exponential_moving_average(prices, period=period, multiplier_numerator=multiplier_numerator)[period:]

    if ax is None:
        plt.plot(ema, label=f'EMA ({period})', **kwargs)
    else:
        ax.plot(ema, label=f'EMA ({period})', **kwargs)


def ichimoku_cloud(highs, lows, closes, ax=None, conversion_period=9, base_period=26, span_b_period=52, lagging_period=26, **kwargs):
    """ Ichimoku Cloud """
    print("WARNING: This function is incomplete")
    cloud = indicators.ichimoku_cloud(highs, lows, closes, conversion_period=conversion_period, base_period=base_period,
                                      span_b_period=span_b_period, lagging_period=lagging_period)

    timesteps = np.arange(0, len(highs))

    if ax is None:
        plt.plot(cloud[0], **kwargs)
        plt.plot(cloud[1], **kwargs)
        plt.plot(cloud[2], **kwargs)
        plt.plot(cloud[3], **kwargs)
        plt.fill_between(timesteps, cloud[1], cloud[3])
    else:
        ax.plot(cloud[0], **kwargs)
        ax.plot(cloud[1], **kwargs)
        ax.plot(cloud[2], **kwargs)
        ax.plot(cloud[3], **kwargs)
        ax.fill_between(timesteps, cloud[1], cloud[3])


def money_flow_index(highs, lows, closes, volumes, ax=None, period=14, **kwargs):
    """ MFI """
    mfi = indicators.money_flow_index(highs, lows, closes, volumes, period=period)

    if ax is None:
        plt.plot(mfi, **kwargs)
    else:
        ax.plot(mfi, **kwargs)


def moving_average_convergence_divergence(prices, ax=None, multiplier_numerator=2, **kwargs):
    """ MACD """
    print("WARNING: This function is incomplete")
    macd = indicators.moving_average_convergence_divergence(prices, multiplier_numerator=multiplier_numerator)

    if ax is None:
        plt.plot(macd, **kwargs)
        plt.hlines(xmin=0, xmax=len(prices), y=0, colors='black')
    else:
        ax.plot(macd, **kwargs)
        ax.hlines(xmin=0, xmax=len(prices), y=0, colors='black')


def relative_strength_index(prices, ax=None, start_index=0, end_index=None, period=14, overbought_level=70, oversold_level=30,
    overbought_color='red', oversold_color='green', hlines_kwargs=dict({}), **kwargs):
    """ Calculate RSI and add it to the dataset """

    if end_index is None:
        if start_index - period > 0:
            prices = prices[start_index-period:]
    else:
        if start_index - period < 0:
            prices = prices[:end_index+1]
        else:
            prices = prices[start_index-period:end_index+1]

    rsi = indicators.relative_strength_index(prices, period=period)[period:]

    if ax is None:
        plt.plot(rsi, label=f'RSI ({period})', **kwargs)
        plt.hlines(xmin=0, xmax=len(rsi)-1, y=overbought_level, color=overbought_color, **hlines_kwargs)
        plt.hlines(xmin=0, xmax=len(rsi)-1, y=oversold_level, color=oversold_color, **hlines_kwargs)
        plt.ylim(0,100)
        plt.ylabel(f'RSI ({period})')
        plt.yticks([0,oversold_level,50,overbought_level,100])
    else:
        ax.plot(rsi, label=f'RSI ({period})', **kwargs)
        ax.hlines(xmin=0, xmax=len(rsi)-1, y=overbought_level, color=overbought_color, **hlines_kwargs)
        ax.hlines(xmin=0, xmax=len(rsi)-1, y=oversold_level, color=oversold_color, **hlines_kwargs)
        ax.set_ylim(0,100)
        ax.set_ylabel(f'RSI ({period})')
        ax.set_yticks([0,oversold_level,50,overbought_level,100])


def simple_moving_average(prices, ax=None, start_index=0, end_index=None, period=50, **kwargs):
    """ SMA """

    if end_index is None:
        if start_index - period > 0:
            prices = prices[start_index-period:]
    else:
        if start_index - period < 0:
            prices = prices[:end_index+1]
        else:
            prices = prices[start_index-period:end_index+1]

    sma = indicators.simple_moving_average(prices, period=period)

    if ax is None:
        plt.plot(sma[period:], label=f'SMA ({period})', **kwargs)
    else:
        ax.plot(sma[period:], label=f'SMA ({period})', **kwargs)


def stochastic_oscillator(highs, lows, closes, ax=None, overbought=80, oversold=20, **kwargs):
    """ Stochastic Oscillator """
    stoch = indicators.stochastic_oscillator(highs, lows, closes)

    if ax is None:
        plt.plot(stoch, **kwargs)
        plt.hlines(xmin=0, xmax=len(highs), y=overbought, colors='red')
        plt.hlines(xmin=0, xmax=len(highs), y=oversold, colors='green')
        plt.ylim(0,100)
        plt.yticks([0,oversold,50,overbought,100])
    else:
        ax.plot(stoch, **kwargs)
        ax.hlines(xmin=0, xmax=len(highs), y=overbought, colors='red')
        ax.hlines(xmin=0, xmax=len(highs), y=oversold, colors='green')
        ax.set_ylim(0,100)
        ax.set_yticks([0,oversold,50,overbought,100])


""" Swing index is not complete """
# def swing_index(opens, highs, lows, closes, ax=None, T=300, **kwargs):
#     """ SI """
#     si = indicators.swing_index(opens, highs, lows, closes, T=T)
#
#     if ax is None:
#         plt.plot(si, **kwargs)
#     else:
#         ax.plot(si, **kwargs)


def volume(volumes, ax=None, start_index=0, end_index=None, height_scale=1.0, add_yticks=False, **kwargs):
    """ volume """
    if end_index is None:
        volumes = volumes[start_index:]
    else:
        volumes = volumes[start_index:end_index+1]

    if ax is None:
        plt.plot(volumes, **kwargs)
    else:
        ax2 = ax.twinx()
        ax2.set_ylim(0, np.max(volumes))
        if add_yticks is False:
            ax2.set_yticks([])
            ax2.set_yticklabels([])
        for timestep in range(len(volumes)):
            ax2.bar(timestep, volumes[timestep]*height_scale/10, **kwargs)


def weighted_moving_average(prices, ax=None, period=50, **kwargs):
    """ WMA """
    wma = indicators.weighted_moving_average(prices, period=period)

    if ax is None:
        plt.plot(wma, **kwargs)
    else:
        ax.plot(wma, **kwargs)


def xticks(dates, period='D', ax=None, **kwargs):
    """ xticks """
    dates = np.array(dates, dtype=f'datetime64[{period}]')
    uniques = np.unique(dates)
    unique_index_array = np.empty([len(uniques)],dtype=int)
    for index in range(len(unique_index_array)):
        unique_index_array[index] = np.min(np.where(dates == uniques[index]))
    new_dates = dates[unique_index_array]

    if ax is None:
        plt.xticks(unique_index_array, new_dates, **kwargs)
    else:
        if len(ax) > 0:
            for axis in ax:
                axis.set_xticks(unique_index_array)
                axis.set_xticklabels(new_dates, **kwargs)
        else:
            ax.set_xticks(unique_index_array)
            ax.set_xticklabels(new_dates, **kwargs)
