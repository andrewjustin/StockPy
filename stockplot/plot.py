import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import indicators


def average_directional_index(stock_object, ax=None, period=14, **plot_kwargs):
    """ ADX """
    adx = indicators.average_directional_index(stock_object, period=period)[period:]

    if ax is None:
        plt.plot(adx, label=f'ADX {period}', **plot_kwargs)
    else:
        ax.plot(adx, label=f'ADX {period}', **plot_kwargs)


def average_true_range(stock_object, ax=None, period=14, method='simple', **plot_kwargs):
    """ Calculate ATR """
    atr = indicators.average_true_range(stock_object, period=period, method=method)[period:]

    if ax is None:
        plt.ylabel("ATR")
        plt.plot(atr, label=f'ATR ({period})', **plot_kwargs)
    else:
        ax.set_ylabel("ATR")
        ax.plot(atr, label=f'ATR ({period})', **plot_kwargs)


def bollinger_bands(stock_object, ax=None, period=20, std_devs=2, upper_band_color='blue',
    lower_band_color='orange', middle_band_color='black', **plot_kwargs):
    """ Bollinger Bands """
    bands = indicators.bollinger_bands(stock_object, period=period, std_devs=std_devs)[:,period:]

    if ax is None:
        plt.plot(bands[0], color=upper_band_color, label=f'Boll. Band - Upper ({period},{std_devs})', **plot_kwargs)
        plt.plot(bands[1], color=middle_band_color, label=f'SMA ({period})', **plot_kwargs)
        plt.plot(bands[2], color=lower_band_color, label=f'Boll. Band - Lower ({period},{std_devs})', **plot_kwargs)
    else:
        ax.plot(bands[0], color=upper_band_color, label=f'Boll. Band - Upper ({period},{std_devs})', **plot_kwargs)
        ax.plot(bands[1], color=middle_band_color, label=f'SMA ({period})', **plot_kwargs)
        ax.plot(bands[2], color=lower_band_color, label=f'Boll. Band - Lower ({period},{std_devs})', **plot_kwargs)


def candles(stock_object, candle_scale=1.0, ax=None, wick_color='black', up_color='green', down_color='red', **vlines_kwargs):
    """ Plot candles """
    opens = stock_object.opens()
    highs = stock_object.highs()
    lows = stock_object.lows()
    closes = stock_object.closes()
    start_index = stock_object.start_index
    end_index = stock_object.end_index

    if len(np.unique([len(opens),len(highs),len(lows),len(closes)])) > 1:
        raise ValueError("Arrays containing opens, highs, lows, and closes must have the same length")

    if end_index is not None:
        opens = opens[start_index:end_index+1]
        highs = highs[start_index:end_index+1]
        lows = lows[start_index:end_index+1]
        closes = closes[start_index:end_index+1]
    else:
        opens = opens[start_index:]
        highs = highs[start_index:]
        lows = lows[start_index:]
        closes = closes[start_index:]

    price_format = lambda x, pos: str(x).rstrip('0').rstrip('.')

    if ax is None:
        for timestep in range(len(opens)):
            plt.vlines(x=timestep, ymin=lows[timestep], ymax=highs[timestep], colors=wick_color, linewidth=1.5*candle_scale, **vlines_kwargs)
            if opens[timestep] > closes[timestep]:
                plt.vlines(x=timestep, ymin=opens[timestep], ymax=closes[timestep], colors=down_color, linewidth=4*candle_scale, **vlines_kwargs)
            else:
                plt.vlines(x=timestep, ymin=closes[timestep], ymax=opens[timestep], colors=up_color, linewidth=4*candle_scale, **vlines_kwargs)
        plt.ylabel("Price ($)")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(price_format))
    else:
        for timestep in range(len(opens)):
            ax.vlines(x=timestep, ymin=lows[timestep], ymax=highs[timestep], colors=wick_color, linewidth=1.5*candle_scale, **vlines_kwargs)
            if opens[timestep] > closes[timestep]:
                ax.vlines(x=timestep, ymin=opens[timestep], ymax=closes[timestep], colors=down_color, linewidth=4*candle_scale, **vlines_kwargs)
            else:
                ax.vlines(x=timestep, ymin=closes[timestep], ymax=opens[timestep], colors=up_color, linewidth=4*candle_scale, **vlines_kwargs)
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter(FuncFormatter(price_format))


def chaikin_volatility(stock_object, ax=None, ema_period=10, look_back_period=10,
    hlines_kwargs=dict({}), **plot_kwargs):
    """ Chaik. Vol. """
    chaik_vol = indicators.chaikin_volatility(stock_object, ema_period=ema_period, look_back_period=look_back_period)[ema_period+look_back_period:]

    if ax is None:
        plt.ylabel("Chaik. Vol.")
        plt.plot(chaik_vol, label=f'Chaik. Vol. ({ema_period},{look_back_period})', **plot_kwargs)
        plt.hlines(xmin=0, xmax=len(chaik_vol)-1, y=0, color='black', **hlines_kwargs)
    else:
        ax.set_ylabel("Chaik. Vol.")
        ax.plot(chaik_vol, label=f'Chaik. Vol. ({ema_period},{look_back_period})', **plot_kwargs)
        ax.hlines(xmin=0, xmax=len(chaik_vol)-1, y=0, color='black', **hlines_kwargs)


def exponential_moving_average(stock_object, ax=None, period=50, multiplier_numerator=2, **plot_kwargs):
    """ EMA """
    ema = indicators.exponential_moving_average(stock_object, period=period, multiplier_numerator=multiplier_numerator)[period:]

    if ax is None:
        plt.plot(ema, label=f'EMA ({period})', **plot_kwargs)
    else:
        ax.plot(ema, label=f'EMA ({period})', **plot_kwargs)


def ichimoku_cloud(stock_object, ax=None, conversion_period=9, base_period=26, span_b_period=52, lagging_period=26, span_a_offset=0,
    span_b_offset=0, conversion_color='black', base_color='red', span_a_color='purple', span_b_color='blue', **plot_kwargs):
    """ Ichimoku Cloud """
    cloud = indicators.ichimoku_cloud(stock_object, conversion_period=conversion_period, base_period=base_period,
        span_b_period=span_b_period, lagging_period=lagging_period, span_a_offset=span_a_offset, span_b_offset=span_b_offset)

    timesteps = np.arange(0, len(cloud[0]))

    if ax is None:
        plt.plot(cloud[0], color=conversion_color, label=f'IC - Conv. Line ({conversion_period})', **plot_kwargs)
        plt.plot(cloud[1], color=base_color, label=f'IC - Base Line ({base_period})', **plot_kwargs)
        plt.plot(cloud[2], color=span_a_color, label=f'IC - Span A ({span_a_offset})', **plot_kwargs)
        plt.plot(cloud[3], color=span_b_color, label=f'IC - Span B ({span_b_period, span_b_offset})', **plot_kwargs)
        plt.fill_between(timesteps+span_a_offset, cloud[2][span_a_offset:], cloud[3][span_b_offset:], where=cloud[3][span_b_offset:] > cloud[2][span_a_offset:],
            color='red', alpha=0.4)
        plt.fill_between(timesteps+span_a_offset, cloud[2][span_a_offset:], cloud[3][span_b_offset:], where=cloud[3][span_b_offset:] < cloud[2][span_a_offset:],
            color='green', alpha=0.4)
    else:
        ax.plot(cloud[0], color=conversion_color, label=f'IC - Conv. Line ({conversion_period})', **plot_kwargs)
        ax.plot(cloud[1], color=base_color, label=f'IC - Base Line ({base_period})', **plot_kwargs)
        ax.plot(cloud[2], color=span_a_color, label=f'IC - Span A ({span_a_offset})', **plot_kwargs)
        ax.plot(cloud[3], color=span_b_color, label=f'IC - Span B ({span_b_period, span_b_offset})', **plot_kwargs)
        ax.fill_between(timesteps+span_a_offset, cloud[2][span_a_offset:], cloud[3][span_b_offset:], where=cloud[3][span_b_offset:] > cloud[2][span_a_offset:],
            color='red', alpha=0.4)
        ax.fill_between(timesteps+span_a_offset, cloud[2][span_a_offset:], cloud[3][span_b_offset:], where=cloud[3][span_b_offset:] < cloud[2][span_a_offset:],
            color='green', alpha=0.4)


def money_flow_index(stock_object, ax=None, period=14, **plot_kwargs):
    """ MFI """
    mfi = indicators.money_flow_index(stock_object, period=period)[period+1:]

    if ax is None:
        plt.ylabel("MFI")
        plt.plot(mfi, label=f'MFI ({period})', **plot_kwargs)
    else:
        ax.set_ylabel("MFI")
        ax.plot(mfi, label=f'MFI ({period})', **plot_kwargs)


def moving_average_convergence_divergence(stock_object, ax=None, short_ema=12, long_ema=26,
    signal_ema=9, multiplier_numerator=2, histogram=True, histogram_up_color='green', histogram_down_color='red',
    hlines_kwargs=dict({}), vlines_kwargs=dict({}), **plot_kwargs):
    """ MACD """
    macd, signal = indicators.moving_average_convergence_divergence(stock_object,
        short_ema=short_ema, long_ema=long_ema, signal_ema=signal_ema, multiplier_numerator=multiplier_numerator)

    macd = macd[long_ema+signal_ema:]
    signal = signal[signal_ema:]

    if ax is None:
        plt.hlines(xmin=0, xmax=len(macd)-1, y=0, colors='gray', **hlines_kwargs)
        if histogram is True:
            for day in range(len(macd)):
                if macd[day]-signal[day] < 0:
                    plt.vlines(x=day, ymin=macd[day]-signal[day], ymax=0, color=histogram_down_color, **vlines_kwargs)
                else:
                    plt.vlines(x=day, ymin=0, ymax=macd[day]-signal[day], color=histogram_up_color, **vlines_kwargs)
        plt.ylabel("MACD")
        plt.plot(macd, label=f'MACD ({short_ema},{long_ema})', **plot_kwargs)
        plt.plot(signal, label=f'MACD signal ({signal_ema})', color='black', **plot_kwargs)
    else:
        ax.hlines(xmin=0, xmax=len(macd)-1, y=0, colors='gray', **hlines_kwargs)
        if histogram is True:
            for day in range(len(macd)):
                if macd[day]-signal[day] < 0:
                    ax.vlines(x=day, ymin=macd[day]-signal[day], ymax=0, color=histogram_down_color, **vlines_kwargs)
                else:
                    ax.vlines(x=day, ymin=0, ymax=macd[day]-signal[day], color=histogram_up_color, **vlines_kwargs)
        ax.set_ylabel("MACD")
        ax.plot(macd, label=f'MACD ({short_ema},{long_ema})', **plot_kwargs)
        ax.plot(signal, label=f'MACD signal ({signal_ema})', color='black', **plot_kwargs)


def relative_strength_index(stock_object, ax=None, period=14, overbought_level=70, oversold_level=30,
    overbought_color='red', oversold_color='green', hlines_kwargs=dict({}), **plot_kwargs):
    """ Calculate RSI and add it to the stock_objectset """
    rsi = indicators.relative_strength_index(stock_object, period=period)[period:]

    if ax is None:
        plt.plot(rsi, label=f'RSI ({period})', **plot_kwargs)
        plt.hlines(xmin=0, xmax=len(rsi)-1, y=overbought_level, color=overbought_color, **hlines_kwargs)
        plt.hlines(xmin=0, xmax=len(rsi)-1, y=oversold_level, color=oversold_color, **hlines_kwargs)
        plt.ylim(0,100)
        plt.ylabel('RSI')
        plt.yticks([0,oversold_level,50,overbought_level,100])
    else:
        ax.plot(rsi, label=f'RSI ({period})', **plot_kwargs)
        ax.hlines(xmin=0, xmax=len(rsi)-1, y=overbought_level, color=overbought_color, **hlines_kwargs)
        ax.hlines(xmin=0, xmax=len(rsi)-1, y=oversold_level, color=oversold_color, **hlines_kwargs)
        ax.set_ylim(0,100)
        ax.set_ylabel('RSI')
        ax.set_yticks([0,oversold_level,50,overbought_level,100])


def simple_moving_average(stock_object, ax=None, period=50, **plot_kwargs):
    """ SMA """
    sma = indicators.simple_moving_average(stock_object, period=period)

    if ax is None:
        plt.plot(sma[period:], label=f'SMA ({period})', **plot_kwargs)
    else:
        ax.plot(sma[period:], label=f'SMA ({period})', **plot_kwargs)


def stochastic_oscillator(stock_object, ax=None, period=14, overbought=80, oversold=20,
    **plot_kwargs):
    """ Stoch. Oscil. """
    stoch = indicators.stochastic_oscillator(stock_object, period=period)[period:]

    if ax is None:
        plt.ylabel('Stoch. Oscil.')
        plt.plot(stoch, label=f'Stoch. Oscil. ({period})', **plot_kwargs)
        plt.hlines(xmin=0, xmax=len(stoch)-1, y=overbought, colors='red')
        plt.hlines(xmin=0, xmax=len(stoch)-1, y=oversold, colors='green')
        plt.ylim(0,100)
        plt.yticks([0,oversold,50,overbought,100])
    else:
        ax.plot(stoch, label=f'Stoch. Oscil. ({period})', **plot_kwargs)
        ax.hlines(xmin=0, xmax=len(stoch)-1, y=overbought, colors='red')
        ax.hlines(xmin=0, xmax=len(stoch)-1, y=oversold, colors='green')
        ax.set_ylabel('Stoch. Oscil.')
        ax.set_ylim(0,100)
        ax.set_yticks([0,oversold,50,overbought,100])


def volume(stock_object, ax=None, height_scale=1.0, add_yticks=False, **kwargs):
    """ volume """
    volumes = stock_object.volumes()[stock_object.start_index:stock_object.end_index+1]

    if ax is None:
        plt.plot(volumes, label='Volume', **kwargs)
    else:
        ax2 = ax.twinx()
        ax2.set_ylim(0, np.nanmax(volumes))
        if add_yticks is False:
            ax2.set_yticks([])
            ax2.set_yticklabels([])
        for timestep in range(len(volumes)):
            ax2.bar(timestep, volumes[timestep]*height_scale/10, color='blue', alpha=0.25, **kwargs)


def weighted_moving_average(stock_object, ax=None, period=50, **plot_kwargs):
    """ WMA """
    wma = indicators.weighted_moving_average(stock_object, period=period)

    if ax is None:
        plt.plot(wma, label=f'WMA ({period})', **plot_kwargs)
    else:
        ax.plot(wma, label=f'WMA ({period})', **plot_kwargs)


def xticks(stock_object, period='D', ax=None, **kwargs):
    """ xticks """
    dates = np.array(stock_object.dates()[stock_object.start_index:stock_object.end_index], dtype=f'datetime64[{period}]')
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
