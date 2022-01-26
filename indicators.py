""" Technical indicators to plot """

import numpy as np


def accumulative_swing_index(opens, highs, lows, closes, T=300):
    """ ASI """
    print("WARNING: accumulative_swing_index is experimental and may result in incorrect values")
    array_length = len(closes)
    asi_array = np.empty(array_length)
    asi = 0

    for day in range(1,array_length):
        C, Cy = closes[day], closes[day-1]
        H, Hy = highs[day], highs[day-1]
        L, Ly = lows[day], lows[day-1]
        O, Oy = opens[day], opens[day-1]
        K = np.max((H-Cy, Cy-L))
        TR = np.max((H-Cy,Cy-L,H-L))
        if Cy > H:
            ER = abs(H-Cy)
        elif L <= Cy <= H:
            ER = 0
        else:
            ER = abs(Cy-L)
        SH = abs(Cy-Oy)

        R = TR - (0.5*ER) + (0.25*SH)
        si_test = 50*((C-Cy + ((C-O)/2) + ((Cy-Oy)/4))/R)*(K/T)
        if si_test < 1000:
            si = 50*((C-Cy + ((C-O)/2) + ((Cy-Oy)/4))/R)*(K/T)
        asi += si
        asi_array[day] = asi

        return asi_array


def average_directional_index(highs, lows, closes, period=14):
    """ ADX """
    print("WARNING: average_directional_index is experimental and may result in incorrect values")
    array_length = len(closes)

    dx_array = np.empty(array_length)
    dm_plus_array = np.empty(array_length)
    dm_minus_array = np.empty(array_length)
    adx_array = np.empty(array_length)
    tr_array = np.empty(array_length)

    adx_array[:period+1] = np.NaN

    for day in range(1,array_length):
        if highs[day]-highs[day-1] > lows[day-1]-lows[day]:
            dm_plus = highs[day] - highs[day-1]
            dm_minus = 0
        else:
            dm_plus = 0
            dm_minus = lows[day-1] - lows[day]

        tr = np.max((highs[day]-lows[day],highs[day]-closes[day-1],lows[day]-closes[day-1]))
        dm_plus_array[day] = dm_plus
        dm_minus_array[day] = dm_minus
        tr_array[day] = tr

    smoothed_dm_plus = np.mean(dm_plus_array[1:period+1])
    smoothed_dm_minus = np.mean(dm_minus_array[1:period+1])
    atr = np.mean(tr_array[1:period+1])
    di_plus = 100*smoothed_dm_plus/atr
    di_minus = 100*smoothed_dm_minus/atr
    dx = 100*abs(di_plus-di_minus)/abs(di_plus+di_minus)
    dx_array[period] = dx
    for day in range(period+1,array_length):
        smoothed_dm_plus = smoothed_dm_plus*(period-1)/period + dm_plus_array[day]
        smoothed_dm_minus = smoothed_dm_minus*(period-1)/period + dm_minus_array[day]
        atr = atr*(period-1)/period + tr_array[day]
        di_plus = 100*smoothed_dm_plus/atr
        di_minus = 100*smoothed_dm_minus/atr
        dx = 100*abs(di_plus-di_minus)/abs(di_plus+di_minus)
        dx_array[day] = dx

    adx = np.mean(dx_array[1:period+1])
    adx_array[period] = adx
    for day in range(period+1,array_length):
        adx = (adx*(period-1) + dx_array[day])/period
        adx_array[day] = adx

    return adx_array


def average_true_range(highs, lows, closes, period=14, method='simple'):
    """ Calculate average true range (ATR) """

    array_length = len(closes)
    tr_array = np.empty(array_length)
    atr_array = np.empty(array_length)

    atr_array[:period+1] = np.NaN

    for day in range(1,array_length):
        tr_array[day] = np.max((highs[day]-lows[day],highs[day]-closes[day-1],closes[day-1]-lows[day]))

    if method == 'exponential' or method == 'wilder':
        if method == 'exponential':
            smoothing_factor = 2/(period+1)
        else:
            smoothing_factor = 1/period

        atr = 0
        for day in range(1,period+1):
            atr += np.max((highs[day]-lows[day],highs[day]-closes[day-1],closes[day-1]-lows[day]))/period

        for day in range(period+1, array_length):
            tr = np.max((highs[day]-lows[day],highs[day]-closes[day-1],closes[day-1]-lows[day]))
            atr = atr*(1-smoothing_factor) + tr*smoothing_factor
            atr_array[day] = atr

    elif method == 'simple':
        for day in range(period, array_length):
            atr_array[day] = np.mean(tr_array[day-period+1:day+1])

    return atr_array


def bollinger_bands(prices, period=20, std_devs=2):
    """ Bollinger Bands """

    array_length = len(prices)

    bolu_array = np.empty(array_length)
    bold_array = np.empty(array_length)
    ma_array = np.empty(array_length)

    bolu_array[:period] = np.NaN
    bold_array[:period] = np.NaN
    ma_array[:period] = np.NaN

    for day in range(period, array_length):
        ma = np.mean(prices[day-period+1:day+1])
        ma_array[day] = ma
        std_dev = np.std(prices[day-period+1:day+1])
        bolu_array[day] = ma + (std_devs*std_dev)
        bold_array[day] = ma - (std_devs*std_dev)

    bands = np.array([bolu_array, ma_array, bold_array])

    return bands


def chaikin_volatility(highs, lows, closes, ema_period=10, look_back_period=10):
    """ Chaikin Volatility """

    array_length = len(closes)

    chaik_array = np.empty(array_length)
    range_array = highs - lows
    ema_array = np.empty(array_length)

    chaik_array[:ema_period+look_back_period+1] = np.NaN

    ema = np.mean(range_array[0:ema_period+1])
    ema_array[ema_period] = ema
    ema_multiplier = 2/(ema_period+1)
    for day in range(ema_period+1,array_length):
        ema = range_array[day]*ema_multiplier + ema*(1-ema_multiplier)
        ema_array[day] = ema

    for day in range(ema_period+look_back_period, array_length):
        chaik_array[day] = (ema_array[day] - ema_array[day-look_back_period])/ema_array[day-look_back_period]

    return chaik_array


def exponential_moving_average(prices, period=50, multiplier_numerator=2):
    """ EMA """
    array_length = len(prices)
    ema_array = np.empty(array_length)
    ema_array[:period] = np.NaN

    ema = np.mean(prices[0:period])
    ema_array[period] = ema
    multiplier = multiplier_numerator/(period+1)

    for day in range(period, array_length):
        ema = prices[day]*multiplier + ema*(1-multiplier)
        ema_array[day] = ema

    return ema_array


def ichimoku_cloud(highs, lows, closes, conversion_period=9, base_period=26, span_b_period=52, lagging_period=26):
    """ Ichimoku Cloud """
    array_length = len(closes)

    conversion_line_array = np.empty(array_length)
    base_line_array = np.empty(array_length)
    leading_span_a_array = np.empty(array_length)
    leading_span_b_array = np.empty(array_length)

    start_period = int(np.max((conversion_period, base_period, span_b_period, lagging_period)))

    conversion_line_array[:start_period] = np.NaN
    base_line_array[:start_period] = np.NaN
    leading_span_a_array[:start_period] = np.NaN
    leading_span_b_array[:start_period] = np.NaN

    for day in range(start_period, array_length):
        conversion_line_array[day] = (np.mean(highs[day-conversion_period+1:day+1])+np.mean(lows[day-conversion_period+1:day+1]))/2
        base_line_array[day] = (np.mean(highs[day-base_period+1:day+1])+np.mean(lows[day-base_period+1:day+1]))/2
        leading_span_a_array[day] = (conversion_line_array[day] + base_line_array[day])/2
        leading_span_b_array[day] = (np.mean(highs[day-span_b_period+1:day+1])+np.mean(lows[day-span_b_period+1:day+1]))/2

    cloud = np.array([conversion_line_array, base_line_array, leading_span_a_array, leading_span_b_array])

    return cloud


def money_flow_index(highs, lows, closes, volumes, period=14):
    """ MFI """
    typical_prices = (highs + lows + closes)/3

    array_length = len(closes)
    raw_mf_up_array = np.empty(array_length)
    raw_mf_down_array = np.empty(array_length)
    mfi_array = np.empty(array_length)

    mfi_array[:period+1] = np.NaN

    for day in range(1,array_length):
        if typical_prices[day] > typical_prices[day-1]:
            raw_mf_up_array[day] = typical_prices[day]*volumes[day]
        else:
            raw_mf_down_array[day] = typical_prices[day]*volumes[day]

    for day in range(period+1,array_length):
        mf_ratio = np.sum(raw_mf_up_array[day-period+1:day+1])/np.sum(raw_mf_down_array[day-period+1:day+1])
        mfi = 100-(100/(1+mf_ratio))
        mfi_array[day] = mfi

    return mfi_array


def moving_average_convergence_divergence(prices, multiplier_numerator=2):
    """ MACD """

    array_length = len(prices)
    macd_array = np.empty(array_length)

    macd_array[:26] = np.NaN

    ema12 = np.mean(prices[14:26])
    ema12_multiplier = multiplier_numerator/(12+1)
    ema26 = np.mean(prices[0:26])
    ema26_multiplier = multiplier_numerator/(26+1)

    for day in range(26, array_length):
        ema12 = prices[day]*ema12_multiplier + ema12*(1-ema12_multiplier)
        ema26 = prices[day]*ema26_multiplier + ema26*(1-ema26_multiplier)
        macd_array[day] = ema12 - ema26

    return macd_array


def relative_strength_index(prices, period=14):
    """ Calculate RSI and add it to the dataset """

    array_length = len(prices)
    change_array = np.empty(array_length)
    rsi_array = np.empty(array_length)
    rsi_array[:period] = np.NaN

    for day in range(1,array_length):
        change_array[day] = prices[day]-prices[day-1]

    changes_up = np.where(change_array < 0, 0, change_array)
    changes_down = np.where(change_array > 0, 0, abs(change_array))

    average_up = np.sum(changes_up[1:period+1])/period
    average_down = np.sum(changes_down[1:period+1])/period
    rs = average_up/average_down
    rsi = 100-(100/(1+rs))
    rsi_array[period] = rsi

    for day in range(period+1, array_length):
        average_up = (average_up*(period-1)+changes_up[day])/period
        average_down = (average_down*(period-1)+changes_down[day])/period
        rs = average_up/average_down
        rsi = 100-(100/(1+rs))
        rsi_array[day] = rsi

    return rsi_array


def simple_moving_average(prices, period=50):
    """ SMA """

    array_length = len(prices)
    sma_array = np.empty(array_length)

    sma_array[:period] = np.NaN

    for day in range(period-1,array_length):
        sma = np.mean(prices[day-period+1:day+1])
        sma_array[day] = sma

    return sma_array


def stochastic_oscillator(highs, lows, closes, period=14):
    """ Stochastic Oscillator """

    array_length = len(closes)
    stoch_array = np.empty(len(closes))
    stoch_array[:period] = np.NaN

    for day in range(period, array_length):
        close = closes[day]
        low_avg = np.min(lows[day-13:day+1])
        high_avg = np.max(highs[day-13:day+1])
        stoch = 100*(close-low_avg)/(high_avg-low_avg)
        stoch_array[day] = stoch

    return stoch_array


def swing_index(opens, highs, lows, closes, T=300):
    """ SI """

    array_length = len(closes)
    si_array = np.empty(array_length)
    si = 0

    for day in range(1,array_length):
        C, Cy = closes[day], closes[day-1]
        H, Hy = highs[day], highs[day-1]
        L, Ly = lows[day], lows[day-1]
        O, Oy = opens[day], opens[day-1]
        K = np.max((H-Cy, Cy-L))
        TR = np.max((H-Cy,Cy-L,H-L))
        if Cy > H:
            ER = abs(H-Cy)
        elif L <= Cy <= H:
            ER = 0
        else:
            ER = abs(Cy-L)
        SH = abs(Cy-Oy)

        R = TR - (0.5*ER) + (0.25*SH)
        si_test = 50*((C-Cy + ((C-O)/2) + ((Cy-Oy)/4))/R)*(K/T)
        if si_test < 1000:
            si = 50*((C-Cy + ((C-O)/2) + ((Cy-Oy)/4))/R)*(K/T)
        si_array[day] = si

        return si_array


def weighted_moving_average(prices, period=50):
    """ WMA """

    array_length = len(prices)
    wma_array = np.empty(array_length)
    denominator = period*(period+1)/2

    wma_array[:period] = np.NaN

    for day in range(period, array_length):
        numerator = 0
        for timestep in range(period):
            numerator += prices[day-timestep]*(period-timestep)
        wma_array[day] = numerator/denominator

    return wma_array
