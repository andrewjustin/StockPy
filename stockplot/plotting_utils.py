""" Tools for plotting """

import numpy as np


def shift_indicator(indicator, shift):
    """
    Adds NaNs to the beginning of an indicator array if it is shorter than the number of days being plotted.
    This will result in a shifted line on the plot.
    """

    nan_array = np.empty([shift])
    nan_array[:] = np.NaN

    shifted_indicator = np.append(nan_array, indicator)

    return shifted_indicator
