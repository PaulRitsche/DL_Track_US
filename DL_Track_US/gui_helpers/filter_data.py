"""Module to filter the data subsequent to the analyses functions."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter


def applyFilters(data, filter_type="median", **kwargs):
    """
    Applies a selected filter to the input data.

    Parameters
    ----------
    data : list of lists
        The input data to be filtered.
    filter_type : str, optional
        The type of filter to apply. Options are "median", "gaussian", "savitzky_golay".
    kwargs : dict
        Additional parameters for the selected filter.

    Returns
    -------
    list of lists
        The filtered data.
    """
    filtered_data = []

    for frame in data:
        if filter_type == "median":
            kernel_size = kwargs.get("kernel_size", 3)
            filtered_data.append(medfilt(frame, kernel_size=kernel_size).tolist())

        elif filter_type == "gaussian":
            sigma = kwargs.get("sigma", 1)
            filtered_data.append(gaussian_filter1d(frame, sigma=sigma).tolist())

        elif filter_type == "savitzky_golay":
            window_length = kwargs.get("window_length", 5)
            polyorder = kwargs.get("polyorder", 2)
            if len(frame) >= window_length:
                filtered_data.append(
                    savgol_filter(frame, window_length, polyorder).tolist()
                )
            else:
                filtered_data.append(frame)  # Keep raw if too short

        else:
            raise ValueError(
                "Unsupported filter type. Choose from 'median', 'gaussian', or 'savitzky_golay'."
            )

    return filtered_data


# taken from https://github.com/dwervin/pyhampel/blob/main/pyhampel/src/HampelFiltering.py


def hampelFilterList(data: list, win_size=5, num_dev=1, center_win=True):
    """
    Applies a Hampel filter to a list of numerical values.

    This function detects and replaces outliers with the rolling median
    within a specified window size.

    Parameters
    ----------
    data : list
        List of numerical values.
    win_size : int, optional
        Window size for rolling median filtering. Default is 5.
    num_dev : int, optional
        Number of standard deviations for outlier detection. Default is 1.
    center_win : bool, optional
        Whether the window is centered on the point. Default is True.

    Returns
    -------
    dict
        A dictionary with:
        - "filtered" : list with filtered values.
        - "outliers" : list where outliers are replaced with NaN.
        - "is_outlier" : list of boolean values indicating outliers.

    Examples
    --------
    >>> data = [10, 12, 100, 14, 13, 15, 300, 18, 16]
    >>> result = hampel_filter_list(data, win_size=3, num_dev=2)
    >>> print(result["filtered"])
    [10, 12, 13, 14, 13, 15, 16, 18, 16]
    >>> print(result["is_outlier"])
    [False, False, True, False, False, False, True, False, False]
    """

    data = np.array(data, dtype=np.float64)
    n = len(data)

    if n < win_size:
        return {
            "filtered": data.tolist(),
            "outliers": np.full(n, np.nan).tolist(),
            "is_outlier": [False] * n,
        }

    L = 0.5  # Scaling factor for standard deviation estimation

    # Rolling median
    rolling_median = (
        pd.Series(data)
        .rolling(window=win_size, center=center_win, min_periods=1)
        .median()
    )

    # Median Absolute Deviation (MAD)
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = (
        pd.Series(data)
        .rolling(window=win_size, center=center_win, min_periods=1)
        .apply(MAD)
    )

    # Define threshold for outlier detection
    threshold = num_dev * L * rolling_MAD

    # Detect outliers
    difference = np.abs(data - rolling_median)
    outlier_idx = difference > threshold

    # Replace outliers with rolling median
    filtered_data = data.copy()
    filtered_data[outlier_idx] = rolling_median[outlier_idx]

    # Create list of outliers (replacing non-outliers with NaN)
    outlier_values = np.full(n, np.nan)
    outlier_values[outlier_idx] = data[outlier_idx]

    return {
        "filtered": filtered_data.tolist(),
        "outliers": outlier_values.tolist(),
        "is_outlier": outlier_idx.tolist(),
    }
