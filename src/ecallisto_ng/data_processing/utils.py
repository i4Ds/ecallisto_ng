from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.signal
from scipy.ndimage import generic_filter, median_filter
from skimage import filters


def calculate_snr(data: pd.DataFrame, window: int = 5) -> np.float64:
    data = subtract_rolling_background(data, window)
    data = data.dropna(axis=0)
    return np.round(np.mean(data) / np.std(data), 3)


def min_max_scale_per_column(data: pd.DataFrame) -> pd.DataFrame:
    """Apply min-max scaling to each column of a DataFrame.

    Parameters: data (pd.DataFrame): The input data with columns representing
    different frequencies.

    Returns: pd.DataFrame: The scaled data where each column is scaled
    independently.
    """
    # Ensuring the data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Apply min-max scaling per column
    scaled_data = (data - data.min()) / (data.max() - data.min())

    return scaled_data


def apply_quantile_filter(
    df: pd.DataFrame, quantile_value: float, size: tuple[int, int] = (3, 3)
) -> pd.DataFrame:
    """Apply quantile filter to a DataFrame.

    Parameters ---------- df : pd.DataFrame     Input DataFrame to filter.
    quantile_value : float     Quantile value to use for filtering (between 0
    and 1). size : tuple of int, optional     Dimensions of the filter kernel.
    Default is (3, 3).

    Returns ------- pd.DataFrame     Filtered DataFrame.
    """

    def quantile_func(values: np.ndarray) -> np.floating:
        return np.quantile(values, quantile_value)

    data = generic_filter(df.values, quantile_func, size=size, mode="reflect")
    df.values[:] = data
    return df


def mean_filter(
    df: pd.DataFrame, kernel_size: tuple[int, int] = (5, 5)
) -> pd.DataFrame:
    """Apply mean filter to a DataFrame using a 2D convolution.

    Parameters ---------- df : pd.DataFrame     Input DataFrame to filter.
    kernel_size : tuple of int, optional     Dimensions of the filter kernel.
    Default is (5, 5).

    Returns ------- pd.DataFrame     Filtered DataFrame.
    """
    kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])
    data = scipy.signal.convolve2d(df.to_numpy(), kernel, "same")
    df.values[:] = data
    return df


def apply_median_filter(
    df: pd.DataFrame, size: tuple[int, int] = (3, 3)
) -> pd.DataFrame:
    """Apply median filter to a DataFrame.

    Parameters ---------- df : pd.DataFrame     Input DataFrame to filter. size
    : tuple of int, optional     Dimensions of the filter kernel. Default is
    (3, 3).

    Returns ------- pd.DataFrame     Filtered DataFrame.
    """
    data = median_filter(df.values, size)
    df.values[:] = data
    return df


def return_strftime_based_on_range(time_range: pd.Timedelta) -> str:
    # Decide on the date-time format based on the time range
    if time_range < pd.Timedelta(days=1):
        date_format = "%H:%M:%S"
    elif time_range < pd.Timedelta(weeks=4):
        date_format = "%Y-%m-%d %H:%M"
    else:
        date_format = "%Y-%m-%d"

    return date_format


def elimwrongchannels(
    df: pd.DataFrame,
    channel_std_mult: float = 5,
    jump_std_mult: float = 2,
    nan_interpolation_method: str = "pchip",
    interpolate_created_nans: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Remove Radio Frequency Interference (RFI) from a spectrogram represented
    by a pandas DataFrame. This function works even when there is missing data
    thanks to interpolation of missing values. However, it could lead to some
    false or different results.

    Parameters ---------- df : pandas.DataFrame     Input DataFrame where the
    index represents time and the columns represent frequency channels.
    channel_std_mult : float, optional     Multiplicative factor for the
    standard deviation threshold used in the first RFI elimination step.
    Channels with standard deviation less than this threshold times the mean
    standard deviation across all channels are retained.     Default is 5.
    jump_std_mult : float, optional     Multiplicative factor for the standard
    deviation threshold used in the second RFI elimination step which deals
    with sharp jumps between channels.     Channels with the absolute
    difference from the mean value less than this threshold times the standard
    deviation of differences are retained.     Default is 2.
    nan_interpolation_method : str, optional     Interpolation method to use
    for missing values. See pandas.DataFrame.interpolate for more details.
    Default is "pchip". interpolate_created_nans : bool, optional     Whether
    to interpolate NaNs created by the first RFI elimination step.     Default
    is True. verbose : bool, optional     Whether to print out the number of
    eliminated channels.

    Returns ------- pandas.DataFrame     DataFrame with RFI removed. The
    DataFrame is oriented in the same way as the input DataFrame (time on index
    and frequency on columns).
    """
    df = df.copy()

    # Store original NaN positions
    nan_positions = df.isna()

    # Fill missing data with interpolation
    df.interpolate(method=nan_interpolation_method, inplace=True)
    df.bfill(inplace=True)  # for cases where NaNs are at the start of a series

    # Transpose df so that rows represent channels and columns represent time
    df = df.T

    # Calculate standard deviation for each channel and scale it to 0-255
    std = df.std(axis=1).fillna(0)
    std = ((std - std.min()) * 255) / (std.max() - std.min())
    std = std.clip(upper=255).astype(int)

    mean_sigma = std.mean()
    positions = std < channel_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    if verbose:
        print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        # Replace the line with nans
        df.iloc[~positions, :] = np.nan

    if interpolate_created_nans:
        # Interpolate the nans
        df = df.interpolate(axis=0, limit_direction="both")

    if verbose:
        print("Eliminating sharp jumps between channels ...")
    y_profile = np.average(filters.roberts(df.values.astype(float)), axis=1)
    y_profile = pd.Series(y_profile - y_profile.mean(), index=df.index)
    mean_sigma = y_profile.std()

    positions = np.abs(y_profile) < jump_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    if verbose:
        print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        # Replace the line with nans
        df.iloc[~positions, :] = np.nan
    else:
        if verbose:
            print("Sorry, all channels are bad ...")
        df = pd.DataFrame()
    if interpolate_created_nans:
        # Interpolate the nans
        df = df.interpolate(axis=0, limit_direction="both")
    # Transpose df back to original orientation
    df = df.T

    # Drop nans
    df.dropna(inplace=True)

    # Bring back original NaN values
    df[nan_positions] = np.nan

    return df


def subtract_constant_background(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Subtract a constant background from a spectrogram represented by a
    pandas DataFrame.

    The constant background is defined as the median value of the first n rows
    (timepoints) of the DataFrame.

    Parameters ---------- df : pandas.DataFrame     Input DataFrame where the
    index represents time and the columns represent frequency channels. n : int
    Number of first rows from which the median value is calculated to define
    the constant background.

    Returns ------- pandas.DataFrame     DataFrame with the constant background
    subtracted. The DataFrame is oriented in the same way as the input
    DataFrame (time on index and frequency on columns).
    """
    df = df.copy()
    return df - df.iloc[0:n].median()


def subtract_rolling_background(
    df: pd.DataFrame,
    window: int = 30,
    center: bool = False,
    how: Literal["median", "quantile"] = "quantile",
    quantile_value: float = 0.05,
    **kwargs: Any,
) -> pd.DataFrame:
    """Subtract a rolling background from a spectrogram represented by a pandas
    DataFrame.

    The rolling background is calculated either as the median or a specific
    quantile value of each rolling window of size `window`.

    Parameters ---------- df : pandas.DataFrame     Input DataFrame where the
    index represents time and the columns represent frequency channels. window
    : int, default 30     Size of the rolling window from which the background
    value is calculated. center : bool, default False     If True, the rolling
    window is centered on the current timepoint. If False, the rolling window
    starts at the current timepoint.     See pandas.DataFrame.rolling for more
    details. how : str, default "quantile"     Method to calculate the rolling
    background. If "median", the median value of the window is used.     If
    "quantile", the quantile defined by `quantile_value` is used.
    quantile_value : float, default 0.5     The quantile value to use when
    `how` is "quantile". Ignored if `how` is not "quantile". **kwargs : dict
    Additional keyword arguments passed to pandas.DataFrame.rolling.

    Returns ------- pandas.DataFrame     DataFrame with the rolling background
    subtracted. The DataFrame is oriented in the same way as the input
    DataFrame (time on index and frequency on columns).
    """
    df = df.copy()
    if how == "median":
        df_rolling = df.rolling(window=window, center=center, **kwargs).median()
    elif how == "quantile":
        df_rolling = df.rolling(window=window, center=center, **kwargs).quantile(
            quantile_value
        )
    else:
        raise ValueError("`how` must be 'median' or 'quantile'")
    return df - df_rolling


def subtract_low_signal_noise_background(
    df: pd.DataFrame, percentile: float = 0.05
) -> pd.DataFrame:
    """Constant background subtraction from the quietest time samples.

    DataFrame port of the IDL / radiospectra ``constbacksub`` algorithm:

    1. Subtract each frequency channel's mean over time.
    2. For each time sample, compute the standard deviation across frequency
       of the mean-subtracted data.
    3. Select the ``int(n_times * percentile) + 1`` time samples with the
       lowest standard deviation (the "quiet" part of the observation).
    4. The background is the per-channel mean of the mean-subtracted data over
       those selected time samples.
    5. Return the mean-subtracted data minus the background, which equals the
       original data minus the per-channel mean over the selected samples.

    The ``+ 1`` in step 3 is kept for exact parity with the original
    implementation; it also guarantees at least one sample is selected.

    NaNs follow pandas semantics: means and standard deviations skip NaNs,
    time samples whose standard deviation is NaN are never selected, and NaN
    entries stay NaN in the output.

    Parameters
    ----------
    df : DataFrame
        DataFrame representing the spectrogram with time as index and
        frequencies as columns.
    percentile : float, default 0.05
        Fraction of time samples with the lowest standard deviation across
        frequency to use as background.

    Returns
    -------
    DataFrame
        Background-subtracted DataFrame with the same index and columns.
    """
    # Subtract the average of each column (frequency channel) over time
    df_ = df - df.mean(axis=0)

    # Standard deviation across frequency for each row (time sample);
    # ddof=0 as in numpy.std used by the original implementation
    time_sdevs = df_.std(axis=1, ddof=0)

    # Select the time samples with the lowest standard deviations (assumed
    # background). Work with positions so duplicate timestamps are harmless.
    n_background = int(len(time_sdevs) * percentile) + 1
    quiet_rows = time_sdevs.reset_index(drop=True).nsmallest(n_background).index
    background = df_.iloc[quiet_rows].mean(axis=0)

    # Subtract this background from each time sample of the DataFrame
    return df_ - background


def intensity_to_linear(df: pd.DataFrame, factor: float = 0.0386) -> pd.DataFrame:
    """
    Convert Callisto values (W) and make them linear.
    Based on the following forumla:
    db = 10 ** (I * factor)

    Parameters:
    df (pd.DataFrame): DataFrame with Callisto values.
    factor (float): Conversion factor in the formula.

    Returns:
    pd.DataFrame: DataFrame with converted intensity values.
    """
    return 10 ** (df * factor)


def linear_to_intensity(df: pd.DataFrame, factor: float = 0.0386) -> pd.DataFrame:
    """
    Convert db (I) back to Callisto values (W).
    Based on the following forumla:
    db = 10 ** (I * factor)

    Parameters:
    df (pd.DataFrame): DataFrame with intensity values.
    factor (float): Conversion factor in the formula.

    Returns:
    pd.DataFrame: DataFrame with converted Callisto values.
    """
    return np.log10(df) / factor
