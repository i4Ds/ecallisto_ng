import numpy as np
import pandas as pd
from skimage import filters


def elimwrongchannels(
    df, channel_std_mult=5, jump_std_mult=2, nan_interpolation_method="pchip"
):
    """
    Remove Radio Frequency Interference (RFI) from a spectrogram represented by a pandas DataFrame.
    This function works even when there is missing data thanks to interpolation of missing values.
    However, it could lead to some false or different results.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame where the index represents time and the columns represent frequency channels.
    channel_std_mult : float, optional
        Multiplicative factor for the standard deviation threshold used in the first RFI elimination step.
        Channels with standard deviation less than this threshold times the mean standard deviation across all channels are retained.
        Default is 5.
    jump_std_mult : float, optional
        Multiplicative factor for the standard deviation threshold used in the second RFI elimination step which deals with sharp jumps between channels.
        Channels with the absolute difference from the mean value less than this threshold times the standard deviation of differences are retained.
        Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with RFI removed. The DataFrame is oriented in the same way as the input DataFrame (time on index and frequency on columns).

    """
    df = df.copy()

    # Store original NaN positions
    nan_positions = df.isna()

    # Fill missing data with interpolation
    df.interpolate(method=nan_interpolation_method, inplace=True)
    df.fillna(
        method="bfill", inplace=True
    )  # for cases where NaNs are at the start of a series

    # Transpose df so that rows represent channels and columns represent time
    df = df.T

    # Calculate standard deviation for each channel and scale it to 0-255
    std = df.std(axis=1).fillna(0)
    std = ((std - std.min()) * 255) / (std.max() - std.min())
    std = std.clip(upper=255).astype(int)

    mean_sigma = std.mean()
    positions = std < channel_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        df = df[positions]

    print("Eliminating sharp jumps between channels ...")
    y_profile = np.average(filters.roberts(df.values.astype(float)), axis=1)
    y_profile = pd.Series(y_profile - y_profile.mean(), index=df.index)
    mean_sigma = y_profile.std()

    positions = np.abs(y_profile) < jump_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        df = df[positions]
    else:
        print("Sorry, all channels are bad ...")
        df = pd.DataFrame()

    # Transpose df back to original orientation
    df = df.T

    # Bring back original NaN values
    df[nan_positions] = np.nan

    return df


def subtract_constant_background(df, n=30):
    """
    Subtract a constant background from a spectrogram represented by a pandas DataFrame.

    The constant background is defined as the median value of the first n rows (timepoints) of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame where the index represents time and the columns represent frequency channels.
    n : int
        Number of first rows from which the median value is calculated to define the constant background.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the constant background subtracted. The DataFrame is oriented in the same way as the input DataFrame (time on index and frequency on columns).

    """
    df = df.copy()
    return df - df.iloc[0:n].median()


def subtract_rolling_background(
    df, window=30, center=False, how="quantile", quantile_value=0.05, **kwargs
):
    """
    Subtract a rolling background from a spectrogram represented by a pandas DataFrame.

    The rolling background is calculated either as the median or a specific quantile value of each rolling window of size `window`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame where the index represents time and the columns represent frequency channels.
    window : int, default 30
        Size of the rolling window from which the background value is calculated.
    center : bool, default False
        If True, the rolling window is centered on the current timepoint. If False, the rolling window starts at the current timepoint.
        See pandas.DataFrame.rolling for more details.
    how : str, default "median"
        Method to calculate the rolling background. If "median", the median value of the window is used.
        If "quantile", the quantile defined by `quantile_value` is used.
    quantile_value : float, default 0.5
        The quantile value to use when `how` is "quantile". Ignored if `how` is not "quantile".
    **kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.rolling.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the rolling background subtracted. The DataFrame is oriented in the same way as the input DataFrame (time on index and frequency on columns).

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
