import numpy as np
import pandas as pd


def calculate_resample_freq(start_datetime, end_datetime, resolution):
    tota_time_delta = end_datetime - start_datetime
    return tota_time_delta / (resolution - 1)


def return_strftime_based_on_range(time_range):
    # Decide on the date-time format based on the time range
    if time_range > pd.Timedelta(days=1):
        date_format = "%Y-%m-%d"
    elif time_range > pd.Timedelta(hours=1):
        date_format = "%Y-%m-%d %H:%M"
    else:
        date_format = "%Y-%m-%d %H:%M:%S"

    return date_format


def return_strftime_for_ticks_based_on_range(time_range):
    # Decide on the date-time format based on the time range
    if time_range < pd.Timedelta(days=1):
        date_format = "%H:%M:%S"
    elif time_range < pd.Timedelta(days=30):
        date_format = "%m-%d %H:%M"
    else:
        date_format = "%m-%d %H:%M:%S"

    return date_format


def fill_missing_timesteps_with_nan(df, start_datetime=None, end_datetime=None):
    """
    Fill missing timesteps in a pandas DataFrame with NaN values. Only needed for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to fill missing timesteps in.
    start_datetime : str or pandas.Timestamp, optional
        If you want to make sure that the returned DataFrame starts at a specific datetime,
        you can specify it here. If not specified, the returned DataFrame will start at the
        first datetime in the input DataFrame.
    end_datetime : str or pandas.Timestamp, optional
        If you want to make sure that the returned DataFrame ends at a specific datetime,
        you can specify it here. If not specified, the returned DataFrame will end at the
        last datetime in the input DataFrame.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with missing timesteps filled with NaN values.

    Notes
    -----
    This function is useful when working with time-series data that has missing timesteps.
    By filling the missing timesteps with NaN values, the DataFrame can be easily visualized
    or analyzed without introducing errors due to missing data.

    The function calculates the median time delta of the input DataFrame, and then creates
    a new index with evenly spaced values based on that delta. It then uses the pandas
    `reindex` function to fill in missing timesteps with NaN values.

    Examples
    --------
    >>> dates = pd.date_range('2023-02-19 01:00', '2023-02-19 05:00', freq='2H')
    >>> freqs = ['10M', '20M', '30M']
    >>> data = np.random.randn(len(dates), len(freqs))
    >>> df = pd.DataFrame(data, index=dates, columns=freqs)
    >>> df = df.drop(df.index[1])
    >>> df = fill_missing_timesteps_with_nan(df)
    >>> print(df)
                            10M       20M       30M
    2023-02-19 01:00:00 -0.349636  0.004947  0.546848
    2023-02-19 03:00:00       NaN       NaN       NaN
    2023-02-19 05:00:00 -0.576182  1.222293 -0.416526
    """
    # Change index to datetime, if it's not already
    df.index = pd.to_datetime(df.index)
    # Add start and end datetimes to the index
    # Fill missing timesteps with NaN values
    time_delta = np.median(np.diff(df.index.values))
    time_delta = pd.Timedelta(time_delta)
    start_datetime = df.index[0] if start_datetime is None else start_datetime
    new_index = pd.date_range(df.index[0], df.index[-1], freq=time_delta)
    # Add missing timesteps incase they are not present in the original index
    if start_datetime:
        while new_index.min() > start_datetime:
            new_index = new_index.insert(0, new_index.min() - time_delta)
    if end_datetime:
        while new_index.max() < end_datetime:
            new_index = new_index.insert(len(new_index), new_index.max() + time_delta)
    df = df.reindex(new_index)
    return df
