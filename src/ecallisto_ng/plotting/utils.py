import numpy as np
import pandas as pd


def return_strftime_based_on_range(time_range):
    # Decide on the date-time format based on the time range
    if time_range < pd.Timedelta(days=1):
        date_format = "%H:%M:%S"
    elif time_range < pd.Timedelta(weeks=4):
        date_format = "%Y-%m-%d %H:%M"
    else:
        date_format = "%Y-%m-%d"

    return date_format


def timedelta_to_sql_timebucket_value(timedelta):
    # Convert to seconds
    seconds = timedelta.total_seconds()

    # Convert to SQL-compatible value
    if seconds >= 86400:  # More than 1 day
        days = seconds / 86400
        sql_value = f"{int(days)} d" if days.is_integer() else f"{days:.1f} d"
    elif seconds >= 3600:  # More than 1 hour
        hours = seconds / 3600
        sql_value = f"{int(hours)} h" if hours.is_integer() else f"{hours:.1f} h"
    elif seconds >= 60:  # More than 1 minute
        minutes = seconds / 60
        sql_value = (
            f"{int(minutes)} min" if minutes.is_integer() else f"{minutes:.1f} min"
        )
    else:
        sql_value = f"{seconds:.1f} s"

    return sql_value


def fill_missing_timesteps_with_nan(df):
    """
    Fill missing timesteps in a pandas DataFrame with NaN values. Only needed for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to fill missing timesteps in.

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
    time_delta = np.median(np.diff(df.index.values))
    time_delta = pd.Timedelta(time_delta)
    new_index = pd.date_range(df.index[0], df.index[-1], freq=time_delta)
    df = df.reindex(new_index)
    return df
