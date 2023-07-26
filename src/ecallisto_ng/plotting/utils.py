import numpy as np
import pandas as pd
import plotly.express as px


def plot_spectogram(
    df,
    instrument_name,
    start_datetime,
    end_datetime,
    size=18,
    round_precision=1,
    color_scale=px.colors.sequential.Plasma,
):
    # Create a new dataframe with rounded column names
    df_rounded = df.copy()
    df_rounded.columns = [f"{float(col):.{round_precision}f}" for col in df.columns]

    # Make datetime prettier
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)
    sd_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    ed_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    fig = px.imshow(
        df_rounded.T,
        color_continuous_scale=color_scale,
        zmin=df.min().min(),
        zmax=df.max().max(),
    )
    fig.update_layout(
        title=f"Spectogram of {instrument_name} between {sd_str} and {ed_str}",
        xaxis_title="Datetime",
        yaxis_title="Frequency",
        font=dict(family="Courier New, monospace", size=size, color="#7f7f7f"),
        plot_bgcolor="black",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )
    return fig


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
