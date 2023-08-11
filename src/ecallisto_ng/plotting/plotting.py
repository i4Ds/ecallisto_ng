import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from ecallisto_ng.data_fetching.get_data import get_data
from ecallisto_ng.plotting.utils import (
    fill_missing_timesteps_with_nan,
    return_strftime_based_on_range,
    timedelta_to_sql_timebucket_value,
)


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


def plot_spectogram_mpl(
    df,
    instrument_name,
    start_datetime,
    end_datetime,
    cmap="plasma",
):
    # Create a new dataframe with rounded column names
    df = df.copy()

    # Make datetime prettier
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    strf_format = return_strftime_based_on_range(end_datetime - start_datetime)
    sd_str = start_datetime.strftime(strf_format)
    ed_str = end_datetime.strftime(strf_format)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set NaN color to black
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="black")

    # The imshow function in matplotlib displays data top-down, so we need to reverse the rows
    cax = ax.imshow(
        df.T.iloc[::-1],
        aspect="auto",
        extent=[0, df.shape[0], 0, df.shape[1]],
        cmap=current_cmap,
    )

    # Calculate the rough spacing for around 15 labels
    spacing = max(1, int(df.shape[1] / 15))

    # Create y-ticks based on the spacing
    all_ticks = np.arange(0, df.shape[1], spacing)

    # Split ticks into major and minor based on the modulo condition
    major_ticks = [i for i in all_ticks if float(df.columns[i]) % 10 == 0]
    minor_ticks = list(set(all_ticks) - set(major_ticks))

    # Set major ticks and their appearance
    ax.set_yticks(major_ticks, minor=False)
    ax.tick_params(axis="y", which="major", length=10, labelsize="medium")
    major_labels = [
        str(int(round(float(df.columns[i])))) for i in major_ticks
    ]  # Round to the nearest integer
    ax.set_yticklabels(major_labels, minor=False)

    # Set minor ticks and their appearance
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis="y", which="minor", length=5, labelsize="small")
    minor_labels = [
        str(round(float(df.columns[i]), 1)) for i in minor_ticks
    ]  # Round based on round_precision
    ax.set_yticklabels(minor_labels, minor=True)

    # Assuming df index is datetime, this will format the x-ticks
    # Compute the spacing required to get close to 30 x-labels
    spacing = max(1, df.shape[0] // 15)

    x_ticks = np.arange(0, df.shape[0], spacing)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(df.index[x_ticks].strftime(strf_format), rotation=30, ha="right")
    # Title
    ax.set_title(f"Spectogram of {instrument_name} between {sd_str} and {ed_str}")
    ax.set_xlabel("Time [UT]")
    ax.set_ylabel("Frequency [MHz]")
    ax.grid(False)

    # Adding colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label("Amplitude")

    fig.tight_layout()
    return fig


def plot_with_fixed_resolution_mpl(
    instrument, start_datetime_str, end_datetime_str, resolution=720
):
    """
    Plots the spectrogram for the given instrument between specified start and end datetime strings
    with a fixed resolution using Matplotlib.

    Parameters:
    - instrument (str): The name of the instrument for which the spectrogram needs to be plotted.
    - start_datetime_str (str or pd.Timestamp): The starting datetime for the data range.
        Can be a string in the format 'YYYY-MM-DD HH:MM:SS' or a Pandas Timestamp.
    - end_datetime_str (str or pd.Timestamp): The ending datetime for the data range.
        Can be a string in the format 'YYYY-MM-DD HH:MM:SS' or a Pandas Timestamp.
    - resolution (int, optional): The desired resolution for plotting. Default is 720.
        Determines the time bucketing for the data aggregation.

    Returns:
    None. A spectrogram is plotted using Matplotlib.

    Usage:
    plot_with_fixed_resolution_mpl('some_instrument', '2022-03-31 18:46:00', '2022-04-01 18:46:00', resolution=500)

    Note:
    The function internally calls other utility functions including:
    - timedelta_to_sql_timebucket_value() to convert the time delta to an appropriate format for SQL queries.
    - get_data() to fetch the data based on the provided parameters.
    - fill_missing_timesteps_with_nan() to handle any missing data points.
    - plot_spectogram_mpl() to generate the actual spectrogram plot.
    """

    # Make datetime prettier
    if isinstance(start_datetime_str, str):
        start_datetime = pd.to_datetime(start_datetime_str)
    if isinstance(end_datetime_str, str):
        end_datetime = pd.to_datetime(end_datetime_str)

    time_delta = (end_datetime - start_datetime) / resolution
    # Create parameter dictionary
    params = {
        "instrument_name": instrument,
        "start_datetime": start_datetime_str,
        "end_datetime": end_datetime_str,
        "timebucket": timedelta_to_sql_timebucket_value(time_delta),
        "agg_function": "MAX",
    }
    # Get data
    df = get_data(**params)
    df_filled = fill_missing_timesteps_with_nan(df)

    # Plot
    plot_spectogram_mpl(df_filled, instrument, start_datetime, end_datetime)
