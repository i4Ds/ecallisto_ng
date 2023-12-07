import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from ecallisto_ng.data_download.downloader import get_ecallisto_data
from ecallisto_ng.data_fetching.get_data import NoDataAvailable
from ecallisto_ng.plotting.utils import (
    calculate_resample_freq,
    return_strftime_based_on_range,
    return_strftime_for_ticks_based_on_range,
)


def plot_spectogram_mpl(
    df,
    instrument_name=None,
    start_datetime=None,
    end_datetime=None,
    title="Radio Flux Density",
    fig_size=(9, 6),
    cmap="plasma",
):
    # Create a new dataframe with rounded column names
    df = df.copy()
    # Check if it's a dictionary. If it is, take the first key
    if isinstance(df, dict):
        if len(df) > 1:
            Warning(
                "The dataframe has more than one instrument. Only the first instrument will be used."
            )
        df = df[list(df.keys())[0]]

    # Drop any rows where the datetime col is NaN
    df = df[df.index.notnull()]

    # Reverse the columns
    df = df.iloc[:, ::-1]

    # If instrument name is not provided, try to get it from the dataframe
    if instrument_name is None:
        instrument_name = df.attrs.get("FULLNAME", "Unknown")

    # If start_datetime is not provided, try to get it from the dataframe
    if start_datetime is None:
        start_datetime = df.index.min()
    if end_datetime is None:
        end_datetime = df.index.max()

    # Make datetime prettier
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    strf_format = return_strftime_based_on_range(end_datetime - start_datetime)
    strf_format_ticks = return_strftime_for_ticks_based_on_range(
        end_datetime - start_datetime
    )
    sd_str = start_datetime.strftime(strf_format)
    ed_str = end_datetime.strftime(strf_format)

    fig, ax = plt.subplots(figsize=fig_size)

    # Set NaN color to black
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="black")

    # The imshow function in matplotlib displays data top-down, so we need to reverse the rows
    cax = ax.imshow(
        df.T.iloc[::-1],
        aspect="auto",
        extent=[0, df.shape[0], 0, df.shape[1]],
        cmap=current_cmap,
        interpolation="none",
    )

    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    # Calculate the rough spacing for around 15 labels
    spacing = max(1, int(df.shape[1] / 15))

    # Create target ticks
    target_ticks = np.unique((df.columns.astype(float) / 10).astype(int) * 10)

    # Finding the closest indices in the DataFrame to the target_ticks
    major_ticks = [
        find_nearest_idx(df.columns.astype(float), tick) for tick in target_ticks
    ]

    # Set major ticks and their appearance
    ax.set_yticks(major_ticks, minor=False)  # This line was missing
    ax.tick_params(axis="y", which="major", length=10, labelsize="medium")

    # Create labels based on the position
    major_labels = [str(int(round(float(df.columns[i]), 0))) for i in major_ticks]
    ax.set_yticklabels(major_labels, minor=False)

    # Assuming df index is datetime, this will format the x-ticks
    # Compute the spacing required to get close to 30 x-labels
    spacing = max(1, df.shape[0] // 15)

    x_ticks = np.arange(0, df.shape[0], spacing)
    ax.set_xticks(x_ticks)
    # Get format
    strf_format_ticks = return_strftime_for_ticks_based_on_range(
        end_datetime - start_datetime
    )
    ax.set_xticklabels(
        df.index[x_ticks].strftime(strf_format_ticks), rotation=60, ha="center"
    )
    # Title
    title = f"{instrument_name} {title} | {sd_str} to {ed_str}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time [UT]")
    ax.set_ylabel("Frequency [MHz]")
    ax.grid(False)

    # Adding colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label("Amplitude")

    fig.tight_layout()
    return fig


def plot_spectogram(
    df,
    instrument_name=None,
    start_datetime=None,
    end_datetime=None,
    title="Radio Flux Density",
    save_path=None,
    resolution=1440,
    samplig_method="max",
    font_size=18,
    fig_size=(600, 1000),
    color_scale=px.colors.sequential.Plasma,
):
    # Check if it's a dictionary. If it is, take the first key
    if isinstance(df, dict):
        if len(df) > 1:
            Warning(
                "The dataframe has more than one instrument. Only the first instrument will be used."
            )
        df = df[list(df.keys())[0]]
    # Create a new dataframe with rounded column names
    df = df.copy()
    df.columns = df.columns.astype(float)

    # If instrument name is not provided, try to get it from the dataframe
    if instrument_name is None:
        instrument_name = df.attrs.get("FULLNAME", "Unknown")

    # If start_datetime is not provided, try to get it from the dataframe
    if start_datetime is None:
        start_datetime = df.index.min()
    if end_datetime is None:
        end_datetime = df.index.max()

    # Make datetime prettier
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    # If resolution is provided, resample the dataframe
    if resolution is not None:
        resample_freq = calculate_resample_freq(
            start_datetime, end_datetime, resolution
        )
        resample_freq = max(resample_freq, pd.Timedelta(milliseconds=250))

        # Resample data
        if samplig_method == "mean":
            df = df.resample(resample_freq).mean()
        elif samplig_method == "max":
            df = df.resample(resample_freq).max()
        elif samplig_method == "min":
            df = df.resample(resample_freq).min()

    fig = px.imshow(
        df.T,
        color_continuous_scale=color_scale,
        zmin=df.min().min(),
        zmax=df.max().max(),
        height=fig_size[0],
        width=fig_size[1],
    )
    fig.update_layout(
        title=f"{instrument_name} {title}",
        xaxis_title="Datetime [UT]",
        yaxis_title="Frequency [MHz]",
        font=dict(family="Computer Modern, monospace", size=font_size, color="#4D4D4D"),
        plot_bgcolor="black",
        xaxis_showgrid=True,
        yaxis_showgrid=False,
    )
    if save_path is not None:
        pio.write_image(fig, save_path)  # Save the figure
    return fig


def plot_with_fixed_resolution_mpl(
    instrument,
    start_datetime_str,
    end_datetime_str,
    sampling_method="max",
    download_from_local=False,
    resolution=1440,
    fig_size=(9, 6),
    verbose=False,
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
    - resolution (int, optional): The desired resolution for plotting. Default is 1440.
        Determines the time bucketing for the data aggregation.
    - fig_size (tuple, optional): The desired figure size. Default is (9, 6).
        The figure size is passed to Matplotlib's `figsize` parameter.

    Returns:
    None. A spectrogram is plotted using Matplotlib.
    """
    start_datetime = pd.to_datetime(start_datetime_str)
    end_datetime = pd.to_datetime(end_datetime_str)

    # Fetch data
    df = get_ecallisto_data(
        start_datetime,
        end_datetime,
        instrument,
        download_from_local=download_from_local,
        verbose=verbose,
    )
    # Check if it's a dictionary. If it is, take the first key
    if isinstance(df, dict):
        if len(df) > 1:
            Warning(
                "The dataframe has more than one instrument. Only the first instrument will be used."
            )
        df = df[list(df.keys())[0]]

    if len(df) == 0:
        print(NoDataAvailable)
        return None

    if resolution is not None:
        # Calculate resampling frequency
        resample_freq = calculate_resample_freq(
            start_datetime, end_datetime, resolution
        )
        resample_freq = max(resample_freq, pd.Timedelta(milliseconds=250))
        # Resample data
        if sampling_method.lower() == "mean":
            df = df.resample(resample_freq).mean()
        elif sampling_method.lower() == "max":
            df = df.resample(resample_freq).max()
        elif sampling_method.lower() == "min":
            df = df.resample(resample_freq).min()

    # Plot
    return plot_spectogram_mpl(
        df, instrument, start_datetime, end_datetime, fig_size=fig_size
    )
