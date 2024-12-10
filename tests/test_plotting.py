from datetime import datetime

import numpy as np

from ecallisto_ng.data_download.downloader import get_ecallisto_data
from ecallisto_ng.plotting.plotting import (
    plot_spectrogram,
    plot_with_fixed_resolution_mpl,
)


def test_single_plot_spectrogram(assa01_dataframe):
    fig = plot_spectrogram(assa01_dataframe)

    # Check the layout
    layout = fig.layout
    assert "Australia-ASSA_01" in layout.title.text
    assert layout.xaxis.title.text == "Time [UT]"
    assert layout.yaxis.title.text == "Frequency [MHz]"

    # Check the data
    data = fig.data[0]
    assert np.nanmin(data.z) >= 0
    assert np.nanmax(data.z) <= 255


def test_mult_plot_spectrogram(assa_dataframes):
    plot_spectrogram(assa_dataframes)


def test_fixed_resolution_plot_spectrogram():
    start_datetime = datetime(2021, 5, 7, 3, 00, 0)
    end_datetime = datetime(2021, 5, 7, 3, 59, 0)

    instru = "Australia-ASSA_01"
    fig = plot_with_fixed_resolution_mpl(
        "Australia-ASSA_01", start_datetime, end_datetime
    )

    assert len(fig.axes) == 2
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time [UT]"
    assert ax.get_ylabel() == "Frequency [MHz]"
    assert instru in ax.get_title()
    imshow_obj = ax.images[0]
    data = imshow_obj.get_array()

    assert data.shape[0] > 0
    assert data.shape[1] > 0
    assert data.min() >= 0
    assert data.max() <= 255
