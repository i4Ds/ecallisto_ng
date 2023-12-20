from datetime import datetime

from ecallisto_ng.data_download.downloader import (
    get_ecallisto_data,
    get_ecallisto_data_generator,
    get_instrument_with_available_data,
)
from ecallisto_ng.data_processing.utils import (
    elimwrongchannels,
    subtract_constant_background,
)
from ecallisto_ng.plotting.plotting import plot_spectogram


def test_single_plot_spectrogram():
    start_datetime = datetime(2021, 5, 7, 3, 00, 0)
    end_datetime = datetime(2021, 5, 7, 3, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    dfs = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="ASSA", verbose=True
    )

    plot_spectogram(dfs["Australia-ASSA_01"])


def test_mult_plot_spectrogram():
    start_datetime = datetime(2021, 5, 7, 3, 00, 0)
    end_datetime = datetime(2021, 5, 7, 3, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    dfs = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="ASSA", verbose=True
    )

    plot_spectogram(dfs)
