from datetime import datetime

import pytest

from ecallisto_ng.data_download.downloader import get_ecallisto_data


@pytest.fixture
def assa01_dataframe():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    df = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="Australia-ASSA_01", verbose=True
    )

    return df


@pytest.fixture
def small_assa01_dataframe():
    start_datetime = datetime(2021, 5, 7, 3, 00, 0)
    end_datetime = datetime(2021, 5, 7, 3, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    df = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="Australia-ASSA_01", verbose=True
    )

    return df


@pytest.fixture
def assa_dataframes():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    dfs = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="ASSA", verbose=True
    )

    return dfs
