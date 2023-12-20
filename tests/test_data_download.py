from datetime import datetime

from ecallisto_ng.data_download.downloader import (
    get_ecallisto_data,
    get_instrument_with_available_data,
)


def test_get_avail_instruments():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)

    avail = get_instrument_with_available_data(start_datetime, end_datetime)
    assert len(avail) > 0
    # Check some instruments
    [
        "AUSTRIA-Krumbach_10",
        "AUSTRIA-MICHELBACH_60",
        "AUSTRIA-OE3FLB_55",
        "AUSTRIA-OE3FLB_57",
        "AUSTRIA-UNIGRAZ_01",
        "AUSTRIA-UNIGRAZ_02",
        "Australia-ASSA_56",
        "Australia-ASSA_57",
        "Australia-ASSA_60",
        "Australia-ASSA_62",
        "Australia-ASSA_63",
        "Australia-LMRO_59",
        "INDIA-GAURI_01",
        "INDIA-OOTY_01",
        "INDIA-OOTY_02",
        "JAPAN-IBARAKI_59",
    ]
    for instrument in avail:
        assert instrument in avail


def test_get_data_multiple():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    dfs = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="ASSA", verbose=True
    )

    assert all(
        [
            x in ["Australia-ASSA_01", "Australia-ASSA_02", "Australia-ASSA_60"]
            for x in dfs.keys()
        ]
    )
    assert dfs["Australia-ASSA_01"].shape == (172490, 193)


def test_get_data_single():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)

    # get_ecallisto_data returns a dictionary, with the keys being the instrument names and the values being the dataframes
    dfs = get_ecallisto_data(
        start_datetime, end_datetime, instrument_name="Australia-ASSA_01", verbose=True
    )

    assert all([x in ["Australia-ASSA_01"] for x in dfs.keys()])
    assert len(dfs) == 1
    assert dfs["Australia-ASSA_01"].shape == (172490, 193)
