from datetime import datetime

from ecallisto_ng.data_download.downloader import (
    get_ecallisto_data_generator,
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


def test_get_data_multiple(assa_dataframes):
    assert len(assa_dataframes) == 3
    assert all(
        [
            x in ["Australia-ASSA_01", "Australia-ASSA_02", "Australia-ASSA_60"]
            for x in assa_dataframes.keys()
        ]
    )
    assert assa_dataframes["Australia-ASSA_01"].shape == (172490, 193)


def test_get_data_single(assa01_dataframe):
    assert len(assa01_dataframe) == 1
    assert all([x in ["Australia-ASSA_01"] for x in assa01_dataframe.keys()])
    assert assa01_dataframe["Australia-ASSA_01"].shape == (172490, 193)


def test_data_generator():
    start_datetime = datetime(2021, 5, 7, 0, 00, 0)
    end_datetime = datetime(2021, 5, 7, 23, 59, 0)
    generator = get_ecallisto_data_generator(start_datetime, end_datetime, ["ASSA"])
    for key, _ in generator:
        assert key in ["Australia-ASSA_01", "Australia-ASSA_02", "Australia-ASSA_60"]
