"""Constructed spectrograms for the unified spectrogram.

These are not archive files. The instrument names follow e-Callisto style so
the site rule can be checked. The numbers are chosen so the shared drift is
obvious and the single-instrument spike is obvious.
"""

import numpy as np
import pandas as pd

from ecallisto_ng.combine_antennas.unified import (
    background_excess,
    enhancement_mask,
    unified_spectrogram,
)


def _spectrogram(values: np.ndarray, instrument: str, antenna: str) -> pd.DataFrame:
    index = pd.date_range("2024-12-01", periods=values.shape[0], freq="1s")
    columns = np.arange(20.0, 20.0 + values.shape[1], dtype=float)
    frame = pd.DataFrame(values, index=index, columns=columns)
    frame.attrs["INSTRUME"] = instrument
    frame.attrs["ANTENNAID"] = antenna
    frame.attrs["FULLNAME"] = f"{instrument}_{antenna}"
    return frame


def _drift(shift: int, n_time: int = 80, n_freq: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.05, size=(n_time, n_freq))
    for step, time in enumerate(range(30, 50)):
        row = time + shift
        freq = step // 2
        if 0 <= row < n_time:
            data[row, freq] = 8.0
            data[row, freq + 1] = 6.0
    return data


def test_excess_is_background_subtracted_and_not_scaled():
    frame = _spectrogram(_drift(0), "AUSTRIA-UNIGRAZ", "01")
    excess = background_excess(frame)
    column = excess.columns[0]
    assert abs(excess[column].median()) < 1e-9
    assert excess.max().max() > 1

    mask = enhancement_mask(excess, snr_threshold=5)
    assert mask[30:50, 0:2].any()
    assert not mask[0:10, :].any()


def test_shared_drift_is_kept_and_single_instrument_spike_is_dropped():
    austria = _spectrogram(_drift(0), "AUSTRIA-UNIGRAZ", "01")
    mexico = _spectrogram(_drift(5), "MEXICO-LANCE", "01")
    swiss = _spectrogram(_drift(-4), "SWISS-IRSOL", "01")
    # A disturbance only Mexico sees. Large enough to be its own feature.
    mexico.iloc[5:8, 14:16] = 25.0

    out = unified_spectrogram(
        [austria, mexico, swiss],
        max_lag=pd.Timedelta(seconds=10),
        reference="AUSTRIA-UNIGRAZ",
    )

    assert out.attrs["FULLNAME"] == "UNIFIED"
    drift = out.iloc[30:50, 0:11]
    spike = out.iloc[5:8, 14:16]
    assert drift.notna().sum().sum() > 30
    assert drift.max().max() > 5
    assert spike.isna().all().all()
    # One copy on the Austrian clock, not a second copy at Mexico's lag.
    assert out.notna().sum().sum() < 60


def test_same_instrument_antennas_do_not_confirm_each_other():
    antenna_a = _spectrogram(np.zeros((80, 16)), "GERMANY-DLR", "01")
    antenna_b = _spectrogram(np.zeros((80, 16)), "GERMANY-DLR", "02")
    antenna_a.iloc[10:16, 4:7] = 20.0
    antenna_b.iloc[10:16, 4:7] = 20.0

    out = unified_spectrogram(
        [antenna_a, antenna_b],
        max_lag=pd.Timedelta(seconds=10),
    )
    assert out.isna().all().all()


def test_burst_absent_from_the_reference_is_kept_on_the_writer_clock():
    quiet = _spectrogram(np.zeros((80, 16)), "AUSTRIA-UNIGRAZ", "01")
    mexico = _spectrogram(_drift(0), "MEXICO-LANCE", "01")
    swiss = _spectrogram(_drift(3), "SWISS-IRSOL", "01")

    out = unified_spectrogram(
        [quiet, mexico, swiss],
        max_lag=pd.Timedelta(seconds=10),
        reference="AUSTRIA-UNIGRAZ",
    )
    # Mexico writes, because it is alphabetically before Swiss and Austria
    # does not contain the shape. The drift stays on Mexico's clock.
    assert out.iloc[30:50].notna().sum().sum() > 30
    assert out.iloc[0:10].isna().all().all()


def test_station_without_those_frequencies_is_not_a_vote():
    austria = _spectrogram(_drift(0), "AUSTRIA-UNIGRAZ", "01")
    mexico = _spectrogram(_drift(2), "MEXICO-LANCE", "01")
    other_band = _spectrogram(np.zeros((80, 16)), "ALASKA-HAARP", "01")
    other_band.columns = np.arange(100.0, 116.0)
    other_band.iloc[30:50, :] = 30.0

    out = unified_spectrogram(
        [austria, mexico, other_band],
        max_lag=pd.Timedelta(seconds=10),
        reference="AUSTRIA-UNIGRAZ",
    )
    assert out.loc[:, austria.columns].notna().sum().sum() > 30
    assert out.loc[:, other_band.columns].isna().all().all()
