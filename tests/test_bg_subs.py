import pytest
import numpy as np
import pandas as pd

from ecallisto_ng.data_processing.utils import (
    elimwrongchannels,
    subtract_constant_background,
    subtract_low_signal_noise_background,
)


@pytest.mark.parametrize(
    "function",
    [
        elimwrongchannels,
        subtract_constant_background,
        subtract_low_signal_noise_background,
    ],
)
def test_bg_subs(small_assa01_dataframe, function):
    df = small_assa01_dataframe["Australia-ASSA_01"]
    df_processed = function(df)

    assert df_processed.shape == df.shape
    assert df_processed.columns.to_list() == df.columns.to_list()
    assert df_processed.index.to_list() == df.index.to_list()
    assert df_processed.isna().sum().sum() == 0
    assert not df_processed.equals(df)


def _constbacksub_reference(im):
    """Numpy transcription of radiospectra ``Spectrogram.constbacksub``.

    ``im`` is shaped (freq, time).
    """
    im = im - np.average(im, axis=1)[:, None]
    sdev_arr = np.std(im, axis=0)
    n_part = int(im.shape[1] * 0.05)
    zist = np.argsort(sdev_arr, kind="stable")[: n_part + 1]
    background = np.average(im[:, zist], axis=1)
    return im - background[:, None]


@pytest.fixture
def synthetic_spectrogram():
    rng = np.random.default_rng(42)
    n_time, n_freq = 400, 50
    # Strong frequency-dependent bandpass plus noise everywhere
    bandpass = np.linspace(100.0, 200.0, n_freq)
    data = bandpass[None, :] + rng.normal(0.0, 5.0, size=(n_time, n_freq))
    # Known quiet period with much lower noise
    quiet = slice(100, 130)
    data[quiet] = bandpass[None, :] + rng.normal(0.0, 0.1, size=(30, n_freq))
    # A broadband burst that must not end up in the background
    data[250:300] += np.linspace(0.0, 80.0, n_freq)[None, :]
    index = pd.date_range("2024-01-01", periods=n_time, freq="250ms")
    columns = np.linspace(45.0, 870.0, n_freq)
    return pd.DataFrame(data, index=index, columns=columns), quiet


def test_low_signal_noise_background_matches_constbacksub(synthetic_spectrogram):
    df, quiet = synthetic_spectrogram

    result = subtract_low_signal_noise_background(df)
    expected = _constbacksub_reference(df.to_numpy().T).T

    np.testing.assert_allclose(result.to_numpy(), expected, rtol=0, atol=1e-9)
    assert result.index.equals(df.index)
    assert result.columns.equals(df.columns)
    # int(400 * 0.05) + 1 = 21 samples, all taken from the quiet period, so
    # the quiet period ends up close to zero in every channel
    assert np.abs(result.iloc[quiet].to_numpy()).max() < 1.0


def test_low_signal_noise_background_handles_nans(synthetic_spectrogram):
    df, quiet = synthetic_spectrogram
    df.iloc[10, 3] = np.nan
    df.iloc[110, 7] = np.nan  # inside the quiet period
    df.iloc[200, :] = np.nan  # fully missing time sample

    result = subtract_low_signal_noise_background(df)

    assert result.isna().equals(df.isna())
    assert np.nanmax(np.abs(result.iloc[quiet].to_numpy())) < 1.0
