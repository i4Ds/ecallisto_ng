import pytest

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
