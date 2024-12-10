import numpy as np

from ecallisto_ng.combine_antennas.utils import (
    align_to_reference,
    get_cross_corr_matrix,
    make_frequencies_match_spectrograms,
    make_times_match_spectrograms,
    shift_spectrograms,
)
from ecallisto_ng.data_processing.utils import (
    apply_median_filter,
    apply_quantile_filter,
    intensity_to_linear,
    mean_filter,
    min_max_scale_per_column,
    subtract_constant_background,
    subtract_low_signal_noise_background,
)
from ecallisto_ng.plotting.utils import downcast_timedelta


def match_spectrograms(datas):
    """
    Match the time and frequency dimensions across a list of spectrogram DataFrames.
    Basically, takes the max and min time and frequency values and makes all the
    DataFrames have the same time and frequency dimensions and fills them with nans.
    Incase a NaN is between numbers, it is linearly interpolated.

    Parameters
    ----------
    datas : list of pd.DataFrame
        List of spectrogram DataFrames to be matched.

    Returns
    -------
    list of pd.DataFrame
        List of matched spectrogram DataFrames.
    """
    data_processed = make_times_match_spectrograms(datas)
    data_processed = make_frequencies_match_spectrograms(data_processed)
    return data_processed


def sync_spectrograms(dfs, shifts=None, method="maximize_cross_correlation"):
    """
    Synchronize a list of spectrograms based on pairwise cross-correlation.
    If the nans are not removed, this method does not work because it can
    lead to pointless shifts.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of spectrogram DataFrames to be synchronized.
    shifts : np.ndarray, optional
        List of shifts to apply to each DataFrame, incase you want to
        calculate the shift outside of this function.
    method : str, optional
        The method used for synchronization. Currently only supports 'maximize_cross_correlation'.
        Default is 'maximize_cross_correlation'.

    Returns
    -------
    list of pd.DataFrame
        List of synchronized spectrogram DataFrames.
    int or None
        Index of the DataFrame used as a reference for synchronization.
    """
    if method != "maximize_cross_correlation":
        raise ValueError("Unsupported method")
    if shifts is not None:
        shifted_specs = shift_spectrograms(dfs, shifts)
        ref_idx = None

    elif method == "maximize_cross_correlation":
        # Check if all dataframes have no missing values.
        # if they do, this method does not work
        for df in dfs:
            if df.isnull().all(axis=1).any():
                ValueError(
                    "Time axis of a df is all nan. This method does not work with nan values."
                )
        cross_corr_matrix = get_cross_corr_matrix(dfs)
        # Find the best reference and align all to it
        ref_idx, shifts_to_ref = align_to_reference(cross_corr_matrix)
        shifted_specs = shift_spectrograms(dfs, shifts_to_ref)
    else:
        raise ValueError(
            "Unsupported method. Only 'maximize_cross_correlation' supported"
            "or provide the shifts manually."
        )

    return shifted_specs, ref_idx


def preprocess_data(
    datas,
    db_space,
    resample_time_delta,
    min_n_frequencies,
    freq_range,
    filter_type,
    filter_size,
    filter_quantile_value,
    subtract_background,
    resample_func="mean",
):
    """
    Process a list of DataFrames based on a series of filtering and transformation steps.

    Parameters
    ----------
    datas : list of pd.DataFrame
        List of DataFrames to be processed.
    min_n_frequencies : int, optional
        Minimum number of frequencies required for processing. Default is 100.
    freq_range : list of float, optional
        Frequency range to keep. Default is [20, 80].
    filter_type : str, optional
        Type of filter to apply ('median' or 'mean'). Default is None.
    filter_size: tuple(int, int)
        Size of filter in pixel.
    filter_quantile_value: float
        Quantile value to use for filtering.
    resample_func : str, optional
        Resampling function to use. Default is 'MEAN'.

    Returns
    -------
    list of pd.DataFrame
        List of processed DataFrames.
    """
    data_processed = []
    for data in datas:
        # Cut away columns based on frequency limits
        columns = np.array([float(col) for col in data.columns])
        data = data.loc[:, (columns > freq_range[0]) & (columns < freq_range[1])]

        # Resample data
        data = downcast_timedelta(data, resample_time_delta, resample_func)

        # Check column conditions
        if len(data.columns) < min_n_frequencies:
            print(
                f"Skipping {data.attrs['FULLNAME']} it has only has {len(data.columns)} out of {min_n_frequencies} frequencies"
            )
            continue

        if db_space:
            data = intensity_to_linear(data)

        if subtract_background:
            data = subtract_constant_background(data)
            data = subtract_low_signal_noise_background(data)

        # Apply filter if specified
        if filter_type == "median":
            data = apply_median_filter(data, filter_size)
        if filter_type == "mean":
            data = mean_filter(data, filter_size)
        if filter_type == "quantile":
            data = apply_quantile_filter(data, filter_quantile_value)

        # Min max scale data per frequency
        data = min_max_scale_per_column(data)

        # Append processed data
        data_processed.append(data)
    return data_processed
