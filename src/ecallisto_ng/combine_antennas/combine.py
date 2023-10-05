import numpy as np
import torch

from ecallisto_ng.combine_antennas.utils import (
    align_to_reference,
    make_frequencies_match_spectograms,
    make_times_match_spectograms,
    pairwise_cross_corr,
    shift_spectrograms,
)
from ecallisto_ng.data_processing.utils import (
    apply_median_filter,
    elimwrongchannels,
    mean_filter,
    subtract_constant_background,
)
from ecallisto_ng.plotting.plotting import fill_missing_timesteps_with_nan


def match_spectrograms(datas):
    """
    Match the time and frequency dimensions across a list of spectrogram DataFrames.

    Parameters
    ----------
    datas : list of pd.DataFrame
        List of spectrogram DataFrames to be matched.

    Returns
    -------
    list of pd.DataFrame
        List of matched spectrogram DataFrames.
    """
    data_processed = make_times_match_spectograms(datas)
    data_processed = make_frequencies_match_spectograms(data_processed)
    return data_processed


def sync_spectrograms(datas, method="maximize_cross_correlation"):
    """
    Synchronize a list of spectrograms based on pairwise cross-correlation.
    If the nans are not removed, this method does not work because it can
    lead to pointless shifts.

    Parameters
    ----------
    datas : list of pd.DataFrame
        List of spectrogram DataFrames to be synchronized.
    method : str, optional
        The method used for synchronization. Currently only supports 'maximize_cross_correlation'.
        Default is 'maximize_cross_correlation'.

    Returns
    -------
    list of pd.DataFrame
        List of synchronized spectrogram DataFrames.
    int
        Index of the DataFrame used as a reference for synchronization.
    """
    if method != "maximize_cross_correlation":
        raise ValueError("Unsupported method")

    datas_torch = [torch.from_numpy(df.values) for df in datas]
    if method == "maximize_cross_correlation":
        # Check if all dataframes have no missing values.
        # if they do, this method does not work
        for df in datas:
            if df.isnull().all(axis=1).any():
                print(
                    "Time axis of a df is all nan. This method does not work with nan values."
                )
                return datas, np.nan
        cross_corr_matrix = pairwise_cross_corr(datas_torch)
    else:
        raise ValueError(
            "Unsupported method. Only 'maximize_cross_correlation' supported"
        )

    # Find the best reference and align all to it
    ref_idx, shifts_to_ref = align_to_reference(cross_corr_matrix)
    shifted_specs = shift_spectrograms(datas, shifts_to_ref)

    return shifted_specs, ref_idx


def preprocess_data(
    datas, min_n_frequencies=30, freq_range=[20, 80], max_freq=150, filter_type=None
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
    max_freq : float, optional.
        Maximum frequency an instrument should measure. Default is 150.
    filter_type : str, optional
        Type of filter to apply ('median' or 'mean'). Default is None.

    Returns
    -------
    list of pd.DataFrame
        List of processed DataFrames.
    """
    data_processed = []
    for data in datas:
        try:
            if max([float(col) for col in data.columns]) > max_freq:
                continue

            # Cut away columns based on frequency limits
            columns = np.array([float(col) for col in data.columns])
            data = data.loc[:, (columns > freq_range[0]) & (columns < freq_range[1])]

            # Check column conditions
            if len(data.columns) < min_n_frequencies:
                continue

            # Data transformations
            data = fill_missing_timesteps_with_nan(data)
            data = elimwrongchannels(data)
            data = subtract_constant_background(data, 100)

            # Apply filter if specified
            if filter_type == "median":
                data = apply_median_filter(data)
            if filter_type == "mean":
                data = mean_filter(data)

            # Cap min value to 0 and scale to [0, 1]
            data[data < 0] = 0
            data = (data - data.min()) / (data.max() - data.min())
            data.fillna(0, inplace=True)

            # Append processed data
            data_processed.append(data)
        except Exception as e:
            print("Error processing data")
            print(e)
    return data_processed
