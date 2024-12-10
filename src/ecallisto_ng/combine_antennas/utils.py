from typing import List

import numpy as np
import pandas as pd

from ecallisto_ng.plotting.utils import fill_missing_timesteps_with_nan


def make_times_match_spectrograms(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Adjusts the time index of the given list of DataFrames to have the same start and end times.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of DataFrames with datetime index.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames with the adjusted datetime index.
    """
    min_datetime = min([df.index.min() for df in dfs])
    max_datetime = max([df.index.max() for df in dfs])
    new_dfs = []
    for df in dfs:
        df = fill_missing_timesteps_with_nan(
            df, start_datetime=min_datetime, end_datetime=max_datetime
        )
        df = df.interpolate(method="linear", axis=0, limit_area="inside")
        new_dfs.append(df)

    return new_dfs


def interpolate_columns(df: pd.DataFrame, all_columns: List[float]) -> pd.DataFrame:
    """
    Interpolates missing columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to interpolate.
    all_columns : List[float]
        List of all columns to include in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values.
    """
    df = df.reindex(columns=all_columns)
    df.interpolate(method="linear", axis=1, inplace=True, limit_area="inside")
    return df


def make_frequencies_match_spectrograms(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Makes frequency columns across multiple spectrogram DataFrames consistent.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of spectrogram DataFrames.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames with matching frequency columns.
    """
    all_columns = sorted(list(set(float(col) for df in dfs for col in df.columns)))
    all_columns = [str(col) for col in all_columns]
    new_dfs = [interpolate_columns(df, all_columns) for df in dfs]
    return new_dfs


def get_max_cross_corr_shift(spec1, spec2):
    """
    Get the shift amount that maximizes the cross-correlation between two spectrograms.

    Parameters
    ----------
    spec1 : np.array
        First spectrogram.
    spec2 : np.array
        Second spectrogram.

    Returns
    -------
    int
        Shift amount for the second spectrogram that maximizes the cross-correlation.
    """
    cross_corr = np.correlate(
        spec1.sum(axis=1).values, spec2.sum(axis=1).values, mode="full"
    )
    return cross_corr.argmax() - (len(spec1) - 1)


def get_cross_corr_matrix(specs: List[pd.DataFrame]) -> np.ndarray:
    """
    Get the cross-correlation matrix between a list of spectrograms.

    Parameters
    ----------
    specs : List[pd.DataFrame]
        List of spectrograms.

    Returns
    -------
    torch.Tensor
        Cross-correlation matrix.
    """
    n_specs = len(specs)
    cross_corr_matrix = np.zeros((n_specs, n_specs))
    for i in range(n_specs):
        for j in range(i + 1, n_specs):
            shift = get_max_cross_corr_shift(specs[i], specs[j])
            cross_corr_matrix[i, j] = shift
            cross_corr_matrix[j, i] = -shift
    return cross_corr_matrix


def find_best_reference(cross_corr_matrix):
    """
    Find the best reference spectrogram based on the minimum sum of shifts.

    Parameters
    ----------
    cross_corr_matrix : np.ndarray
        Cross-correlation matrix.

    Returns
    -------
    int
        Index of the best reference spectrogram.
    """
    abs_sum_shifts = np.sum(np.abs(cross_corr_matrix), axis=1)
    return abs_sum_shifts.argmin()


def align_to_reference(cross_corr_matrix):
    """
    Align all spectrograms to the best reference based on the cross-correlation matrix.

    Parameters
    ----------
    cross_corr_matrix : torch.Tensor
        Cross-correlation matrix.

    Returns
    -------
    int, torch.Tensor
        Index of reference and shifts needed to align to the reference.
    """
    ref_idx = find_best_reference(cross_corr_matrix)
    shifts_to_ref = cross_corr_matrix[ref_idx]
    return ref_idx, shifts_to_ref


def shift_spectrograms(spec_list, shifts):
    """
    Shift spectrograms based on the given shifts.

    Parameters
    ----------
    spec_list : list of pd.DataFrame
        List of spectrograms.
    shifts : np.ndarray
        Shift amounts for each spectrogram.

    Returns
    -------
    list of torch.Tensor
        List of shifted spectrograms.
    """
    shifted_spectrograms = []
    for shift_, spec in zip(shifts, spec_list):
        shifted_spec = spec.shift(int(shift_))
        shifted_spectrograms.append(shifted_spec)
    return shifted_spectrograms


def round_frequencies_to_nearest_bin(dfs, bin_size, method="rebin"):
    """
    Rounds each frequency column in multiple DataFrames to the nearest bin edge and groups them.
    This is so that the frequencies are consistent across multiple DataFrames and we don't
    have to deal with a huge number of columns.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of DataFrames containing the spectrograms. Columns in each DataFrame are frequencies.
    bin_size : float
        The size of the frequency bins.
    method : str, optional
        The method used for rounding. Either by rebinning or by rounding to the nearest bin edge.

    Returns
    -------
    list of pandas.DataFrame
        New list of DataFrames with binned frequencies.
    """
    rounded_dfs = []
    for df in dfs:
        if method == "round":
            rounded_df = round_col_to_nearest_bin(df.copy(), bin_size)
        elif method == "rebin":
            rounded_df = rebin_dataframe(df.copy(), bin_size)
        rounded_dfs.append(rounded_df)

    return rounded_dfs


def round_col_to_nearest_bin(df, bin_size):
    """
    Rounds each frequency column to the nearest bin edge and groups them.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the spectrogram. Columns are frequencies.
    bin_size : float
        The size of the frequency bins.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with binned frequencies.
    """

    # Cast column labels to float, round them, then cast back to str
    rounded_columns = np.round(df.columns.astype(float) / bin_size) * bin_size
    rounded_columns = np.round(rounded_columns, 1)
    df.columns = rounded_columns.astype(str)

    # Group by rounded frequencies and average the values
    return df.T.groupby(df.columns).mean().T


def compute_weights(old_freqs, new_freqs, new_res):
    """
    Vectorized computation of weights for each old frequency based on their overlap with the new frequency bins.

    Parameters:
    old_freqs (np.array): Array of old frequency values.
    new_freqs (np.array): Array of new frequency bin values.
    new_res (float): New resolution.

    Returns:
    np.array: 2D array of weights for each old frequency against each new frequency.
    """
    # Calculate the boundaries of each old frequency bin
    left_bounds = np.zeros_like(old_freqs)
    right_bounds = np.zeros_like(old_freqs)

    left_bounds[1:] = (old_freqs[1:] + old_freqs[:-1]) / 2
    left_bounds[0] = old_freqs[0] - (old_freqs[1] - old_freqs[0]) / 2

    right_bounds[:-1] = left_bounds[1:]
    right_bounds[-1] = old_freqs[-1] + (old_freqs[-1] - old_freqs[-2]) / 2

    # Calculate overlaps for each old frequency against each new frequency
    new_freqs_expanded = new_freqs.reshape(1, -1)
    overlaps = np.maximum(
        np.minimum(right_bounds.reshape(-1, 1), new_freqs_expanded + new_res / 2)
        - np.maximum(left_bounds.reshape(-1, 1), new_freqs_expanded - new_res / 2),
        0,
    )

    return overlaps


def round_to_nearest(x, base):
    return base * round(x / base)


def rebin_dataframe(df, new_res):
    """
    Optimized rebinning of DataFrame using vectorized operations.

    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and frequency columns.
    new_res (float): New resolution.

    Returns:
    pd.DataFrame: Rebinned DataFrame.
    """
    old_freqs = np.round(np.array(df.columns, dtype=np.float64), 5)

    start_freq = new_res * round_to_nearest(old_freqs.min(), new_res)
    end_freq = new_res * round_to_nearest(old_freqs.max(), new_res)
    new_freqs = np.arange(start_freq, end_freq, new_res)
    weights = compute_weights(old_freqs, new_freqs, new_res)

    # Normalize weights and compute rebinned values
    normalized_weights = weights / weights.sum(axis=0)
    rebinned_values = df.values @ normalized_weights

    rebinned_df = pd.DataFrame(rebinned_values, index=df.index, columns=new_freqs)

    return rebinned_df
