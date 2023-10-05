from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ecallisto_ng.plotting.utils import fill_missing_timesteps_with_nan


def make_times_match_spectograms(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
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


def make_frequencies_match_spectograms(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
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


def align_spectrograms_middle(spec1, spec2):
    """
    Aligns two spectrograms by finding the shift for max cross-correlation over the middle 50%.

    Parameters
    ----------
    spec1 : torch.Tensor
        First spectrogram.
    spec2 : torch.Tensor
        Second spectrogram.

    Returns
    -------
    int
        Shift amount.
    """
    summed_spec1 = torch.nansum(spec1, dim=1)
    summed_spec2 = torch.nansum(spec2, dim=1)
    middle2 = summed_spec2[
        int(0.25 * len(summed_spec2)) : int(0.75 * len(summed_spec2))
    ]
    cross_corr = F.conv1d(
        summed_spec1[None, None, :], middle2[None, None, :].flip(dims=[2])
    )
    shift = torch.argmax(cross_corr) - len(middle2)
    return shift.item()


def pairwise_cross_corr(spec_list):
    """
    Compute pairwise cross-correlation between a list of spectrograms.

    Parameters
    ----------
    spec_list : list of torch.Tensor
        List of spectrograms.

    Returns
    -------
    torch.Tensor
        Cross-correlation matrix.
    """
    N = len(spec_list)
    cross_corr_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(i + 1, N):
            shift = align_spectrograms_middle(spec_list[i], spec_list[j])
            cross_corr_matrix[i][j] = shift
            cross_corr_matrix[j][i] = -shift
    return cross_corr_matrix


def find_best_reference(cross_corr_matrix):
    """
    Find the best reference spectrogram based on the minimum sum of shifts.

    Parameters
    ----------
    cross_corr_matrix : torch.Tensor
        Cross-correlation matrix.

    Returns
    -------
    int
        Index of the best reference spectrogram.
    """
    abs_sum_shifts = torch.sum(torch.abs(cross_corr_matrix), dim=1)
    ref_idx = torch.argmin(abs_sum_shifts)
    return ref_idx


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
    spec_list : list of torch.Tensor
        List of spectrograms.
    shifts : torch.Tensor
        Shift amounts for each spectrogram.

    Returns
    -------
    list of torch.Tensor
        List of shifted spectrograms.
    """
    shifted_spectrograms = []
    for shift_, spec in zip(shifts, spec_list):
        shifted_spec = spec.shift(periods=int(shift_))
        shifted_spectrograms.append(shifted_spec)
    return shifted_spectrograms


def round_frequencies_to_nearest_bin(dfs, bin_size):
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

    Returns
    -------
    list of pandas.DataFrame
        New list of DataFrames with binned frequencies.
    """
    rounded_dfs = []
    for df in dfs:
        rounded_df = round_col_to_nearest_bin(df.copy(), bin_size)
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
