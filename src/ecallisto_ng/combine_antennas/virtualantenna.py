try:
    import torch
except:
    print("PyTorch not found. Please install it.")

from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ecallisto_ng.combine_antennas.combine import (
    match_spectrograms,
    preprocess_data,
    sync_spectrograms,
)
from ecallisto_ng.combine_antennas.utils import round_frequencies_to_nearest_bin


class EcallistoVirtualAntenna:
    """
    A class to create a virtual antenna from multiple e-CALLISTO spectrograms by preprocessing,
    synchronizing, matching, and combining them. It enables background subtraction, filtering,
    and adjustment in dB space, along with frequency binning and quantile stacking to combine
    the spectrograms into a single virtual antenna representation.

    Parameters
    ----------
    min_n_frequencies : int, optional
        Minimum number of frequencies required in a spectrogram, by default 30.
    freq_range : Optional[Tuple[int, int]], optional
        Frequency range to consider in the spectrograms, by default [-np.inf, np.inf].
    subtract_background : bool, optional
        Flag to enable or disable background subtraction, by default True.
    filter_type : Optional[Literal['median', 'mean']], optional
        Type of filter to apply ('median' or 'mean'), by default None.
    filter_size : Tuple[int, int], optional
        Size of the filter kernel, by default (3,3).
    db_space : bool, optional
        Flag to indicate whether the data is in dB space or not, by default True.
        
    Example
    -------
    >>> virtual_antenna = EcallistoVirtualAntenna(
        min_n_frequencies=30,
        freq_range=(50, 400),
        subtract_background=True,
        filter_type='median',
        filter_size=(3, 3),
        db_space=True
    )
    >>> dfs = [pd.DataFrame(np.random.rand(100, 100)), pd.DataFrame(np.random.rand(100, 100))]
    >>> synced_data, ref_idx = virtual_antenna.preprocess_match_sync(dfs, method='round', bin_width=0.2)
    >>> combined_spectrogram = virtual_antenna.combine(dfs, quantile=0.5)
    """
    def __init__(
        self,
        min_n_frequencies: int = 30,
        freq_range: Optional[Tuple[int, int]] = [-np.inf, np.inf],
        subtract_background: bool = True,
        filter_type: Optional[Literal['median', 'mean']] = 'median',
        filter_size: Tuple[int, int] = (3,3), 
        db_space: bool = True,
    ):
        self.min_n_frequencies = min_n_frequencies
        self.freq_range = freq_range
        self.subtract_background = subtract_background
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.db_space = db_space
        self.data = None

    def _preprocess(self, dfs: List[pd.DataFrame]):
        data_processed = preprocess_data(
            dfs,
            db_space=self.db_space,
            min_n_frequencies=self.min_n_frequencies,
            subtract_background=self.subtract_background,
            filter_type=self.filter_type,
            filter_size=self.filter_size, 
            freq_range=self.freq_range,
        )
        return data_processed
    
    def _sync_and_match(self, data_processed):
        matched_data = match_spectrograms(data_processed)
        synced_data, ref_idx = sync_spectrograms(matched_data)
        return synced_data, ref_idx

    def preprocess_match_sync(
        self,
        dfs: List[pd.DataFrame],
        method: Literal["round", "rebin"] = "round",
        bin_width: float = 0.2,
    ):
        """
        Preprocesses, matches, and synchronizes a list of spectrogram DataFrames.
        This is a higher-level function that calls the individual preprocessing, matching,
        and synchronization methods.

        Parameters:
        - dfs (List[pd.DataFrame]): List of spectrogram DataFrames to process.
        - method (Literal["round", "rebin"], optional): Method to use for frequency binning. Defaults to "round".
        - bin_width (float, optional): Bin width for frequency binning. Defaults to 0.2.

        Returns:
        - Tuple[List[pd.DataFrame], int]: A tuple containing the list of synchronized and processed spectrograms
        and the index of the reference spectrogram.
        """
        if method == "rebin":
            print(
                f"Warning! Rebinning is very unstable. When a bin contains any NANs, the whole bin will be NAN."
            )
        print(f"Combining {len(dfs)} spectograms.")
        data_processed = self._preprocess(dfs)
        print(f"Binning the frequencies with a bin width of {bin_width}.")
        data_binned = round_frequencies_to_nearest_bin(
            data_processed, bin_width, method=method
        )
        print("Matching and syncing the spectograms.")
        synced_data, ref_idx = self._sync_and_match(data_binned)
        print(f"Reference spectogram is {dfs[ref_idx].attrs['INSTRUME']}.")
        return synced_data, ref_idx

    def combine(
        self,
        dfs: List[pd.DataFrame],
        quantile: float = 0.5,
    ):
        """
        Combines multiple spectrograms into a virtual antenna spectrogram using quantile stacking.
        This function computes the quantile across the stack of spectrograms at each time-frequency point.

        Parameters:
        - dfs (List[pd.DataFrame]): List of spectrogram DataFrames to combine.
        - quantile (float): Quantile to use for stacking. Defaults to 0.5 (median).

        Returns:
        - pd.DataFrame: DataFrame representing the combined virtual antenna spectrogram.
        """
        torch_shifted = torch.stack([torch.from_numpy(df.values) for df in dfs])
        torch_quantile = torch.nanquantile(torch_shifted, quantile, dim=0)

        df = pd.DataFrame(torch_quantile, columns=dfs[0].columns, index=dfs[0].index)
        df.attrs["FULLNAME"] = "VIRTUAL"
        return df
