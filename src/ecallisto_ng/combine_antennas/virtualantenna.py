# %%
try:
    import torch
except:
    print("PyTorch not found. Please install it.")

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ecallisto_ng.combine_antennas.combine import (
    match_spectrograms,
    preprocess_data,
    sync_spectrograms,
)
from ecallisto_ng.combine_antennas.utils import round_frequencies_to_nearest_bin


class EcallistoVirtualAntenna:
    def __init__(
        self,
        min_n_frequencies: int = 30,
        freq_range: Optional[Tuple[int, int]] = [-np.inf, np.inf],
        subtract_background: bool = True,
        filter_type: Optional[str] = None,
        db_space: bool = True,
    ):
        self.min_n_frequencies = min_n_frequencies
        self.freq_range = freq_range
        self.subtract_background = subtract_background
        self.filter_type = filter_type
        self.db_space = db_space
        self.data = None

    def _preprocess(self, dfs: List[pd.DataFrame]):
        data_processed = preprocess_data(
            dfs,
            db_space=self.db_space,
            min_n_frequencies=self.min_n_frequencies,
            subtract_background=self.subtract_background,
            filter_type=self.filter_type,
            freq_range=self.freq_range,
        )
        return data_processed

    def _sync_and_match(self, data_processed):
        matched_data = match_spectrograms(data_processed)
        synced_data, ref_idx = sync_spectrograms(matched_data)
        return synced_data, ref_idx

    def combine(
        self,
        dfs: List[pd.DataFrame],
        method: Literal["round", "rebin"] = "rebin",
        bin_width: float = 0.2,
    ):
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
