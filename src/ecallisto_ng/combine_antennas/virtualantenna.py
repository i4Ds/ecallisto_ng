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
        Size of the filter kernel, by default (12,12).
    db_space : bool, optional
        Flag to indicate whether the data is in dB space or not, by default True.

    Example
    -------
    >>> virtual_antenna = EcallistoVirtualAntenna(
        min_n_frequencies=30,
        freq_range=(50, 400),
        subtract_background=True,
        filter_type='median',
        filter_size=(12, 12),
        db_space=True
    )
    >>> dfs = [pd.DataFrame(np.random.rand(100, 100)), pd.DataFrame(np.random.rand(100, 100))]
    >>> synced_data, ref_idx = virtual_antenna.preprocess_match_sync(dfs, method='round', bin_width=0.2)
    >>> combined_spectrogram = virtual_antenna.combine(dfs, quantile=0.5)
    """

    def __init__(
        self,
        min_n_frequencies: int = 30,
        resample_time_delta: pd.Timedelta = pd.Timedelta(250, unit="ms"),
        freq_range: Optional[Tuple[int, int]] = [-np.inf, np.inf],
        subtract_background: bool = True,
        filter_type: Optional[Literal["median", "mean", "quantile"]] = "median",
        filter_quantile_value: float = 0.5,  # TODO
        filter_size: Tuple[int, int] = (12, 12),
        db_space: bool = True,
    ):
        self.min_n_frequencies = min_n_frequencies
        self.freq_range = freq_range
        self.resample_time_delta = resample_time_delta
        self.subtract_background = subtract_background
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.filter_quantile_value = filter_quantile_value
        self.db_space = db_space
        self.data = None

    def _preprocess(self, dfs: List[pd.DataFrame]):
        data_processed = preprocess_data(
            dfs,
            db_space=self.db_space,
            resample_time_delta=self.resample_time_delta,
            min_n_frequencies=self.min_n_frequencies,
            subtract_background=self.subtract_background,
            filter_type=self.filter_type,
            filter_size=self.filter_size,
            filter_quantile_value=self.filter_quantile_value,
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
        print(f"Combining {len(dfs)} spectrograms.")
        data_processed = self._preprocess(dfs)
        print(f"Binning the frequencies with a bin width of {bin_width}.")
        data_binned = round_frequencies_to_nearest_bin(
            data_processed, bin_width, method=method
        )
        print("Matching and syncing the spectrograms.")
        synced_data, ref_idx = self._sync_and_match(data_binned)
        print(f"Reference spectrogram is {dfs[ref_idx].attrs['INSTRUME']}.")
        return synced_data, ref_idx

    @staticmethod
    def _combine_quantile(dfs, quantile):
        torch_shifted = torch.stack([torch.from_numpy(df.values) for df in dfs])
        torch_quantile = torch.nanquantile(torch_shifted, quantile, dim=0)
        return pd.DataFrame(torch_quantile, columns=dfs[0].columns, index=dfs[0].index)

    @staticmethod
    def _combine_loss(dfs, epochs, ignore_ratio, grad_penalty_weight, lr):
        tensor_list = [torch.tensor(df.values, dtype=torch.float32) for df in dfs]
        noise_tensor = torch.rand(tensor_list[0].shape, requires_grad=True)

        optimizer = torch.optim.Adam([noise_tensor], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            losses = torch.stack(
                [torch.nanmean(torch.abs(t - noise_tensor)) for t in tensor_list]
            )

            # Determine the threshold to ignore the top x% dataframes
            threshold = torch.quantile(losses, 1 - ignore_ratio)

            # Compute masked losses, ignoring dataframes with losses above the threshold
            mask = losses <= threshold
            masked_losses = torch.where(
                mask, losses, torch.tensor(0.0, device=losses.device)
            )
            mae_loss = torch.mean(masked_losses)

            # Calculate the gradient penalty for the noise tensor
            grad_x = torch.abs(torch.diff(noise_tensor, dim=1))
            grad_y = torch.abs(torch.diff(noise_tensor, dim=0))
            grad_penalty = grad_penalty_weight * (
                torch.mean(grad_x) + torch.mean(grad_y)
            )

            # Total loss
            total_loss = mae_loss + grad_penalty

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {mae_loss.item()}")

        # Use only the non-ignored dataframes for the final combination
        optimized_tensor_list = [tensor for tensor, m in zip(tensor_list, mask) if m]
        optimized_tensor_stack = torch.stack(optimized_tensor_list)
        optimized_noise_tensor = torch.mean(optimized_tensor_stack, dim=0)

        optimized_noise_df = pd.DataFrame(
            optimized_noise_tensor.detach().numpy(),
            index=dfs[0].index,
            columns=dfs[0].columns,
        )

        return optimized_noise_df

    def combine(
        self,
        dfs: List[pd.DataFrame],
        method: Literal["quantile", "loss"] = "quantile",
        quantile: float = 0.5,
        epochs: int = 1000,
        ignore_ratio: float = 0.1,
        grad_penalty_weight: float = 0.001,
        lr: float = 0.01,
    ) -> pd.DataFrame:
        """
        Combines multiple spectrograms into a virtual antenna spectrogram.
        This function provides different methods for combining the spectrograms.

        Parameters:
        - dfs (List[pd.DataFrame]): List of spectrogram DataFrames to combine.
        - method (str): Method for combining the spectrograms. Options: 'quantile', 'loss'. Defaults to 'quantile'.
        - quantile (float): Quantile to use for stacking. Defaults to 0.5 (median).
        - epochs (int): Number of epochs for optimization. Defaults to 1000.
        - ignore_ratio (float): Ratio of top loss dataframes to ignore. Defaults to 0.1.
        - grad_penalty_weight (float): Weight for the gradient penalty. Defaults to 0.001.
        - lr (float): Learning rate for optimization. Defaults to 0.01.

        Returns:
        - pd.DataFrame: DataFrame representing the combined virtual antenna spectrogram.
        """
        if method == "quantile":
            df = self._combine_quantile(dfs, quantile)
        elif method == "loss":
            df = self._combine_loss(dfs, epochs, ignore_ratio, grad_penalty_weight, lr)
        else:
            raise ValueError("Invalid method. Supported methods: 'quantile', 'loss'")
        df.attrs["FULLNAME"] = "VIRTUAL"
        return df
