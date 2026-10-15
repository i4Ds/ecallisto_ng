"""Unified spectrogram: keep emission that independent instruments share.

Solar emission is the time–frequency shape that appears in more than one
instrument, up to a clock lag. Emission seen in only one instrument is left
out. This is a separate path from ``EcallistoVirtualAntenna``, which takes one
lag for a whole chunk and then a quantile across every station.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

# 1.4826 * MAD is a robust stand-in for the standard deviation of a Gaussian.
_MAD_TO_SIGMA = 1.4826


def instrument_of(df: pd.DataFrame) -> str:
    """Return the instrument name. Antennas of one instrument are one site."""
    name = df.attrs.get("INSTRUME")
    if not name:
        raise ValueError("Each spectrogram needs attrs['INSTRUME']")
    return str(name)


def background_excess(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each frequency's own median. Do not min-max scale.

    The median is the station's background only when the enhancement is the
    minority of the samples in that frequency. A burst that fills most of the
    interval is absorbed into the background and will not be marked.
    """
    values = df.to_numpy(dtype=float)
    baseline = np.nanmedian(values, axis=0)
    excess = pd.DataFrame(values - baseline, index=df.index, columns=df.columns)
    excess.attrs = dict(df.attrs)
    return excess


def enhancement_mask(excess: pd.DataFrame, snr_threshold: float) -> np.ndarray:
    """Mark samples above ``snr_threshold`` robust standard deviations.

    The scale is computed per frequency from the median absolute deviation.
    A frequency with no variation is never enhanced. ``snr_threshold`` is a
    starting cut, not a measured false-alarm rate.
    """
    values = excess.to_numpy(dtype=float)
    center = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - center), axis=0)
    scale = _MAD_TO_SIGMA * mad
    enhanced = np.isfinite(values) & (scale > 0) & (values > snr_threshold * scale)
    return enhanced


def unified_spectrogram(
    dfs: Sequence[pd.DataFrame],
    snr_threshold: float = 5.0,
    max_lag: pd.Timedelta = pd.Timedelta(seconds=30),
    min_correlation: float = 0.5,
    min_overlap: float = 0.4,
    min_pixels: int = 4,
    reference: str | None = None,
) -> pd.DataFrame:
    """Merge enhanced emission that shares its shape across instruments.

    Parameters
    ----------
    dfs :
        Spectrograms with time on the index, frequency on the columns, and
        ``attrs['INSTRUME']`` set. Two antennas with the same ``INSTRUME``
        cannot confirm each other.
    snr_threshold :
        Enhancement cut in robust standard deviations above the per-frequency
        median. Not a measured false-alarm rate.
    max_lag :
        Largest clock shift to try. The clock-error distribution of the
        network has not been measured; this bound is a parameter.
    min_correlation :
        Minimum Pearson correlation of the feature shape after the best lag.
    min_overlap :
        Fraction of the feature's enhanced pixels that must also be enhanced
        in the other instrument after that lag.
    min_pixels :
        Smallest connected enhancement that is treated as a feature.
    reference :
        Instrument whose clock labels the output when a feature can be matched
        to it. Defaults to the instrument with the most finite samples.

    Returns
    -------
    pd.DataFrame
        Median excess of the agreeing instruments. Pixels that are quiet, or
        enhanced in only one instrument, are NaN. ``attrs['FULLNAME']`` is
        ``UNIFIED``.

    Notes
    -----
    A feature is written once, by the alphabetically first instrument that
    shows it. If that feature also matches the reference instrument, it is
    placed on the reference clock. If the reference does not show the shape,
    the feature stays on the writer's own clock: there is no shared signal
    from which to invent a lag.
    """
    prepared = [_prepare(df, snr_threshold) for df in dfs]
    if len({item["instrument"] for item in prepared}) < 2:
        return _empty_like(prepared)

    ref_name = reference or _default_reference(prepared)
    ref_items = [item for item in prepared if item["instrument"] == ref_name]
    if not ref_items:
        known = sorted({item["instrument"] for item in prepared})
        raise ValueError(f"Reference {ref_name!r} is not in {known}")
    ref = ref_items[0]

    origin, step = _time_step(prepared)
    for item in prepared:
        item["samples"] = _sample_index(item["frame"].index, origin, step)
        item["sample_lookup"] = {}
        item["sample_to_time"] = {}
        for row, sample in enumerate(item["samples"]):
            key = int(sample)
            if key not in item["sample_lookup"]:
                item["sample_lookup"][key] = row
                item["sample_to_time"][key] = pd.Timestamp(item["frame"].index[row])
        item["column_lookup"] = {
            freq: i for i, freq in enumerate(item["columns"])
        }
    max_lag_samples = max(int(round(pd.Timedelta(max_lag) / step)), 0)

    records: list[tuple[pd.Timestamp, float, float]] = []
    for source in prepared:
        for rows, cols in _features(source["mask"], min_pixels):
            witnesses = _agreeing_instruments(
                source,
                rows,
                cols,
                prepared,
                max_lag_samples,
                min_correlation,
                min_overlap,
            )
            names = {source["instrument"], *witnesses}
            if len(names) < 2 or source["instrument"] != min(names):
                continue
            on_reference = source["instrument"] == ref_name
            lag_to_ref = 0
            if not on_reference:
                placed = _best_match(source, rows, cols, ref, max_lag_samples)
                if (
                    placed is not None
                    and placed[1] >= min_correlation
                    and placed[2] >= min_overlap
                ):
                    lag_to_ref = placed[0]
                    on_reference = True
            clock = ref if on_reference else None
            records.extend(
                _merged_pixels(source, rows, cols, witnesses, clock, lag_to_ref)
            )

    out = _records_to_frame(records, prepared)
    out.attrs["INSTRUME"] = "UNIFIED"
    out.attrs["FULLNAME"] = "UNIFIED"
    out.attrs["REFERENCE"] = ref_name
    return out


def _prepare(df: pd.DataFrame, snr_threshold: float) -> dict:
    frame = df.copy()
    frame.columns = [float(column) for column in frame.columns]
    excess = background_excess(frame)
    return {
        "instrument": instrument_of(frame),
        "frame": excess,
        "values": excess.to_numpy(dtype=float),
        "mask": enhancement_mask(excess, snr_threshold),
        "columns": [float(column) for column in excess.columns],
    }


def _empty_like(prepared: list[dict]) -> pd.DataFrame:
    if not prepared:
        out = pd.DataFrame()
    else:
        index = prepared[0]["frame"].index
        columns = list(prepared[0]["columns"])
        for item in prepared[1:]:
            index = index.union(item["frame"].index)
            columns = sorted(set(columns) | set(item["columns"]))
        out = pd.DataFrame(np.nan, index=index, columns=columns)
    out.attrs["INSTRUME"] = "UNIFIED"
    out.attrs["FULLNAME"] = "UNIFIED"
    return out


def _default_reference(prepared: list[dict]) -> str:
    counts: dict[str, int] = {}
    for item in prepared:
        finite = int(np.isfinite(item["values"]).sum())
        name = item["instrument"]
        counts[name] = counts.get(name, 0) + finite
    return max(counts, key=lambda name: counts[name])


def _time_step(prepared: list[dict]) -> tuple[pd.Timestamp, pd.Timedelta]:
    origin = min(item["frame"].index.min() for item in prepared)
    deltas = []
    for item in prepared:
        diff = pd.Series(item["frame"].index).diff().dropna()
        if not diff.empty:
            deltas.append(diff.median())
    if not deltas:
        raise ValueError("Spectrograms need at least two time samples")
    step = pd.Timedelta(pd.Series(deltas).median())
    if step <= pd.Timedelta(0):
        raise ValueError("Time index must increase")
    return pd.Timestamp(origin), step


def _sample_index(
    index: pd.Index, origin: pd.Timestamp, step: pd.Timedelta
) -> np.ndarray:
    delta = pd.Series(pd.DatetimeIndex(index)) - origin
    return np.rint(delta / step).to_numpy(dtype=int)


def _features(
    mask: np.ndarray, min_pixels: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    labeled, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    features = []
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labeled == label_id)
        if len(coords) < min_pixels:
            continue
        features.append((coords[:, 0], coords[:, 1]))
    return features


def _agreeing_instruments(
    source: dict,
    rows: np.ndarray,
    cols: np.ndarray,
    prepared: list[dict],
    max_lag_samples: int,
    min_correlation: float,
    min_overlap: float,
) -> dict[str, tuple[dict, int]]:
    """Best antenna of each other instrument that matches this feature.

    The lag is the integer added to a source sample to reach the same moment
    in the other observation. Frequency is never shifted.
    """
    found: dict[str, tuple[dict, int, float]] = {}
    for other in prepared:
        if other["instrument"] == source["instrument"]:
            continue
        matched = _best_match(source, rows, cols, other, max_lag_samples)
        if matched is None:
            continue
        lag, score, overlap = matched
        if score < min_correlation or overlap < min_overlap:
            continue
        current = found.get(other["instrument"])
        if current is None or score > current[2]:
            found[other["instrument"]] = (other, lag, score)
    return {name: (item, lag) for name, (item, lag, _) in found.items()}


def _best_match(
    source: dict,
    rows: np.ndarray,
    cols: np.ndarray,
    other: dict,
    max_lag_samples: int,
) -> tuple[int, float, float] | None:
    patch, component, row_positions, other_cols = _feature_patch(
        source, rows, cols, other
    )
    if patch is None:
        return None
    source_samples = source["samples"][row_positions]
    best: tuple[int, float, float] | None = None
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        other_patch, other_enhanced = _shifted_patch(
            other, source_samples, lag, patch.shape, other_cols
        )
        score = _pearson(patch, other_patch)
        if not np.isfinite(score):
            continue
        overlap = float(other_enhanced[component].mean()) if component.any() else 0.0
        if best is None or score > best[1]:
            best = (lag, score, overlap)
    return best


def _feature_patch(
    source: dict,
    rows: np.ndarray,
    cols: np.ndarray,
    other: dict,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[None, None, None, None]
):
    """Shape of one feature on the frequencies both observations cover."""
    row_positions = np.unique(rows)
    col_positions = np.unique(cols)
    row_map = {int(row): i for i, row in enumerate(row_positions)}
    col_map = {int(col): i for i, col in enumerate(col_positions)}
    patch = np.zeros((len(row_positions), len(col_positions)), dtype=float)
    component = np.zeros(patch.shape, dtype=bool)
    for row, col in zip(rows, cols):
        i = row_map[int(row)]
        j = col_map[int(col)]
        component[i, j] = True
        patch[i, j] = source["values"][int(row), int(col)]

    keep = []
    other_cols = []
    for j, col in enumerate(col_positions):
        freq = source["columns"][int(col)]
        other_col = other["column_lookup"].get(freq)
        if other_col is None:
            continue
        keep.append(j)
        other_cols.append(other_col)
    if not keep:
        return None, None, None, None
    keep_arr = np.array(keep, dtype=int)
    return (
        patch[:, keep_arr],
        component[:, keep_arr],
        row_positions,
        np.array(other_cols, dtype=int),
    )


def _shifted_patch(
    other: dict,
    source_samples: np.ndarray,
    lag: int,
    shape: tuple[int, int],
    other_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    other_patch = np.full(shape, np.nan)
    other_enhanced = np.zeros(shape, dtype=bool)
    for i, sample in enumerate(source_samples):
        other_row = other["sample_lookup"].get(int(sample) + lag)
        if other_row is None:
            continue
        other_patch[i, :] = other["values"][other_row, other_cols]
        other_enhanced[i, :] = other["mask"][other_row, other_cols]
    return other_patch, other_enhanced


def _merged_pixels(
    source: dict,
    rows: np.ndarray,
    cols: np.ndarray,
    witnesses: dict[str, tuple[dict, int]],
    clock: dict | None,
    lag_to_clock: int,
) -> list[tuple[pd.Timestamp, float, float]]:
    records = []
    for row, col in zip(rows, cols):
        source_sample = int(source["samples"][int(row)])
        freq = source["columns"][int(col)]
        values = [float(source["values"][int(row), int(col)])]
        for item, lag in witnesses.values():
            other_row = item["sample_lookup"].get(source_sample + lag)
            other_col = item["column_lookup"].get(freq)
            if other_row is None or other_col is None:
                continue
            values.append(float(item["values"][other_row, other_col]))
        if clock is None:
            timestamp = pd.Timestamp(source["frame"].index[int(row)])
        else:
            timestamp = clock["sample_to_time"].get(source_sample + lag_to_clock)
            if timestamp is None:
                continue
        records.append((timestamp, freq, float(np.nanmedian(values))))
    return records


def _records_to_frame(
    records: list[tuple[pd.Timestamp, float, float]], prepared: list[dict]
) -> pd.DataFrame:
    index = prepared[0]["frame"].index
    columns = set(prepared[0]["columns"])
    for item in prepared[1:]:
        index = index.union(item["frame"].index)
        columns |= set(item["columns"])
    out = pd.DataFrame(
        np.nan, index=index.unique().sort_values(), columns=sorted(columns)
    )
    grouped: dict[tuple[pd.Timestamp, float], list[float]] = {}
    for timestamp, freq, value in records:
        grouped.setdefault((pd.Timestamp(timestamp), float(freq)), []).append(value)
    for (timestamp, freq), values in grouped.items():
        out.loc[timestamp, freq] = float(np.nanmedian(values))
    return out


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 4:
        return np.nan
    x = left[finite] - left[finite].mean()
    y = right[finite] - right[finite].mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)
