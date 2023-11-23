import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


def filter_dataframes(
    dfs, start_date, end_date, verbose=False, freq_start=None, freq_end=None
):
    """
    Filter the dataframes in a dictionary by a date range.

    Parameters
    ----------
    dfs : dict of str: `~pandas.DataFrame`
        Dictionary of instrument names and their corresponding dataframes.
    start_date : datetime-like
        The start date for the filter.
    end_date : datetime-like
        The end date for the filter.
    verbose : bool
        Whether to print progress information.
    freq_start : float or None
        The start frequency for the filter.
    freq_end : float or None
        The end frequency for the filter.

    Returns
    -------
    dict of str: `~pandas.DataFrame`
        Dictionary of instrument names and their corresponding dataframes.
    """
    if verbose:
        print("Filtering dataframes.")
    for instrument, df in dfs.items():
        dfs[instrument] = df.loc[start_date:end_date]

    if verbose:
        print("Filtering frequencies.")
    if freq_start:
        for instrument, df in dfs.items():
            dfs[instrument] = df.loc[:, freq_start:freq_end]
    if freq_end:
        for instrument, df in dfs.items():
            dfs[instrument] = df.loc[:, freq_start:freq_end]

    # Check if any dfs are empty
    empty_instruments = []
    for instrument, df in dfs.items():
        if df.empty:
            empty_instruments.append(instrument)
    for instrument in empty_instruments:
        del dfs[instrument]

    # Update the header
    if verbose:
        print("Updating headers after filtering.")
    for instrument, df in dfs.items():
        df = readd_edit_header(df, dfs[instrument].attrs)
    return dfs


def extract_datetime_from_filename(file_name):
    """
    Extract datetime from the filename.

    Parameters
    ----------
    file_name : str
        The filename from which to extract the datetime.

    Returns
    -------
    datetime
        The extracted datetime object, or None if parsing fails.
    """
    # Filename format: 'LOCATION_YYYYMMDD_HHMMSS_X.fit.gz'
    match = re.search(r"_(\d{8})_(\d{6})", file_name)
    if match:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return None


def instrument_name_to_globbing_pattern(instrument_name=None):
    """
    Convert an instrument name (and optional antenna number) to a globbing pattern suitable for matching in file URLs.

    Parameters
    ----------
    instrument_name : str
        The instrument name to be matched in the file URLs.

    Returns
    -------
    str
        A matching pattern string.
    """
    if instrument_name is None:
        return "*.fit.gz"
    antenna_number = None
    if instrument_name[-2:].isdigit():
        antenna_number = instrument_name[-2:]
        instrument_name = instrument_name[:-3]
    glob_pattern = "*" + instrument_name + "*"
    if antenna_number:
        glob_pattern += antenna_number
    glob_pattern += ".fit.gz"
    return glob_pattern


def combine_non_unique_frequency_axis(freq_axis, data, agg_function=np.max):
    """Combine non-unique frequency axis data.

    Parameters
    ----------
    spec : `~astropy.io.fits.hdu.hdulist.HDUList`
        The spectrogram to combine the frequency axis data of.
    method : callable
        The method to use to combine the frequency axis data. Defaults to "mean".

    Returns
    -------
    unique_freq_axis : `~numpy.ndarray`
        The unique frequency axis data.
    data : `~numpy.ndarray`
        The combined data.


    Notes
    -----
    The function first finds the unique frequency axis data and the indices of the non-unique frequency axis data.
    It then combines the non-unique frequency axis data using the method specified by the `method` parameter.
    """
    unique_freq, indices = np.unique(freq_axis, return_inverse=True)
    data = np.array(
        [agg_function(data[indices == i], axis=0) for i in range(len(unique_freq))]
    )
    return unique_freq, data


def spec_time_to_pd_datetime(start_datetime, time_axis):
    """
    Convert a time axis array to pandas datetime objects, offset by a starting datetime.

    Parameters
    ----------
    start_datetime : datetime or Timestamp
        The starting datetime to which the time axis offsets will be applied.
    time_axis : array_like
        An array of time offsets in seconds.

    Returns
    -------
    pandas.Series
        A pandas Series of datetime objects corresponding to each time offset.

    Notes
    -----
    This function adds the given time offsets in seconds to the start datetime
    and converts the result to pandas datetime objects.
    """
    return start_datetime + pd.to_timedelta(time_axis, unit="s")


def extract_instrument_name(file_path):
    """Extract the instrument name from a file path.

    Parameters
    ----------
    file_path : str
        The file path to extract the instrument name from.

    Returns
    -------
    str
        The extracted instrument name, converted to lowercase with underscores in place of hyphens.


    Example
    -------
    >>> extract_instrument_name('/var/lib/ecallisto/2023/01/27/ALASKA-COHOE_20230127_001500_612.fit.gz')
    'ALASKA_COHOE_612'

    Notes
    -----
    The function first selects the last part of the file path and removes the extension.
    Then, it replaces hyphens with underscores and splits on underscores to get the parts of the file name.
    The function concatenates these parts, adding a numeric part of the file name if it is less than 6 digits.
    """
    # select last part of file path and remove extension
    file_name = file_path.split("/")[-1].split(".")[0]

    # Remove datetime
    instrument_name = file_name.split("_")[0]
    antenna_number = file_name.split("_")[-1]
    return instrument_name + "_" + antenna_number


def extract_identical_dicts(dicts):
    """
    Extract identical keys and values from a list of dictionaries.

    Parameters
    ----------
    dicts : list of dict
        The list of dictionaries to extract identical keys and values from.

    Returns
    -------
    dict
        A dictionary of identical keys and values.
    """
    identical_keys = set.intersection(*[set(d.keys()) for d in dicts])
    identical_values = {}
    for key in identical_keys:
        values = [d[key] for d in dicts]
        if len(set(values)) == 1:
            identical_values[key] = values[0]
    return identical_values


def readd_edit_header(df, dict_):
    """
    Re-add and edit header information to a DataFrame.

    This function updates the header of a DataFrame with new values and adds additional
    time-related and instrument information. It preserves the order of the original header keys.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to which header information will be added or updated.
    dict_ : dict
        Dictionary containing header information to be updated or added to `df`.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with updated header information.

    Notes
    -----
    The function assumes that the DataFrame `df` has an attribute `header`, which is a dictionary
    used to store header information. The DataFrame's index is used to derive `DATE-OBS`, `TIME-OBS`,
    `DATE-END`, and `TIME-END` values.
    """
    for key, value in dict_.items():
        df.attrs[key] = value
    # Add DATE-OBS and TIME-OBS
    df.attrs["DATE-OBS"] = df.index[0].strftime("%Y-%m-%d")
    df.attrs["TIME-OBS"] = df.index[0].strftime("%H:%M:%S")
    df.attrs["DATE-END"] = df.index[-1].strftime("%Y-%m-%d")
    df.attrs["TIME-END"] = df.index[-1].strftime("%H:%M:%S")
    df.attrs["NAXIS1"] = len(df)
    df.attrs["FULLNAME"] = df.attrs["INSTRUME"] + "_" + df.attrs["ANTENNAID"]
    return df


def concat_dfs_by_instrument(dfs, verbose=False):
    instruments = {}
    # Extract attrs from each df
    headers = [df.attrs for df in dfs]
    if verbose:
        print("Combining headers.")
    for df in dfs:
        instrument = df.attrs["INSTRUME"] + "_" + df.attrs["ANTENNAID"]
        if instrument not in instruments:
            instruments[instrument] = []
        instruments[instrument].append(df)

    if verbose:
        print("Concatenating dataframes.")
    for instrument, dfs in instruments.items():
        headers = [df.attrs for df in dfs]
        identical_headers = extract_identical_dicts(headers)
        instruments[instrument] = pd.concat(dfs).sort_index()
        instruments[instrument] = readd_edit_header(
            instruments[instrument], identical_headers
        )

    return instruments


def masked_spectogram_to_array(data, freq_axis):
    """
    Converts a masked spectogram to an array by removing all masked values.
    """
    # Get row with no masked values
    idxs = np.where(~np.any(np.ma.getmaskarray(data), axis=1))[0]
    # Keep only frequencies with no masked values
    freq_axis = freq_axis[idxs]
    # keep only rows in idxs
    data = np.ma.getdata(data)
    data = data[idxs, :]

    return data, freq_axis


def ecallisto_fits_to_pandas(fits_file):
    """
    Convert eCallisto FITS data to a pandas DataFrame.

    Parameters
    ----------
    fits_file : astropy.io.fits.HDUList
        An HDUList object representing the FITS file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the FITS data with time as index and frequencies as columns.

    Notes
    -----
    This function processes eCallisto FITS files, extracting the time axis, frequency axis,
    and data values. It handles non-unique frequencies by combining them and converts the
    time axis to pandas datetime objects. FITS header information is added as attributes
    to the DataFrame.

    Non-unique frequency axes are combined using the `combine_non_unique_frequency_axis` function,
    which is not defined in this snippet and should be provided separately.
    """
    time_axis = fits_file[1].data[0][0]
    freq_axis = fits_file[1].data[0][1]
    data = np.array(fits_file[0].data, dtype=np.uint8)

    # Remove masked values
    data, freq_axis = masked_spectogram_to_array(data, freq_axis)

    if not len(np.unique(freq_axis)) == len(freq_axis):
        freq_axis, data = combine_non_unique_frequency_axis(freq_axis, data)

    start_datetime = pd.to_datetime(
        fits_file[0].header["DATE-OBS"] + " " + fits_file[0].header["TIME-OBS"]
    )
    # Cast freq_axis to float to avoid issues with pandas
    freq_axis = freq_axis.astype(float)

    pd_time_axis = spec_time_to_pd_datetime(start_datetime, time_axis)
    df = pd.DataFrame(data=data.T, index=pd_time_axis, columns=freq_axis)

    # Sort columns, so that they are in ascending order
    df = df.sort_index(axis=1)

    # Add the header
    for key, value in fits_file[0].header.items():
        df.attrs[key] = value

    # Make columns to floats and sort them
    df.columns = df.columns.astype(float)
    df = df.sort_index(axis=1)

    return df
