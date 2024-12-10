import fnmatch
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial
from multiprocessing import current_process

import pandas as pd
import requests
from astropy.io import fits
from bs4 import BeautifulSoup
from tqdm import tqdm

from ecallisto_ng.data_download.utils import (
    concat_dfs_by_instrument,
    ecallisto_fits_to_pandas,
    extract_datetime_from_filename,
    extract_instrument_name,
    filter_dataframes,
    instrument_name_to_globbing_pattern,
)

BASE_URL = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto"
LOCAL_PATH = "/mnt/nas05/data01/radio/2002-20yy_Callisto"


def get_ecallisto_data(
    start_datetime,
    end_datetime,
    instrument_name=None,
    verbose=False,
    freq_start=None,
    freq_end=None,
    download_from_local=False,
):
    """
    Get the e-Callisto data within a date range and optional instrument regex pattern.
    For big requests, it is recommended to use the generator function `get_ecallisto_data_generator`,
    which allows for processing each file individually without loading everything into memory at once.

    Parameters
    ----------
    start_datetime : datetime-like
        The start date for the file search.
    end_datetime : datetime-like
        The end date for the file search.
    instrument_string : str or None
        The instrument name you want to match in file URLs. If None, all files are considered.
        Substrings also work, such as 'ALASKA'.
    verbose : bool
        Whether to print progress information.
    freq_start : float or None
        The start frequency for the filter.
    freq_end : float or None
        The end frequency for the filter.
    download_from_local : bool
        Whether to download the files from the local directory instead of the remote directory.
        Useful if you are working with a local copy of the data.

    Returns
    -------
    dict of str: `~pandas.DataFrame` or `~pandas.DataFrame`
        Dictionary of instrument names and their corresponding dataframes.
    """
    if not os.path.exists(LOCAL_PATH) and download_from_local:
        print(
            f"Local directory {LOCAL_PATH} does not exist. Downloading from remote directory."
        )
        download_from_local = False
    if download_from_local:
        file_urls = get_local_file_paths(start_datetime, end_datetime, instrument_name)
    else:
        file_urls = get_remote_files_url(start_datetime, end_datetime, instrument_name)
    if not file_urls and verbose:
        print(
            f"No files found for {instrument_name} between {start_datetime} and {end_datetime}."
        )
        return {}
    dfs = download_fits_process_to_pandas(file_urls, verbose)
    dfs = concat_dfs_by_instrument(dfs, verbose)
    dfs = filter_dataframes(
        dfs,
        start_datetime,
        end_datetime,
        verbose,
        freq_start=freq_start,
        freq_end=freq_end,
    )
    return dfs


def get_ecallisto_data_generator(
    start_datetime,
    end_datetime,
    instrument_name=None,
    freq_start=None,
    freq_end=None,
    verbose=False,
    base_url=BASE_URL,
):
    """
    Generator function to yield e-Callisto data one file at a time within a date range.
    It returns a tuple with (instrument_name, dataframe)

    This function is a generator, using `yield` to return dataframes one by one. This is beneficial
    for handling large datasets or when working with limited memory, as it allows for processing
    each file individually without loading everything into memory at once.

    Parameters
    ----------
    start_datetime : datetime-like
        The start date for the file search.
    end_datetime : datetime-like
        The end date for the file search.
    instrument_names : List[str] or str or None
        The instrument name you want to match in file URLs. If None, all files are considered.
    freq_start : float or None
        The start frequency for the filter.
    freq_end : float or None
        The end frequency for the filter.
    base_url : str
        The base URL of the remote file directory.

    Yields
    ------
    pandas.DataFrame
        A tuple with (instrument_name, dataframe)

    Example
    -------
    >>> start = <start_datetime>
    >>> end = <end_datetime>
    >>> instrument = <instrument_name>
    >>> data_generator = get_ecallisto_data_generator(start, end, instrument)
    >>> for instrument_name, data_frame in data_generator:
    ...     process_data(data_frame)  # Replace with your processing function or whatever you want to do with the data
    """
    if isinstance(instrument_name, str):
        instrument_name = [instrument_name]
    if instrument_name is None:
        # Get all instrument names with available data. This makes the generator more efficient
        # because it doesn't have to check for each instrument name individually.
        instrument_name = get_instrument_with_available_data(
            start_datetime, end_datetime
        )

    for instrument_name_ in instrument_name:
        file_urls = get_remote_files_url(
            start_datetime, end_datetime, instrument_name_, base_url
        )
        if not file_urls and verbose:
            print(
                f"No files found for {instrument_name} between {start_datetime} and {end_datetime}."
            )
            return {}
        try:
            dfs = download_fits_process_to_pandas(file_urls)
            dfs = concat_dfs_by_instrument(dfs)
            dfs = filter_dataframes(
                dfs,
                start_datetime,
                end_datetime,
                freq_start=freq_start,
                freq_end=freq_end,
            )
            for key, value in dfs.items():
                yield key, value
        except Exception as e:
            print(f"Error for {instrument_name_}: {e}")
            print(f"Skipping {instrument_name_}.")
            continue


def get_instrument_with_available_data(
    start_date=None, end_date=None, instrument_name=None
):
    """
    Retrieve sorted list of unique instrument names with available data in a specified date range.

    Parameters
    ----------
    start_date : pd.Timestamp or None
        The start date for querying data. If None, the current timestamp is used.
    end_date : pd.Timestamp or None
        The end date for querying data. If None, it is set to three days prior to the current timestamp.
    instrument_name : str, optional
        Name of the specific instrument to query. If None, all available instruments are considered.

    Returns
    -------
    list of str
        A sorted list of unique instrument names for which data is available in the specified date range.
        Returns an empty list if no data is found.

    Notes
    -----
    - The function depends on `get_remote_files_url` to fetch URLs of available data files.
    - `extract_instrument_name` is used to parse instrument names from the file URLs.
    - If both `start_date` and `end_date` are None, the function defaults to a date range from the current date to three days prior.

    Examples
    --------
    >>> get_instrument_with_available_data(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-10'), 'Instrument')
    ['InstrumentA', 'InstrumentB', 'InstrumentX']
    """
    if start_date is None or end_date is None:
        # Set start_date to now
        end_date = pd.Timestamp.now()
        # Set end_date to 1 day ago
        start_date = end_date - pd.Timedelta(days=1)
    file_urls = get_remote_files_url(start_date, end_date, instrument_name)
    if not file_urls:
        print(
            f"No files found for {instrument_name} between {start_date} and {end_date}."
        )
        return {}
    instrument_names = [extract_instrument_name(file_url) for file_url in file_urls]
    return sorted(list(set(instrument_names)))


def download_fits_process_to_pandas(file_urls, verbose=False):
    # Check if we're in a daemon process
    is_daemon = current_process().daemon
    partial_f = partial(fetch_fits_to_pandas, verbose=verbose)

    if not is_daemon:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            if verbose:
                # Show progress bar if verbose is True
                results = list(
                    tqdm(
                        executor.map(partial_f, file_urls),
                        total=len(file_urls),
                        desc="Downloading and processing files",
                    )
                )
            else:
                # Execute without progress bar if verbose is False
                results = list(executor.map(partial_f, file_urls))
    else:
        # Sequential processing for daemon processes
        if verbose:
            results = list(
                tqdm(
                    map(partial_f, file_urls),
                    total=len(file_urls),
                    desc="Downloading and processing files (sequential)",
                )
            )
        else:
            results = list(map(partial_f, file_urls))
    # Check if any of the results are None
    if verbose and any(result is None for result in results):
        print("Some files could not be downloaded (See traceback above).")
    # Remove None values
    results = [result for result in results if result is not None]
    return results


def fetch_fits_to_pandas(file_url, verbose=False):
    try:
        fits_file = fits.open(file_url, cache=False)
        df = ecallisto_fits_to_pandas(fits_file)
        # Add the instrument name to it
        df.attrs["ANTENNAID"] = extract_instrument_name(file_url)[-2:]
        return df
    except Exception as e:
        if verbose:
            print(f"Error for {file_url}: {e}")
            traceback.print_exc()
        return None


def fetch_date_files(date_url):
    """
    Fetch and parse file URLs from a given date URL.

    Parameters
    ----------
    date_url : str
        The URL for a specific date to fetch files from.
    session : requests.Session
        The requests session object for HTTP requests.

    Returns
    -------
    list of str
        List of file URLs ending with '.gz'.
    """
    session = requests.Session()
    response = session.get(date_url)
    try:
        soup = BeautifulSoup(
            response.content, "lxml"
        )  # using lxml parser because it's faster
        file_names = [
            link.get("href")
            for link in soup.find_all("a")
            if link.get("href").endswith(".gz")
        ]
        to_return = [date_url + file_name for file_name in file_names]
    except Exception as e:
        print(f"Error fetching {date_url}.")
        print(e)
        to_return = []
    finally:
        session.close()
        return to_return


def get_remote_files_url(
    start_date,
    end_date,
    instrument_name=None,
    base_url="http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto",
):
    """
    Get the remote file URLs within a date range and optional instrument regex pattern.

    Parameters
    ----------
    start_date : datetime-like
        The start date for the file search.
    end_date : datetime-like
        The end date for the file search.
    instrument_string : str or None
        The instrument name you want to match in file URLs. If None, all files are considered.
        Substrings also work, such as 'ALASKA'.
    base_url : str
        The base URL of the remote file directory.

    Returns
    -------
    list of str
        List of file URLs that match the criteria.
    """
    file_urls = []
    date_urls = [
        f"{base_url}/{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/"
        for date in pd.date_range(start_date, end_date, inclusive="both")
    ]

    # Check if we're in a daemon process
    is_daemon = current_process().daemon

    if not is_daemon:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Parallel processing
            results = executor.map(fetch_date_files, date_urls)
    else:
        # Sequential processing for daemon processes
        results = map(fetch_date_files, date_urls)

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    glob_pattern = instrument_name_to_globbing_pattern(instrument_name)
    file_urls = fnmatch.filter(results, glob_pattern)

    # Extact datetime from filename
    file_datetimes = [
        extract_datetime_from_filename(file_name) for file_name in file_urls
    ]

    # Filter out files that are not in the date range
    file_urls = [
        file_url
        for file_url, file_datetime in zip(file_urls, file_datetimes)
        if start_date - timedelta(minutes=15)
        <= file_datetime
        <= end_date + timedelta(minutes=15)
    ]  # Timedelta because a file contains up to 15 minutes of data

    return file_urls


def get_local_file_paths(
    start_date,
    end_date,
    instrument_name=None,
    base_path=LOCAL_PATH,
):
    """
    Get the local file paths within a date range and optional instrument regex pattern.

    Parameters
    ----------
    start_date : datetime-like
        The start date for the file search.
    end_date : datetime-like
        The end date for the file search.
    instrument_name : str or None
        The instrument name to match in file paths. If None, all files are considered.
        Substrings also work, such as 'ALASKA'.
    base_path : str
        The base path of the local file directory.

    Returns
    -------
    list of str
        List of file paths that match the criteria.
    """
    file_paths = []
    for date in pd.date_range(start_date, end_date, inclusive="both"):
        # Define the path for the year, month, and day
        date_path = os.path.join(
            base_path, str(date.year), str(date.month).zfill(2), str(date.day).zfill(2)
        )

        if instrument_name:
            glob_pattern = instrument_name_to_globbing_pattern(instrument_name)
            search_pattern = os.path.join(date_path, f"{glob_pattern}")
        else:
            search_pattern = os.path.join(date_path, "*")

        # Use glob to find files that match the search pattern
        for file_path in glob.glob(search_pattern):
            # Assuming extract_datetime_from_filename extracts the datetime from the filename
            file_datetime = extract_datetime_from_filename(file_path)
            # Check if the file's datetime is within the range
            if (
                start_date - timedelta(minutes=15)
                <= file_datetime
                <= end_date + timedelta(minutes=15)
            ):
                file_paths.append(file_path)

    return file_paths
