import fnmatch
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import pandas as pd
import requests
from astropy.io import fits
from bs4 import BeautifulSoup

from ecallisto_ng.data_download.utils import (
    concat_dfs_by_instrument,
    ecallisto_fits_to_pandas,
    extract_datetime_from_filename,
    extract_instrument_name,
    filter_dataframes,
    instrument_name_to_globbing_pattern,
)

BASE_URL = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"
def get_ecallisto_data(start_datetime, end_datetime, instrument_name=None, freq_start=None, freq_end=None, base_url=BASE_URL):
    """
    Get the e-Callisto data within a date range and optional instrument regex pattern.

    Parameters
    ----------
    start_datetime : datetime-like
        The start date for the file search.
    end_datetime : datetime-like
        The end date for the file search.
    instrument_string : str or None
        The instrument name you want to match in file URLs. If None, all files are considered.
        Substrings also work, such as 'ALASKA'.
    freq_start : float or None
        The start frequency for the filter.
    freq_end : float or None
        The end frequency for the filter.

    base_url : str
        The base URL of the remote file directory.

    Returns
    -------
    dict of str: `~pandas.DataFrame`
        Dictionary of instrument names and their corresponding dataframes.
    """
    file_urls = get_remote_files_url(start_datetime, end_datetime, instrument_name, base_url)
    if not file_urls:
        print(f"No files found for {instrument_name} between {start_datetime} and {end_datetime}.")
        return {}
    dfs = download_fits_process_to_pandas(file_urls)
    dfs = concat_dfs_by_instrument(dfs)
    dfs = filter_dataframes(dfs, start_datetime, end_datetime, freq_start=freq_start, freq_end=freq_end)
    if len(dfs) == 1:
        return dfs[list(dfs.keys())[0]]
    else:
        return dfs

def get_instrument_with_available_data(start_date, end_date, instrument_name=None, base_url=BASE_URL):
    file_urls = get_remote_files_url(start_date, end_date, instrument_name, base_url)
    if not file_urls:
        print(f"No files found for {instrument_name} between {start_date} and {end_date}.")
        return {}
    instrument_names = [extract_instrument_name(file_url) for file_url in file_urls]
    return sorted(list(set(instrument_names)))

def fetch_fits_to_pandas(file_url):
    fits_file = fits.open(file_url, cache=False)
    df = ecallisto_fits_to_pandas(fits_file)
    # Add the instrument name to it
    df.attrs['ANTENNAID'] = extract_instrument_name(file_url)[-2:]
    return df

def download_fits_process_to_pandas(file_urls):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map each URL to a fetch function with a session
        results = executor.map(fetch_fits_to_pandas, file_urls)
    # Flatten the results and return them
    return [x for x in results]


def fetch_date_files(date_url, session):
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
    response = session.get(date_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')  # using lxml parser because it's faster
        file_names = [link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.gz')]
        return [date_url + file_name for file_name in file_names]
    return []


def get_remote_files_url(start_date, end_date, instrument_name=None, base_url="http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"):
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
    date_urls = [f"{base_url}/{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/"
                 for date in pd.date_range(start_date, end_date, inclusive="both")]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a session for each worker
        sessions = [requests.Session() for _ in range(os.cpu_count())]
        # Map each URL to a fetch function with a session
        results = executor.map(fetch_date_files, date_urls, sessions)

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    glob_pattern = instrument_name_to_globbing_pattern(instrument_name)
    file_urls = fnmatch.filter(results, glob_pattern)

    # Extact datetime from filename
    file_datetimes = [extract_datetime_from_filename(file_name) for file_name in file_urls]

    # Filter out files that are not in the date range
    file_urls = [file_url for file_url, file_datetime in zip(file_urls, file_datetimes)
                 if start_date - timedelta(minutes=15) <= file_datetime <= end_date + timedelta(minutes=15)] # Timedelta because a file contains up to 15 minutes of data

    # Close all sessions
    for session in sessions:
        session.close()

    return file_urls
