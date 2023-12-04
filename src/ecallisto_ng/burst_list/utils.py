import os

import pandas as pd


def load_burst_list(burst_list_file=None):
    """
    Load a burst list from a specified Excel file into a pandas DataFrame.

    This function reads an Excel file containing information about bursts and
    loads it into a pandas DataFrame. If no file path is provided, it defaults to
    'burst_list.xlsx' located in a 'data' folder in the same directory as this script.

    Parameters
    ----------
    burst_list_file : str, optional
        Path to the Excel file containing the burst list. If not provided, defaults to
        'burst_list.xlsx' in the 'data' subdirectory of the script's directory.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the burst list data.

    Examples
    --------
    Load a burst list from the default location:

    >>> burst_list = load_burst_list()

    Load a burst list from a specific file:

    >>> burst_list = load_burst_list("path/to/burst_list.xlsx")
    """
    if burst_list_file is None:
        burst_list_file = os.path.join(
            os.path.dirname(__file__), "data", "burst_list.xlsx"
        )
    if burst_list_file.endswith(".csv"):
        burst_list = pd.read_csv(burst_list_file)
    elif burst_list_file.endswith(".xlsx"):
        burst_list = pd.read_excel(burst_list_file)
    else:
        raise ValueError("Unknown file format")
    return burst_list
