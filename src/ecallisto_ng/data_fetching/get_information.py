import pandas as pd
import requests

BASE_URL = "https://v000792.fhnw.ch"

def get_tables(verbose=False):
    """
    Fetch all the available table names from the eCallisto API.

    Parameters
    ----------
    verbose : bool
        Whether to print out the response from the API.
    
    Returns
    -------
    list of str
        A list containing the names of the available tables.
    """
    response = requests.get(BASE_URL + "/api/tables")

    if response.status_code == 200:
        table_names = response.json()['tables']
        if verbose:
            print(f"Received table names: {table_names}")
        return table_names
    else:
        raise ValueError(f"Error getting table names from API: {response.text}")


def get_table_names_with_data_between_dates(start_datetime, end_datetime, verbose=False):
    """
    Fetch all the available table names that contain data between the specified dates from the eCallisto API.

    Parameters
    ----------
    start_datetime : str
        The start datetime of the data availability check.
    end_datetime : str
        The end datetime of the data availability check.
    verbose : bool
        Whether to print out the response from the API.

    Returns
    -------
    list of str
        A list containing the names of the available tables that contain data between the specified dates.
    """
    data = {
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
    }

    assert pd.to_datetime(start_datetime) <= pd.to_datetime(end_datetime), (
        f"start_datetime ({start_datetime}) must be before end_datetime ({end_datetime})"
    )

    response = requests.post(BASE_URL + "/api/data_availability", json=data)

    if response.status_code == 200:
        table_names = response.json()['table_names']
        if verbose:
            print(f"Received table names: {table_names}")
        return table_names
    else:
        raise ValueError(f"Error getting table names from API: {response.text}")

def check_table_data_availability(instrument_name, start_datetime, end_datetime, verbose=False, **kwargs):
    """
    Check if a table has data between the specified dates using the eCallisto API.

    Parameters
    ----------
    table_name : str
        The name of the table to check.
    start_datetime : str
        The start datetime of the data availability check.
    end_datetime : str
        The end datetime of the data availability check.
    verbose : bool
        Whether to print out the response from the API.

    Returns
    -------
    bool
        True if the table has data between the specified dates, False otherwise.
    """
    data = {
        "instrument_name": instrument_name,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
    }

    assert pd.to_datetime(start_datetime) <= pd.to_datetime(end_datetime), (
        f"start_datetime ({start_datetime}) must be before end_datetime ({end_datetime})"
    )

    response = requests.post(BASE_URL + "/api/table_data_check", json=data)

    if response.status_code == 200:
        has_data = response.json()['has_data']
        if verbose:
            print(f"Table '{instrument_name}' has data between {start_datetime} and {end_datetime}: {has_data}")
        return has_data
    else:
        raise ValueError(f"Error checking table data availability from API: {response.text}")
