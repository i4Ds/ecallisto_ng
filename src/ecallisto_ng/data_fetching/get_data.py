import pandas as pd
import requests

BASE_URL = "https://v000792.fhnw.ch/api/data"


def get_data(instrument_name, start_datetime, end_datetime):
    """
    Get data from the eCallisto API.

    Parameters
    ----------
    instrument_name : str
        The name of the instrument to get data from.
    start_datetime : str
        The start datetime of the data to get.
    end_datetime : str
        The end datetime of the data to get.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the data from the eCallisto API.
    """
    data = {
        "instrument_name": instrument_name,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
    }
    response = requests.post(BASE_URL, json=data, timeout=180)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        return df
    else:
        raise ValueError(f"Error getting data from API: {response.text}")
