import time

import pandas as pd
import requests

BASE_URL = "https://v000792.fhnw.ch"


def get_data(
    instrument_name,
    start_datetime,
    end_datetime,
    timebucket=None,
    agg_function=None,
    return_type="json",
    verbose=False,
):
    """
    Get data from the eCallisto API. See: https://v000792.fhnw.ch/api/redoc
    Of course, this is just a wrapper around the requests.post function.
    Depending on the size, the request can take a while. For example, two
    weeks of data, aggregated in a specific way, can take around 20 seconds.

    Parameters
    ----------
    instrument_name : str
        The name of the instrument to get data from.
    start_datetime : str
        The start datetime of the data to get.
    end_datetime : str
        The end datetime of the data to get.
    timebucket : str
        In what time frame the data should be grouped (see timescaledb
        "timebucket" function)
    agg_function : str
        The aggregation function to use (see timescaledb "timebucket" function)
    return_type : str
        The type of data to return. Either 'json' or 'fits'. If 'json', the
        data is returned as a pandas DataFrame. If 'fits', the data is
        downloaded as a fits file and the filename is returned.
    verbose : bool
        Whether to print out the response from the API.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the data from the eCallisto API.
    """
    data = {
        "instrument_name": instrument_name,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "timebucket": timebucket,
        "agg_function": agg_function,
        "return_type": return_type,
    }

    assert pd.to_datetime(start_datetime) < pd.to_datetime(end_datetime), (
        f"start_datetime ({start_datetime}) must be before end_datetime ({end_datetime})"
    )

    response = requests.post(BASE_URL + "/api/data", json=data, timeout=180)

    if response.status_code == 200:
        url = (
            response.json()["json_url"]
            if return_type == "json"
            else response.json()["fits_url"]
        )
        url = BASE_URL + url
        while True:
            # Sleep for a short period of time to allow the data to be fetched
            time.sleep(5)
            # Check if the file is available yet
            file_response = requests.get(url)
            if file_response.status_code == 200:
                # If the file is available, return the data
                respone_json = file_response.json()
                if 'error' in respone_json.keys():
                    raise ValueError(respone_json['error'])
                if return_type == "json":
                    df = pd.DataFrame(file_response.json())
                    return df
                else:
                    fits_path = (
                        f"{instrument_name}_{start_datetime}_{end_datetime}.fits"
                    )
                    with open(fits_path, "wb") as f:
                        f.write(file_response.content)
                    return fits_path
            elif file_response.status_code == 404:
                # If the file is not found, continue waiting
                continue
            else:
                raise ValueError(f"Error getting file from API: {file_response.text}")
    else:
        raise ValueError(f"Error getting data from API: {response.text}")

