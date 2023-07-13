import os
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

    assert pd.to_datetime(start_datetime) <= pd.to_datetime(end_datetime), (
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


# Because of SQL limitation, the names of the tables do not perfectly match the instrument names.
# This function converts the instrument name to the table name.
def extract_instrument_name(file_path):
    """
    Because of SQL limitation, the names of the tables do not perfectly match the instrument names.
    This function converts the instrument name to the table name.

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
    'alaska_cohoe_612'

    Notes
    -----
    The function first selects the last part of the file path and removes the extension.
    Then, it replaces hyphens with underscores and splits on underscores to get the parts of the file name.
    The function concatenates these parts, adding a numeric part of the file name if it is less than 6 digits.
    """
    # select last part of path and remove extension
    file_name_parts = os.path.basename(file_path).split(".")[0]
    # replace '-' with '_' and split on '_' to get the parts of the file name
    file_name_parts = file_name_parts.replace("-", "_").lower().split("_")
    file_name = ""
    for part in file_name_parts:
        if not part.isnumeric():
            file_name += "_" + part
    if (
        len(file_name_parts[-1]) < 6 and file_name_parts[-1].isnumeric()
    ):  # Sometimes, the last part is an ID number for when the station has multiple instruments.
        # We want to add this to the file name if it's not a time (6 digits).
        file_name = file_name + "_" + file_name_parts[-1]

    return file_name[1:]  # Remove the first '-'


def reverse_extract_instrument_name(instrument_name, include_number=False):
    """
    Because of SQL limitation, the names of the tables do not perfectly match the instrument names.
    Convert a lower-case instrument name with underscores to its original hyphenated form.

    Parameters
    ----------
    instrument_name : str
        The instrument name in lower-case with underscores.
    include_number : bool, optional
        Whether to include the last number in the output or not. Default is False.

    Returns
    -------
    str
        The original instrument name with hyphens.

    Example
    -------
    >>> reverse_extract_instrument_name('alaska_cohoe_612')
    'ALASKA-COHOE'
    >>> reverse_extract_instrument_name('alaska_cohoe_612', include_number=True)
    'ALASKA-COHOE_612'

    """
    # Replace underscores with hyphens and upper all the letters
    parts = [part.upper() for part in instrument_name.split("_")]
    if not include_number:
        # Remove the last part if it's a number
        if parts[-1].isnumeric():
            parts.pop()
    # Join the parts with hyphens and return the result
    return "-".join(parts)
