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
    data_folder="ecallisto_ng_cache",
    verbose=False,
    max_retries=3,
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
    data_folder : str
        Where to save the cached data.
    verbose : bool
        Whether to print the progress or not.
    max_retries : int
        The maximum number of retries to download the data.
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
    }

    assert pd.to_datetime(start_datetime) <= pd.to_datetime(
        end_datetime
    ), f"start_datetime ({start_datetime}) must be before end_datetime ({end_datetime})"

    # Clean up dates to make it compatible with windows
    start_datetime = start_datetime.replace(":", "-")
    end_datetime = end_datetime.replace(":", "-")

    # Create file path
    file_path = os.path.join(
        data_folder,
        f"{instrument_name}_{start_datetime}_{end_datetime}_{timebucket}_{agg_function}.parquet",
    )
    os.makedirs(data_folder, exist_ok=True)
    if os.path.exists(file_path):
        print(f"Reading data from {file_path}")
        return pd.read_parquet(file_path)

    # Send the request to the API
    response = requests.post(BASE_URL + "/api/data", json=data, timeout=180)
    # Check if the request was successful
    if response.status_code == 200:
        if verbose:
            print(
                "Data retrieval started successfully. Waiting for file to be ready..."
            )
        # Get the URL for the parquet file
        parquet_url = response.json()["data_parquet_url"]
        json_url = response.json()["info_json_url"]
        file_id = response.json()["file_id"]

        # Now we can start polling the URL until the parquet file is ready for download
        n_tries = 0
        while True:
            try:
                if verbose:
                    print(f"Trying to download file {file_id}")
                # Try to download the parquet file
                file_response = requests.get(BASE_URL + parquet_url)
                # If the file is available, save it to disk
                if file_response.status_code == 200:
                    print("File downloaded successfully")
                    with open(file_path, "wb") as f:
                        f.write(file_response.content)
                    # Check that the file is bigger than 8 bytes (sometimes, the API returns an empty file)
                    if os.path.getsize(file_path) > 8:
                        # Return the file as a DataFrame
                        return pd.read_parquet(file_path)
                    else:
                        # Remove file and try again
                        os.remove(file_path)
                        raise ValueError(
                            f"Error downloading file: {file_response.status_code}. File is empty."
                        )
                elif file_response.status_code == 204:
                    # Check if the file creation causes any errors
                    json_response = requests.get(
                        BASE_URL + json_url
                    )  # This json contains information about your data request (e.g. status)
                    # Check the status of the json
                    if "status" in json_response.json():
                        if json_response.json()["status"] == "processing":
                            # If the file is not found, sleep for a short period and try again
                            if verbose:
                                print(f"File {file_id} not ready yet, waiting...")
                            time.sleep(3)
                        elif json_response.json()["status"] == "ok":
                            if verbose:
                                print(
                                    f"File {file_id} succesfully written! Will return file."
                                )
                        else:
                            raise ValueError(
                                f"Error downloading file: {json_response.json()['status']}"
                            )
                    elif "error" in json_response.json():
                        raise ValueError(
                            f"Error downloading file: {json_response.json()['error']}"
                        )
                else:
                    print(f"Error downloading file: {file_response.status_code}")
                    json_response = requests.get(BASE_URL + json_url)
                    print(json_response.json())
                    break
            except Exception as e:
                print(
                    f"""Error downloading file: {e}. Try {n_tries} of {max_retries}. Retrying in 3 seconds...
                    """
                )
                n_tries += 1
                if n_tries > max_retries:
                    raise ValueError(
                        f"Error downloading file: {file_response.status_code}. Max retries reached."
                    )
                time.sleep(3)
    else:
        print(f"Error starting data retrieval: {response.status_code}")


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
