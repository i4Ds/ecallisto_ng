# Ecallisto NG 
Ecallisto NG is a comprehensive Python package tailored for interacting with Ecallisto data. It focuses on facilitating efficient data manipulation, processing, and visualization, leveraging the power of Pandas for data handling and PyTorch for advanced computations. The package is particularly optimized for dealing with large datasets, providing tools for slicing, filtering, and resampling data to make spectrogram plotting more manageable.

## Table of Contents
- [Background](#background)
- [Installation](#installation)
  - [PyPI](#pypi)
  - [Dev Installation](#dev-installation)
  - [Virtual Antenna Creation](#creation-of-the-virtual-antenna)
- [Using Pandas with Ecallisto NG](#pandas)
- [Examples and Usage](#examples-and-usage)
  - [Data Fetching](#data-fetching-deprecated)
  - [Data Processing and Visualization](#data-processing-and-visualization)
  - [Plotting](#plotting)
  - [spectrogram Editing](#spectrogram-editing)
- [Additional Information](#additional-information)
  - [Note on .attrs and FITS Header](#note-on-attrs-and-fits-header)
  - [Contributing](#contributing)

## Background
The package is built with Python 3.9<3.12 and primarily uses the Pandas library for data manipulation. It's core functionality is centered around efficient data processing and visualization. The data provided by e-Callisto is stored in a pandas Dataframe, where the index is the time and the column names are the observed frequencies.

## PyPI
Available on PyPI: https://pypi.org/project/ecallisto-ng/

## Installation
To install this package, use pip for installation. Execute the following command in your terminal with basic functionality, such as plotting and data fetching:
```pip install ecallisto-ng```
If you want to use the virtual antenna:
```pip install ecallisto-ng[va]```
If you want to use the detection of flares, powered by ML:
```pip install ecallisto-ng[ml]```

To use the notebook, please install the notebook dependencies.

## Dev Installation
Of course, you can also install the package from source. To do so, clone the repository and install the package in editable mode:
```pip install -e .```
```pip install -e .[va]```
```pip install -e .[ml]```

## Pandas
Pandas is an open-source data analysis and manipulation tool, pivotal to Ecallisto NG. Learning Pandas is essential for effectively using Ecallisto NG, as it allows for sophisticated data slicing, filtering, and aggregation. More on Pandas: https://pandas.pydata.org/docs/

## Examples and Usage
Please have a look at the jupyter notebook under `example`.

# Getting data
## `get_ecallisto_data` Function

This function fetches e-Callisto data within a specified date range and optional instrument regex pattern. It's suitable for smaller datasets. For larger datasets, consider using the `get_ecallisto_data_generator` function.

## Parameters
- `start_datetime` (datetime-like): The start date for the file search.
- `end_datetime` (datetime-like): The end date for the file search.
- `instrument_string` (str, List[str] or None): Instrument name(s) for file URL matching. If `None`, all files are considered.
- `freq_start` (float or None): The start frequency for filtering.
- `freq_end` (float or None): The end frequency for filtering.
- `base_url` (str): Base URL of the remote file directory.

## Returns
- (dict of str: pandas.DataFrame) or pandas.DataFrame: A dictionary of instrument names and their corresponding dataframes. If only one instrument is found, it returns a single dataframe.

## Example
```python
from ecallisto_ng.data_download.downloader import (
    get_ecallisto_data,
)
from datetime import datetime

start = datetime(2021, 5, 7, 0, 00, 0)
end = datetime(2021, 5, 7, 23, 59, 0)
instrument_name = "Australia-ASSA_01"

dfs = get_ecallisto_data(start, end, instrument_name)
df = dfs[instrument_name] # Returns a dict of pd.Dataframes because instrument_name can also be a substring, e.g. "ASSA".
```
# Getting data via a generator
## `get_ecallisto_data_generator` Function

A generator function that yields e-Callisto data one file at a time within a date range. It's ideal for handling large datasets or when working with limited memory.

## Parameters
- `start_datetime` (datetime-like): The start date for the file search.
- `end_datetime` (datetime-like): The end date for the file search.
- `instrument_name` (List[str], str, or None): Instrument name(s) for file URL matching. If `None`, all files are considered.
- `freq_start` (float or None): The start frequency for filtering.
- `freq_end` (float or None): The end frequency for filtering.
- `base_url` (str): Base URL of the remote file directory.

## Yields
- (str, pandas.DataFrame): A tuple containing the instrument name and its corresponding DataFrame.

## Example
```python
from ecallisto_ng.data_download.downloader import (
    get_ecallisto_data_generator,
)
from datetime import datetime

start = datetime(2021, 5, 7, 0, 00, 0)
end = datetime(2021, 5, 7, 23, 59, 0)
instrument_name = ["Australia-ASSA_01", "Australia-ASSA_02"]

data_generator = get_ecallisto_data_generator(start, end, instrument_name)
for instrument_name, data_frame in data_generator:
    print(instrument_name)
    print(f"{df.shape=}")
```
## Plotting 
Ecallisto NG provides basic plotting capabilities. Here's an example of how to generate a spectrogram (make sure that df is defined):
```python
from ecallisto_ng.plotting.plotting import plot_spectrogram

plot_spectrogram(df)
```
Make use of .resample to reduce the size of the data. Alternatively, you can pass a `resolution` parameter to the plot_spectrogram. Here's an example:
```python
plot_spectrogram(df.resample("1min").max())
plot_spectrogram(df, resolution=720) # Pixels
```
For more documentation on resample, please refer to the [Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
## spectrogram editing
We also provide some basic functionalities to edit the spectrogram. Here's how you can do it (make sure that df is defined):
```python
from ecallisto_ng.data_processing.utils import elimwrongchannels, subtract_constant_background, subtract_low_signal_noise_background
from datetime import datetime
from ecallisto_ng.plotting.plotting import plot_spectrogram

# Filter keep frequencies only between 40 and 70 MHz
df = df.loc[:, 20:80]

# Select specific time
start = datetime(2021, 5, 7, 3, 39, 0)
end = datetime(2021, 5, 7, 3, 42, 0)
df = df.loc[start:end]

# Edit data
df = elimwrongchannels(df)
df = subtract_low_signal_noise_background(df)
df = subtract_constant_background(df)

plot_spectrogram(df)
```
## Additional Information
### Note on .attrs and FITS Header
The function utilizes DataFrames with the .attrs attribute to store FITS header information. This attribute is a dictionary-like object and contains metadata about the FITS file, including header details. Accessing .attrs is essential for understanding the context of the data:

```python
print(df.attrs)
```

These simple commands allow you to easily manipulate spectrogram data, enabling effective use of the Ecallisto API for your needs.

### Contributing
Contributions to Ecallisto NG are very welcome! If you have an idea for an improvement or have found a bug, please feel free to contribute. The preferred way to contribute is by submitting a Pull Request (PR) or creating an issue on our GitHub repository. This way, we can discuss potential changes or fixes and maintain the quality of the project.