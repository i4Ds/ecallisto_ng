# Ecallisto NG 
Ecallisto NG is a compact yet effective Python package designed to facilitate seamless interaction with the Ecallisto API. 
The package is constructed in Python 3.9 and utilizes the `requests` library to directly access the Ecallisto API via the link: [https://v000792.fhnw.ch/api/redoc](https://v000792.fhnw.ch/api/redoc).

## Installation
To install this package, clone this repository and use pip for installation. Execute the following command in your terminal:
```pip install -e .```

## PyPI
Ecallisto NG is conveniently available on PyPI as well. To download, visit the following link: [https://pypi.org/project/ecallisto-ng/](https://pypi.org/project/ecallisto-ng/)

## Example
Please have a look at the jupyter notebook under `example`. 

## Usage
Here's a guide on how to use the different features of Ecallisto NG:

### Data Fetching
Fetching data is easy using the `get_data` function, housed under the `ecallisto_ng.data_fetching.get_data` module. Here's an example:

```python
from ecallisto_ng.data_fetching.get_data import get_data

parameters = {
    "instrument_name": "austria_unigraz_01",
    "start_datetime": "2021-03-01 06:30:00",
    "end_datetime": "2021-03-07 23:30:00",
    "timebucket": "15m",
    "agg_function": "MAX",
}

df = get_data(**parameters)
```

### Getting Data Availability
You can also check the availability of data using the `get_tables` and `get_table_names_with_data_between_dates` function, housed under the `ecallisto_ng.data_fetching.get_information` module. Here's an example:

```python
from ecallisto_ng.data_fetching.get_information import get_tables, get_table_names_with_data_between_dates
from datetime import datetime, timedelta

get_tables()[:5]

get_table_names_with_data_between_dates(
    start_datetime=(datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
    end_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
```

### Plotting 
Ecallisto NG provides basic plotting capabilities. Here's an example of how to generate a spectogram:
```python
from ecallisto_ng.plotting.utils import fill_missing_timesteps_with_nan, plot_spectogram

df_filled = fill_missing_timesteps_with_nan(df)
plot_spectogram(df_filled,  parameters["instrument_name"], parameters["start_datetime"], parameters["end_datetime"])
```

### Spectogram editing
We also provide some basic functionalities to edit the spectogram. Here's how you can do it:
```python
from ecallisto_ng.data_processing.utils import elimwrongchannels, subtract_constant_background, subtract_rolling_background

df = elimwrongchannels(df)
df = fill_missing_timesteps_with_nan(df)
df = subtract_constant_background(df)
df = subtract_rolling_background(df)

plot_spectogram(df,  parameters["instrument_name"], parameters["start_datetime"], parameters["end_datetime"])
```
These simple commands allow you to easily manipulate spectogram data, enabling effective use of the Ecallisto API for your needs.
