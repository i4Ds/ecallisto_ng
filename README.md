# Ecallisto NG 
## Ecallisto NG is a very basic and simple package to access the Ecallisto API. 
## It is written in Python 3.9 and uses the requests library and accesses the Ecallisto API at https://v000792.fhnw.ch/api/data

## Installation
Clone this repository and install it with pip
```pip install -e .```
## PYPI
This package is also available on PYPI: https://pypi.org/project/ecallisto-ng/
## Usage
### Data fetching
```python
from ecallisto_ng.data_fetching.get_data import get_data
parameters = {
    "instrument_name": "austria_unigraz_01",
    "start_datetime": "2021-03-01 06:30:00",
    "end_datetime": "2021-03-07 23:30:00",
    "timebucket": "15m",
    "agg_function": "MAX",
}

df = get_data(parameters)
```

### Plotting 
We offer some basic plotting functions. 
```python
from ecallisto_ng.plotting.plot import fill_missing_timesteps_with_nan, plot_spectogram

df = fill_missing_timesteps_with_nan(df)
plot_spectogram(df,  parameters["instrument_name"], parameters["start_datetime"], parameters["end_datetime"])
```
