import pandas as pd
import glob
from datasets import Dataset, Image
from PIL import Image as PILImage

import os
import pandas as pd
from datetime import datetime, timedelta
from ecallisto_ng.data_download.downloader import get_ecallisto_data
from tqdm import tqdm


def load_radio_dataset(base_path: str) -> Dataset:
    """
    Loads a radio dataset from parquet files located within the specified base path.

    This function searches for parquet files within the given base path, extracts
    metadata such as antenna information and datetime from the file paths, converts
    these data into a Pandas DataFrame, and then transforms it into a Hugging Face
    Dataset. It also reads image data from the parquet files and converts them into
    PIL images.

    Parameters
    ----------
    base_path : str
        The base directory path where the parquet files are located. The parquet files
        are expected to be in subdirectories named after antennas.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object containing the image data and associated metadata.
    """

    images = glob.glob(f"{base_path}*/*.parquet")
    df = pd.DataFrame({"image": images})
    df["antenna"] = df["image"].apply(lambda x: x.split("/")[-2])
    df["datetime"] = df["image"].apply(
        lambda x: x.split("/")[-1].replace(".parquet", "")
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d_%H-%M-%S")

    dataset = Dataset.from_pandas(df)

    def load_image_from_parquet(example):
        parquet_path = example["image"]
        df_parquet = pd.read_parquet(parquet_path)
        example["image"] = PILImage.fromarray(df_parquet.values.T)
        return example

    dataset = dataset.map(load_image_from_parquet)
    dataset = dataset.cast_column("image", Image())

    return dataset


def create_overlapping_parquets(
    start_datetime: datetime,
    end_datetime: datetime,
    instruments: list,
    folder: str = "~/.cache/ecallisto_ng/data",
    duration: timedelta = timedelta(minutes=15),
    min_duration: timedelta = timedelta(minutes=10),
    overlap: timedelta = timedelta(minutes=1),
):
    folder = os.path.expanduser(folder)
    os.makedirs(folder, exist_ok=True)
    start_datetimes = pd.date_range(
        start_datetime, end_datetime, inclusive="both", freq=duration - overlap
    )
    for instrument in tqdm(instruments, desc="[Instruments]", position=1):
        for start_datetime in tqdm(
            start_datetimes, desc="[Dates]", leave=False, position=2
        ):
            dfs = get_ecallisto_data(
                start_datetime,
                start_datetime + duration,
                instrument_name=instrument,
            )
            for inst, df in dfs.items():
                if df is not None and not df.empty:
                    if df.index.max() - df.index.min() > min_duration:
                        filename = (
                            f"{start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.parquet"
                        )
                        path = os.path.join(folder, inst, filename)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        df.to_parquet(path)
