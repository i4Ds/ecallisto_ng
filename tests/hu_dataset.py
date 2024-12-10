import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
from glob import glob

from PIL import Image as PILImage
from datasets import Dataset
from ecallisto_ng.data_download.hu_dataset import (
    load_radio_dataset,
    create_overlapping_parquets,
)


def test_create_overlapping_parquets_and_load():
    start_date = datetime(2024, 1, 1, 11, 0, 0)
    end_date = datetime(2024, 1, 1, 12, 0, 0)
    instruments = ["GERMANY-DLR_63"]
    temp_dir = os.path.expanduser("~/.cache/ecallisto_ng/TEST/")

    # Create Parquet files
    create_overlapping_parquets(
        start_datetime=start_date,
        end_datetime=end_date,
        instruments=instruments,
        folder=temp_dir,
        duration=timedelta(minutes=15),
        min_duration=timedelta(minutes=10),
        overlap=timedelta(minutes=1),
    )

    # Verify parquet created
    files = glob(f"{temp_dir}*/*.parquet")
    assert len(files) > 0
    assert files[0].endswith(".parquet")

    # Verify files
    for file in files:
        df = pd.read_parquet(file)
        assert df.index.min() >= start_date
        assert df.index.max() <= end_date + timedelta(minutes=15)
        assert df.index.max() - df.index.min() > timedelta(minutes=10)
        assert df.shape[0] > 0

    # Load dataset
    ds = load_radio_dataset(temp_dir)

    assert isinstance(ds, Dataset)
    assert len(ds) > 0
    assert list(ds[0].keys()) == ["image", "antenna", "datetime"]

    # Check image is PIL
    example = ds[0]
    assert isinstance(example["image"], PILImage.Image)

    # Check image size
    assert len(example["image"].size) == 2
    assert example["image"].size[0] >= 3000
    assert example["image"].size[1] >= 150

    # Clean up
    shutil.rmtree(temp_dir)
