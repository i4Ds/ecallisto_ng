import os
import sys
import torch
import mlflow
import torchvision
import numpy as np
import ecallisto_ng
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from ecallisto_ng.plotting.plotting import plot_spectogram_mpl
from ecallisto_ng.data_download.downloader import get_ecallisto_data


class FlareDetection:
    """
    Class for detecting flares in solar radio data using a pre-trained deep learning model.

    Args:
        model_id (str): The unique identifier of the pre-trained model to use.
        standard_cnn (bool, optional): Turn true, if loaded model is not a custom model. Default is True.
        device (str, optional): The device to run the model on, one of ["cuda", "mps", "cpu"]. Default is auto-detected.

    Attributes:
        device (str): The device the model is running on.
        standard_cnn (bool): True if using a standard CNN model, False if using a custom model.
        model (torch.nn.Module): The loaded pre-trained PyTorch model for flare detection.

    """

    def __init__(
        self,
        model_id="a853ec9b54244b4ab37dce5498597fd3",
        device=None,
        standard_cnn=True,
        tracking_uri="https://dagshub.com/FlareSense/Flaresense.mlflow",
    ):
        """
        Prepares and initializes an instance of the class.

        Args:
            model_id (str): The ID of the model to be loaded. Defaults to "33ad4d2d26fe416db7dd9eda2e4fd2fb".
            standard_cnn (bool): Whether to use the standard CNN architecture. Defaults to True.
            device: The device to be used for model execution. If not specified, "cuda" is used if available, "mps" if supported, else "cpu".
        """
        if device is None:
            # If no device is specified, use "cuda" if available, else "mps" if supported, else "cpu"
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.standard_cnn = standard_cnn

        # Add ecallisto_ng to the path, because of relative imports in the .pth files
        package_dir = os.path.dirname(ecallisto_ng.__file__)
        desired_dir = os.path.dirname(package_dir)  # src/ecallisto_ng
        desired_dir = os.path.dirname(desired_dir)  # src
        sys.path.append(desired_dir)

        # Load the PyTorch model using MLflow and move it to the specified device
        mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow.pytorch.load_model(f"runs:/{model_id}/model/").to(self.device)
        self.model.eval()

    def preprocess(self, image, length=[224, 224]):
        """
        Preprocess input solar radio data for prediction.

        Args:
            image (pd.DataFrame): Solar radio data as a pandas DataFrame.
            length (list, optional): Desired length for the preprocessed data. Default is [224, 224].

        Returns:
            torch.Tensor: Preprocessed image data as a torch.Tensor.

        """
        length_s = torch.tensor((image.index.max() - image.index.min()).total_seconds())
        resampling_s = torch.round((length_s / length[0])).int().item()
        image = image.resample(f"{resampling_s}s").max()
        image = image.interpolate(method="linear", limit_direction="both")
        image = image.values.T
        image = torch.tensor(image).float()
        return image

    def detect(self, datetime, instrument="Australia-ASSA_62", window_length="1h"):
        """
        Detect flares in solar radio data for a given datetime, instrument, and window length.

        Args:
            datetime (str or pd.Timestamp): Date and time to start flare detection.
            instrument (str, optional): Name of the radio instrument. Default is "Australia-ASSA_62".
            window_length (str, optional): Length of the time window for detection. Default is "1h".

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: Solar radio data for the specified time range and instrument.
                - dict: Flare probability predictions for each minute.
                - matplotlib.figure.Figure: Matplotlib figure showing the data and flare probability.

        Raises:
            ValueError: If the window_length is less than 15 minutes.
            ValueError: If the end_time is in another day.

        """
        datetime = pd.to_datetime(datetime)
        window_length = pd.to_timedelta(window_length)

        # Check if the window length is at least 15 minutes, models are trained on 15 minute windows
        if window_length < pd.to_timedelta("30min"):
            raise ValueError("window_length must be at least 30min")

        start_time = datetime
        end_time = datetime + window_length
        print(f"Detecting flares from {start_time} to {end_time} on {instrument}")

        # If end_time is in another day, throw an error (allow exactly 00:00:00)
        if end_time.day != start_time.day and end_time != pd.to_datetime(end_time.date()):
            raise ValueError("API currently does not handle data crossing the midnight boundary")

        # Get data for the specified time range and instrument
        data = get_ecallisto_data(start_time, end_time, instrument)
        if data == {}:
            print(f"No data found for {start_time} - {end_time} on {instrument}")
            return None
        data = data[instrument]

        # Generate 15min windows of data
        img_tensor = []
        n_predictions = int((window_length.total_seconds() // 60) - 15) + 1
        for i in range(n_predictions):
            delta_start = pd.to_timedelta(i, unit="m")
            delta_end = pd.to_timedelta(i + 15, unit="m")

            data_temp = data.copy()
            data_temp = data_temp.loc[start_time + delta_start : start_time + delta_end]

            # Preprocess the data for prediction
            data_temp = self.preprocess(data_temp)
            data_temp = data_temp.unsqueeze(0)
            img_tensor.append(data_temp)

        # Stack the 15min windows into a single tensor and resize to 224x224
        img_tensor = torch.stack(img_tensor).to(self.device)
        img_tensor = torchvision.transforms.functional.resize(img_tensor, [224, 224], antialias=True)

        # If the model is not a custom model, expand the tensor to 3 color channels
        if self.standard_cnn:
            img_tensor = img_tensor.expand(-1, 3, -1, -1)

        # Make predictions on the data
        with torch.no_grad():
            predictions = self.model(img_tensor)
        predictions = predictions.flatten().cpu().numpy()

        # Average the predictions for each minute
        minute_by_minute = defaultdict(list)
        for i in range(n_predictions):
            for j in range(15):
                minute_by_minute[i + j].append(predictions[i])
        minute_by_minute = {k: sum(v) / len(v) for k, v in minute_by_minute.items()}

        # Plot the data
        fig = plot_spectogram_mpl(data, fig_size=(12, 6))

        # Get the current axis
        ax1 = fig.gca()

        # Plot the flare probability on top of the data
        x = np.linspace(0, data.shape[0], len(minute_by_minute))
        y = np.array(list(minute_by_minute.values())) * data.shape[1]
        ax1.plot(x, y, color="red", linewidth=2, label="Flare Certainty")
        ax1.legend(loc="upper right")

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Set the limits for the secondary y-axis (0-100%)
        ax2.set_ylim(0, 100)
        ticks = np.linspace(0, 100, 11)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels([f"{int(t)}%" for t in ticks])
        ax2.tick_params(axis="y", labelsize=8, colors="red")

        # Show the plot with a tight layout
        plt.tight_layout()

        # Get the figure
        fig = plt.gca().get_figure()

        return data, minute_by_minute, fig
