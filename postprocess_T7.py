import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DEVICE = {"RISE_1": "57044", "RISE_3": "57030", "RISE_5": "57045", "RISE_7": "57047", "RISE_9": "57046",
          "RISE_11": "57029", "RISE_13": "57028"}

DATA = 'Data/T7'

HEADER = ["Time", "57028:ch1", "57028:ch2", "57028:ch3", "57029:ch1", "57029:ch2", "57030:ch1", "57030:ch2",
          "57044:ch1", "57044:ch2", "57045:ch1", "57045:ch2", "57047:ch2", "57047:ch1", "57046:ch2", "57046:ch1",
          "57045:ch3", "57044:ch3", "57029:ch3"]

def lord_parser(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, skiprows=36, delimiter=';', names=HEADER, header=0)
    df['Time'] = pd.to_datetime(df['Time']) + pd.Timedelta(hours=3)  # Add 3 hours to the 'Time' column
    return df

def baseline_correction(data: pd.Series) -> pd.Series:
    return data - data.mean()

def post_process():
    lord_file = os.path.join(DATA, 'lord\\3_t7.csv')
    df = lord_parser(lord_file)
    # Remove data for device 57044
    filtered_device_df, device_metrics = filter_device_ch1_dataframes(df)
    # Apply baseline correction to each ch1 column and update in the dataframe
    for device_name, device_df in filtered_device_df.items():
        for col in device_df.columns[1:]:
            device_df[col] = baseline_correction(device_df[col])
    # Plot all ch1 acceleration data from all devices in one plot
    plt.figure(figsize=(10, 6))
    for device_name, device_df in filtered_device_df.items():
        for col in device_df.columns[1:]:
            plt.plot(device_df['Time'], device_df[col], label=f"{device_name} ({col})")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Acceleration (Corrected)")
    plt.title("Baseline Corrected Acceleration Data for ch1 Across Devices (Excluding Device 57044)")
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

def calculate_rms_and_peak_to_peak(data):
    rms = np.sqrt(np.mean(data ** 2))
    peak_to_peak = data.max() - data.min()
    return rms, peak_to_peak

def filter_device_ch1_dataframes(df: pd.DataFrame) -> dict:
    ch1_device_dfs = {}
    device_metrics = {}

    for device_name, device_code in DEVICE.items():
        # Skip device 57044
        if device_code == "57044":
            continue

        # Find columns for the specific device that end with ":ch1"
        ch1_columns = [col for col in df.columns if col.startswith(device_code) and col.endswith(':ch1')]

        # Only create a dataframe if there are any ch1 columns for the device
        if ch1_columns:
            ch1_device_df = df[['Time'] + ch1_columns].copy()
            ch1_device_dfs[device_name] = ch1_device_df

            # Calculate RMS and Peak-to-Peak for each ch1 column
            metrics = {}
            for col in ch1_columns:
                rms, peak_to_peak = calculate_rms_and_peak_to_peak(ch1_device_df[col])
                metrics[col] = {'RMS': rms, 'Peak-to-Peak': peak_to_peak}
            device_metrics[device_name] = metrics
    return ch1_device_dfs, device_metrics


if __name__ == '__main__':
    post_process()
