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
    print(device_metrics)
    # Apply baseline correction to each ch1 column and update in the dataframe
    for device_name, device_df in filtered_device_df.items():
        for col in device_df.columns[1:]:
            device_df[col] = device_df[col] / 1000
            device_df[col] = baseline_correction(device_df[col])
    # Plot all ch1 acceleration data from all devices in one plot
    plt.figure(figsize=(10, 6))
    for device_name, device_df in filtered_device_df.items():
        for col in device_df.columns[1:]:
            plt.plot(device_df['Time'], device_df[col], label=f"{device_name} ({col})")
            plt.xlabel("Time (HH:MM)")
            plt.ylabel("Acceleration (Corrected)")
            plt.title(f"Baseline Corrected Acceleration Data for ch1 {device_name} (Excluding Device 57044)")
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


def calculate_rms_and_peak_to_peak_in_window(data: pd.DataFrame, start_time: pd.Timestamp, window_duration='1min'):
    """
    Calculate RMS and Peak-to-Peak values in a specific time window, adjusting for the nearest timestamp.

    Parameters:
    - data: DataFrame with 'Time' column and acceleration data.
    - start_time: Approximate starting timestamp of the window (datetime-like).
    - window_duration: Duration of the window (default: '1min').

    Returns:
    - window_metrics: Dictionary with RMS and Peak-to-Peak values for each axis.
    """
    # Find the closest time to the given start_time
    closest_time_idx = (data['Time'] - start_time).abs().idxmin()
    closest_time = data.loc[closest_time_idx, 'Time']

    print(f"Closest available start time to {start_time} is {closest_time}.")

    # Define the time window
    end_time = closest_time + pd.Timedelta(window_duration)
    window_data = data[(data['Time'] >= closest_time) & (data['Time'] < end_time)]

    # Calculate metrics for each column in the window
    window_metrics = {}
    for col in window_data.columns[1:]:
        rms, peak_to_peak = calculate_rms_and_peak_to_peak(window_data[col])
        window_metrics[col] = {'RMS': rms, 'Peak-to-Peak': peak_to_peak}

    return window_metrics, closest_time

def post_process_with_window():
    lord_file = os.path.join(DATA, 'lord\\3_t7.csv')
    df = lord_parser(lord_file)

    # Remove data for device 57044
    filtered_device_df, _ = filter_device_ch1_dataframes(df)

    # Example: Calculate metrics for a 1-minute window starting at an approximate time
    approximate_start_time = pd.Timestamp('2024-11-27 13:20:00')
    for device_name, device_df in filtered_device_df.items():
        window_metrics, actual_start_time = calculate_rms_and_peak_to_peak_in_window(device_df, approximate_start_time)
        print(f"Metrics for {device_name} in 1-minute window starting at {actual_start_time}:")
        for col, metrics in window_metrics.items():
            print(f"  {col} -> RMS: {metrics['RMS']:.3f}, Peak-to-Peak: {metrics['Peak-to-Peak']:.3f}")

if __name__ == '__main__':
    #post_process()
    post_process_with_window()
