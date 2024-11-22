import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fft import fft, fftfreq

DEVICE = {"RISE_1": "57044", "RISE_3": "57030", "RISE_5": "57045", "RISE_7": "57047", "RISE_9": "57046",
          "RISE_11": "57029", "RISE_13": "57028"}

DATA = 'Data/T7'


def all_sbp_parser(folder_path: str) -> pd.DataFrame:
    all_data = []
    for sbp_device in DEVICE.keys():
        device_files = [f for f in os.listdir(folder_path)]
        for file in device_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, delimiter='\t',
                             names=['Date', 'Time', 'Seconds_zeroed', 'Seconds_synced', 'Acc_x', 'Acc_y', 'Acc_z'],
                             skiprows=1)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
            df.drop(columns=['Date', 'Time'], inplace=True)
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def all_lord_parser(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, skiprows=36, delimiter=';', header=0)
    df['Time'] = pd.to_datetime(df['Time']) + timedelta(hours=3)

    lord_data = []
    for device_id in DEVICE.values():
        acc_columns = [col for col in df.columns if col.startswith(device_id)]
        device_data = pd.DataFrame()
        device_data['DateTime'] = df['Time']
        if f"{device_id}:ch1" in acc_columns:
            device_data['Acc_x'] = df[f"{device_id}:ch1"]
        if f"{device_id}:ch2" in acc_columns:
            device_data['Acc_y'] = df[f"{device_id}:ch2"]
        if f"{device_id}:ch3" in acc_columns:
            device_data['Acc_z'] = df[f"{device_id}:ch3"]
        lord_data.append(device_data)

    combined_lord_df = pd.concat(lord_data, ignore_index=True)
    return combined_lord_df


def find_max_accelerations_sbp():
    sbp_folder = os.path.join(DATA, 'sbp')
    combined_sbp_df = all_sbp_parser(sbp_folder)

    max_x = combined_sbp_df.loc[combined_sbp_df['Acc_x'].abs().idxmax()]
    max_y = combined_sbp_df.loc[combined_sbp_df['Acc_y'].abs().idxmax()]
    max_z = combined_sbp_df.loc[combined_sbp_df['Acc_z'].abs().idxmax()]

    max_values = {
        'x': {'value': max_x['Acc_x'], 'timestamp': max_x['DateTime']},
        'y': {'value': max_y['Acc_y'], 'timestamp': max_y['DateTime']},
        'z': {'value': max_z['Acc_z'], 'timestamp': max_z['DateTime']}
    }
    return max_values


def extract_time_window(df: pd.DataFrame, center_time: pd.Timestamp, window=timedelta(minutes=1)) -> pd.DataFrame:
    start_time = center_time - window
    end_time = center_time + window
    return df[(df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)]


def plot_data_and_fourier(data: pd.DataFrame, axis: str, device_type: str, max_time: pd.Timestamp):
    if data.empty:
        print(f"No data available for {device_type} in {axis}-axis within the 1-minute window around {max_time}.")
        plt.figure(figsize=(12, 6))
        plt.title(f"{device_type} {axis.upper()}-axis - No Data Available")
        plt.legend([f"No {device_type} data in time window"], loc="upper right")
        plt.show()
    else:
        time_deltas = (data['DateTime'] - data['DateTime'].iloc[0]).dt.total_seconds()

        # Time Series Plot
        plt.figure(figsize=(12, 6))
        plt.plot(data['DateTime'], data[f'Acc_{axis}'], label=f'{device_type} {axis.upper()}-axis')
        plt.title(f"{device_type} {axis.upper()}-axis Time Series")
        plt.xlabel("Time")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # Fourier Transform Plot
        plt.figure(figsize=(12, 6))
        N = len(data[f'Acc_{axis}'].dropna())
        yf = fft(data[f'Acc_{axis}'].dropna().values)
        xf = fftfreq(N, d=np.mean(np.diff(time_deltas)))[:N // 2]
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        plt.title(f"{device_type} {axis.upper()}-axis Fourier Transform")
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, 20)
        plt.ylabel("Amplitude")
        plt.show()


def analyze_max_accelerations():
    # Get SBP max values and timestamps
    sbp_max_values = find_max_accelerations_sbp()
    sbp_folder = os.path.join(DATA, 'sbp')
    sbp_df = all_sbp_parser(sbp_folder)
    lord_file = os.path.join(DATA, 'lord', '3_t7.csv')
    lord_df = all_lord_parser(lord_file)

    for axis, data in sbp_max_values.items():
        max_time = data['timestamp']
        print(f"Analyzing {axis.upper()}-axis for max SBP acceleration at {max_time}.")

        # Get 1-minute window for SBP data
        sbp_window = extract_time_window(sbp_df, max_time)
        plot_data_and_fourier(sbp_window, axis, "SBP", max_time)

        # Get 1-minute window for LORD data
        lord_window = extract_time_window(lord_df, max_time)
        plot_data_and_fourier(lord_window, axis, "LORD", max_time)


if __name__ == '__main__':
    analyze_max_accelerations()