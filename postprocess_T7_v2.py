import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta

# Existing DEVICE dictionary
DEVICE = {"RISE_1": "57044", "RISE_3": "57030", "RISE_5": "57045", "RISE_7": "57047", "RISE_9": "57046",
          "RISE_11": "57029", "RISE_13": "57028"}

DATA = 'Data/T7'


def lord_parser(filename: str) -> pd.DataFrame:
    # Parses LORD device data
    df = pd.read_csv(filename, skiprows=36, delimiter=';', header=0)
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def sbp_parser(folder_path: str) -> dict:
    # Parses SBP device data files
    sbp_data = {}
    for sbp_device in DEVICE.keys():
        device_files = [f for f in os.listdir(folder_path)]
        dfs = []
        for file in device_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, delimiter='\t',
                             names=['Date', 'Time', 'Seconds_zeroed', 'Seconds_synced', 'Acc_x', 'Acc_y', 'Acc_z'],
                             skiprows=1)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
            df.drop(columns=['Date', 'Time'], inplace=True)
            dfs.append(df)
        sbp_data[sbp_device] = pd.concat(dfs, ignore_index=True)
    return sbp_data


def find_overlap(lord_df: pd.DataFrame, sbp_df: pd.DataFrame) -> (pd.Timestamp, pd.Timestamp):
    # Finds the overlapping time range
    start_time = max(lord_df['Time'].min(), sbp_df['DateTime'].min())
    end_time = min(lord_df['Time'].max(), sbp_df['DateTime'].max())
    return start_time, end_time


def plot_in_windows(lord_df: pd.DataFrame, sbp_df: pd.DataFrame, start_time, end_time, window_size=5):
    current_time = start_time
    while current_time + timedelta(seconds=window_size) <= end_time:
        window_end = current_time + timedelta(seconds=window_size)

        lord_window = lord_df[(lord_df['Time'] >= current_time) & (lord_df['Time'] < window_end)]
        sbp_window = sbp_df[(sbp_df['DateTime'] >= current_time) & (sbp_df['DateTime'] < window_end)]

        plt.figure(figsize=(12, 6))

        # Plot LORD device data
        for col in lord_window.columns[1:]:  # Skip the 'Time' column
            plt.plot(lord_window['Time'], lord_window[col], label=f'LORD {col}')

        # Plot SBP device data
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_x'], label='SBP Acc_x', linestyle='--')
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_y'], label='SBP Acc_y', linestyle='--')
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_z'], label='SBP Acc_z', linestyle='--')

        plt.xlabel("Time")
        plt.ylabel("Acceleration")
        plt.title(f"Device Data from {current_time} to {window_end}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        current_time += timedelta(seconds=window_size)


def post_process():
    # Load LORD device data
    lord_file = os.path.join(DATA, 'lord\\3_t7.csv')
    lord_df = lord_parser(lord_file)

    # Load SBP device data
    sbp_folder = os.path.join(DATA, 'sbp')
    sbp_data = sbp_parser(sbp_folder)

    # Process data for each SBP and its corresponding LORD device
    for sbp_device, sbp_df in sbp_data.items():
        lord_device = DEVICE[sbp_device]

        # Extract LORD data for this device
        lord_device_df = lord_df[[col for col in lord_df.columns if col.startswith(lord_device)]]
        lord_device_df['Time'] = lord_df['Time']

        # Find overlapping period
        start_time, end_time = find_overlap(lord_device_df, sbp_df)

        if start_time and end_time and start_time < end_time:
            print(f"Overlapping period for {sbp_device} and LORD {lord_device}: {start_time} to {end_time}")
            plot_in_windows(lord_device_df, sbp_df, start_time, end_time, window_size=5)
        else:
            print(f"No overlapping period found for {sbp_device} and LORD {lord_device}")


if __name__ == '__main__':
    post_process()
