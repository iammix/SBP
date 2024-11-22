import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta

# DEVICE dictionary with mappings
DEVICE = {"RISE_1": "57044", "RISE_3": "57030", "RISE_5": "57045", "RISE_7": "57047", "RISE_9": "57046",
          "RISE_11": "57029", "RISE_13": "57028"}

DATA = 'Data/T7'


def lord_parser(filename: str) -> pd.DataFrame:
    # Parses LORD device data
    df = pd.read_csv(filename, skiprows=36, delimiter=';', header=0)
    df['Time'] = pd.to_datetime(df['Time']) + timedelta(hours=3)
    count_per_second = df['Time'].dt.floor('S').value_counts().sort_index()
    print(f"LORD data points per second for file '{filename}': {count_per_second}")

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
            #
            # count_per_second = df['DateTime'].dt.floor('S').value_counts().sort_index()
            # print(f"SBP data points per second for file '{file}': {count_per_second}")



            dfs.append(df)
        sbp_data[sbp_device] = pd.concat(dfs, ignore_index=True)
    return sbp_data


def find_common_seconds(lord_df: pd.DataFrame, sbp_df: pd.DataFrame) -> pd.Series:
    # Identify seconds with data in both LORD and SBP DataFrames
    lord_seconds = lord_df['Time'].dt.floor('S').unique()
    sbp_seconds = sbp_df['DateTime'].dt.floor('S').unique()
    common_seconds = np.intersect1d(lord_seconds, sbp_seconds)
    return pd.to_datetime(common_seconds)


def plot_milliseconds(lord_df: pd.DataFrame, sbp_df: pd.DataFrame, common_seconds: pd.Series):
    for second in common_seconds:
        # Filter data within each common second
        lord_window = lord_df[(lord_df['Time'] >= second) & (lord_df['Time'] < second + timedelta(seconds=1))]
        sbp_window = sbp_df[(sbp_df['DateTime'] >= second) & (sbp_df['DateTime'] < second + timedelta(seconds=1))]

        plt.figure(figsize=(12, 6))

        # Plot LORD device data with millisecond precision
        for col in lord_window.columns[1:]:  # Skip the 'Time' column
            plt.plot(lord_window['Time'], lord_window[col], label=f'LORD {col}')

        # Plot SBP device data with millisecond precision
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_x'], label='SBP Acc_x', linestyle='--')
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_y'], label='SBP Acc_y', linestyle='--')
        plt.plot(sbp_window['DateTime'], sbp_window['Acc_z'], label='SBP Acc_z', linestyle='--')

        plt.xlabel("Time (ms within second)")
        plt.ylabel("Acceleration")
        plt.title(f"Device Data from {second}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()


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

        # Find common seconds
        common_seconds = find_common_seconds(lord_device_df, sbp_df)

        if not common_seconds.empty:
            print(f"Plotting data for common seconds between {sbp_device} and LORD {lord_device}")
            plot_milliseconds(lord_device_df, sbp_df, common_seconds)
        else:
            print(f"No common seconds found for {sbp_device} and LORD {lord_device}")


def all_sbp_parser(folder_path: str) -> pd.DataFrame:
    # Parses SBP device data files and combines them into one DataFrame
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

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def find_max_accelerations_sbp():
    # Load all SBP device data into one combined DataFrame
    sbp_folder = os.path.join(DATA, 'sbp')
    combined_sbp_df = all_sbp_parser(sbp_folder)

    # Find the max absolute acceleration for each axis and corresponding timestamp
    max_x = combined_sbp_df.loc[combined_sbp_df['Acc_x'].abs().idxmax()]
    max_y = combined_sbp_df.loc[combined_sbp_df['Acc_y'].abs().idxmax()]
    max_z = combined_sbp_df.loc[combined_sbp_df['Acc_z'].abs().idxmax()]

    print("Maximum Accelerations:")
    print(f"X-Axis: {max_x['Acc_x']} at {max_x['DateTime']}")
    print(f"Y-Axis: {max_y['Acc_y']} at {max_y['DateTime']}")
    print(f"Z-Axis: {max_z['Acc_z']} at {max_z['DateTime']}")


def all_lord_parser(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, skiprows=36, delimiter=';', header=0)
    df['Time'] = pd.to_datetime(df['Time']) + timedelta(hours=3)

    # Identify available channels
    lord_data = []
    for device_id in DEVICE.values():
        acc_columns = [col for col in df.columns if col.startswith(device_id)]

        # Assign x, y, z based on available channels for each device
        device_data = pd.DataFrame()
        device_data['Time'] = df['Time']
        if f"{device_id}:ch1" in acc_columns:
            device_data['Acc_x'] = df[f"{device_id}:ch1"]
        if f"{device_id}:ch2" in acc_columns:
            device_data['Acc_y'] = df[f"{device_id}:ch2"]
        if f"{device_id}:ch3" in acc_columns:
            device_data['Acc_z'] = df[f"{device_id}:ch3"]

        lord_data.append(device_data)

    # Combine all device data into one DataFrame for analysis
    combined_lord_df = pd.concat(lord_data, ignore_index=True)
    return combined_lord_df

def find_max_accelerations_lord():
    lord_file = os.path.join(DATA, 'lord', '3_t7.csv')
    lord_df = all_lord_parser(lord_file)
    # Find max absolute acceleration for each axis and corresponding timestamp
    max_x = lord_df.loc[lord_df['Acc_x'].abs().idxmax()]
    max_y = lord_df.loc[lord_df['Acc_y'].abs().idxmax()]
    max_z = lord_df.loc[lord_df['Acc_z'].abs().idxmax()]

    print(f"Maximum Accelerations for LORD:")
    print(f"X-Axis: {max_x['Acc_x']} at {max_x['Time']}")
    print(f"Y-Axis: {max_y['Acc_y']} at {max_y['Time']}")
    print(f"Z-Axis: {max_z['Acc_z']} at {max_z['Time']}\n")


if __name__ == '__main__':
    find_max_accelerations_lord()
