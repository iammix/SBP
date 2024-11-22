import os
import pandas as pd
import numpy as np
from datetime import timedelta

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

    combined_lord_df = pd.concat(lord_data, ignore_index=True)
    return combined_lord_df

def find_max_accelerations_sbp():
    sbp_folder = os.path.join(DATA, 'sbp')
    combined_sbp_df = all_sbp_parser(sbp_folder)

    # Find the max absolute acceleration for each axis and corresponding timestamp
    max_x = combined_sbp_df.loc[combined_sbp_df['Acc_x'].abs().idxmax()]
    max_y = combined_sbp_df.loc[combined_sbp_df['Acc_y'].abs().idxmax()]
    max_z = combined_sbp_df.loc[combined_sbp_df['Acc_z'].abs().idxmax()]

    max_values = {
        'x': {'value': max_x['Acc_x'], 'timestamp': max_x['DateTime']},
        'y': {'value': max_y['Acc_y'], 'timestamp': max_y['DateTime']},
        'z': {'value': max_z['Acc_z'], 'timestamp': max_z['DateTime']}
    }
    return max_values

def find_closest_lord_timestamp(lord_df: pd.DataFrame, sbp_timestamp: pd.Timestamp):
    # Filter LORD data to the same second as SBP timestamp
    lord_same_second = lord_df[(lord_df['Time'] >= sbp_timestamp.floor('S')) &
                               (lord_df['Time'] < sbp_timestamp.floor('S') + timedelta(seconds=1))]

    if not lord_same_second.empty:
        # Calculate the absolute time difference and find the closest one
        lord_same_second['Time_Diff'] = (lord_same_second['Time'] - sbp_timestamp).abs()
        closest_row = lord_same_second.loc[lord_same_second['Time_Diff'].idxmin()]
        return closest_row['Time'], closest_row[['Acc_x', 'Acc_y', 'Acc_z']]
    else:
        return None, None

def find_max_and_closest():
    # Process SBP max values
    sbp_max_values = find_max_accelerations_sbp()
    lord_file = os.path.join(DATA, 'lord', '3_t7.csv')
    lord_df = all_lord_parser(lord_file)

    # Check LORD device data for the closest values to SBP max timestamps
    for axis, data in sbp_max_values.items():
        sbp_timestamp = data['timestamp']
        print(f"SBP max {axis.upper()}-axis acceleration at {sbp_timestamp} with value {data['value']}")

        # Find closest LORD timestamp to the SBP maximum timestamp for each axis
        closest_time, closest_values = find_closest_lord_timestamp(lord_df, sbp_timestamp)
        if closest_time is not None:
            print(f"Closest LORD timestamp to SBP {axis.upper()}-axis max: {closest_time}")
            print(f"LORD Acceleration at closest time: {closest_values.to_dict()}\n")
        else:
            print(f"No LORD data found within the same second as the SBP {axis.upper()}-axis max.\n")

if __name__ == '__main__':
    find_max_and_closest()