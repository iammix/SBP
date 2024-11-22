import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
from scipy.signal import periodogram, butter, filtfilt

# Define the main data directory
data_directory = 'Data/T7/sbp'
save_directory = 'Data/T7/plots'
os.makedirs(save_directory, exist_ok=True)

def process():
    # Dictionary to store data for each device
    device_data = {}

    # Read data for each device
    for device_folder in os.listdir(data_directory):
        device_folder_path = os.path.join(data_directory, device_folder)
        device_df = pd.DataFrame()
        
        if os.path.isdir(device_folder_path):
            for file_name in os.listdir(device_folder_path):

                if file_name.endswith('.txt'):
                    file_path = os.path.join(device_folder_path, file_name)
                    
                    # Read the file and parse only the necessary columns
                    df = pd.read_csv(file_path, delimiter='\t',
                                     usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']  # Rename columns for ease of use
                    
                    # Convert time column to datetime format for easier filtering
                    df['time'] = pd.to_datetime(df['time'])

                    
                    # Convert acceleration from milli-g to g and apply baseline correction
                    df['acc_x'] = df['acc_x'] - df['acc_x'].mean()
                    df['acc_y'] = df['acc_y'] - df['acc_y'].mean()
                    df['acc_z'] = df['acc_z'] - df['acc_z'].mean()


                    device_df = pd.concat(device_df, df)
                    # Store the DataFrame in the dictionary with device name as key
                    device_data[device_folder] = device_df

    # Dictionary to store the global maximum values and corresponding device for each axis
    global_max_info = {}

    # Find the global maximum for each axis across all devices
    for axis in ['acc_x', 'acc_y', 'acc_z']:
        global_max_value = float('-inf')
        global_max_time = None
        global_max_device = None
        
        for device, df in device_data.items():
            # Get the maximum value and time for the current device and axis
            device_max_value = df[axis].max()
            device_max_time = df.loc[df[axis].idxmax(), 'time']
            
            # Update the global maximum if the device has a higher max
            if device_max_value > global_max_value:
                global_max_value = device_max_value
                global_max_time = device_max_time
                global_max_device = device
        
        # Store the global max info
        global_max_info[axis] = {
            'value': global_max_value,
            'time': global_max_time,
            'device': global_max_device
        }

    # Plot the data for each axis, using the global max time as reference and compute PSD
    for axis, max_info in global_max_info.items():
        max_time = max_info['time']
        
        # Plot time-domain data in the 2-minute window
        plt.figure(figsize=(12, 8))
        plt.title(f'2-minute window around global max {axis} acceleration {max_info["value"]:.4f}(g)')
        plt.xlabel('Time (hh:mm:ss)')
        plt.ylabel(f'{axis} acceleration (g)')
        
        for device, df in device_data.items():
            # Filter data to 1 minute before and after the global max time
            window_df = df[(df['time'] >= max_time - pd.Timedelta(minutes=1)) &
                           (df['time'] <= max_time + pd.Timedelta(minutes=1))]
            
            # Plot the time-domain data within this window
            plt.plot(window_df['time'], window_df[axis], linewidth=0.5, label=device)

        save_path = os.path.join(save_directory, f'global_max_{axis}_acceleration_rise.png')
        plt.legend()
        plt.savefig(save_path)
        plt.show()

        # Plot PSD for each device in the same window
        plt.figure(figsize=(12, 8))
        plt.title(f'PSD of 2-minute window around global max {axis} for all devices')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (g^2/Hz)')
        
        for device, df in device_data.items():
            # Extract signal for the 2-minute window
            signal = window_df[axis].values
            
            # Compute PSD using the periodogram method
            freqs, psd = periodogram(signal, fs=125)  # Adjust `fs` if necessary
            # Find the frequency with the highest amplitude
            max_psd_index = np.argmax(psd)
            max_psd_freq = freqs[max_psd_index]
            max_psd_value = psd[max_psd_index]
            plt.plot(freqs, psd, linewidth=0.5, label=f'{device}')
            # Add a vertical line for the max frequency and annotate
            plt.axvline(x=max_psd_freq, color='r', linestyle='--', linewidth=0.8)
            plt.text(max_psd_freq, max_psd_value, f'{max_psd_freq:.2f} Hz', color='red', 
                     ha='center', va='bottom')
        
        save_path_psd = os.path.join(save_directory, f'psd_global_max_{axis}.png')
        plt.legend()
        plt.savefig(save_path_psd)
        plt.show()

def find_and_plot_silent_interval(df, axis='acc_x', peak_threshold=0.2, window_minutes=2):
    """
    Finds a 2-minute interval without significant peaks and plots it.
    
    Parameters:
    - df: DataFrame containing time and acceleration columns.
    - axis: The axis to check for silence ('acc_x', 'acc_y', 'acc_z').
    - peak_threshold: The threshold above which values are considered as peaks.
    - window_minutes: Duration of the window to find (in minutes).
    """

    # Calculate window size in terms of number of samples (assuming data is recorded in regular intervals)
    window_size = pd.Timedelta(minutes=window_minutes)
    
    # Slide over the data with the defined window size
    for start_time in df['time']:
        # Define the end time of the window
        end_time = start_time + window_size
        
        # Filter the data within this time window
        window_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]


        if window_df[axis].abs().max() < peak_threshold:
            # If no significant peaks are found, plot this interval and return
            plt.figure(figsize=(10, 6))
            plt.plot(window_df['time'], window_df[axis], label=f'Silent interval on {axis}')
            plt.title(f'Silent 2-minute interval in {axis} without peaks > {peak_threshold} g')
            plt.xlabel('Time')
            plt.ylabel(f'{axis} acceleration (g)')
            plt.legend()
            plt.show()
            return

    print("No silent interval found within the specified criteria.")

def find_silent_intervals(device_data, axis='acc_x', peak_threshold=0.2, window_minutes=1):
    """
    Finds a 2-minute silent interval (without significant peaks) for each device and plots all in one figure.
    
    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - axis: The axis to check for silence ('acc_x', 'acc_y', 'acc_z').
    - peak_threshold: The threshold above which values are considered peaks.
    - window_minutes: Duration of the window to find (in minutes).
    """
    window_size = pd.Timedelta(minutes=window_minutes)
    silent_intervals = {}  # To store silent intervals for each device
    
    # Loop through each device to find a silent interval
    for device, df in device_data.items():
        # Slide over the data with the defined window size
        for start_time in df['time']:
            end_time = start_time + window_size
            # Filter data within this time window
            window_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
            # Check if this window has any peak above the threshold
            if window_df[axis].abs().max() < peak_threshold:
                silent_intervals[device] = window_df
                break  # Stop once a silent interval is found for this device

    # Plot all silent intervals in one plot
    plt.figure(figsize=(12, 8))
    plt.title(f'2-minute Silent Intervals in {axis} for All Devices (no peaks > {peak_threshold} g)')
    plt.xlabel('Time')
    plt.ylabel(f'{axis} acceleration (g)')

    for device, interval_df in silent_intervals.items():
        plt.plot(interval_df['time'], interval_df[axis], linewidth=0.5, label=device)

    plt.legend()
    plt.show()

def find_common_silent_interval(device_data, axis, peak_threshold=0.2, window_minutes=1):
    """
    Finds a common 2-minute silent interval without significant peaks across all devices and plots it.
    
    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - axis: The axis to check for silence ('acc_x', 'acc_y', 'acc_z').
    - peak_threshold: The threshold above which values are considered as peaks.
    - window_minutes: Duration of the window to find (in minutes).
    """
    window_size = pd.Timedelta(minutes=window_minutes)
    
    # Find the earliest and latest time range across all devices
    start_time = max(df['time'].min() for df in device_data.values())
    end_time = min(df['time'].max() for df in device_data.values())
    
    # Slide the 2-minute window across the common time range
    current_time = start_time
    while current_time + window_size <= end_time:
        common_silent = True  # Assume it's a silent interval until proven otherwise
        for device, df in device_data.items():
            # Filter data within this time window for the current device
            window_df = df[(df['time'] >= current_time) & (df['time'] < current_time + window_size)]
            # Check if this window has any peak above the threshold
            if window_df[axis].abs().max() >= peak_threshold:
                common_silent = False
                break  # No need to check other devices if one has a peak

        # If all devices have no peaks above the threshold, this is a common silent interval
        if common_silent:
            plt.figure(figsize=(12, 8))
            plt.title(f'Common 1-minute Silent Interval in {axis} for SBP Devices')
            plt.xlabel('Time (hh:mm:sec)')
            plt.ylabel(f'{axis} acceleration (g)')
            
            # Plot each device's data within the common silent interval
            for device, df in device_data.items():
                window_df = df[(df['time'] >= current_time) & (df['time'] < current_time + window_size)]
                plt.plot(window_df['time'], window_df[axis], linewidth=0.5, label=device)
            
            plt.legend()
            # Save the plot to a file
            save_path = os.path.join(save_directory, f'common_silent_interval_{axis}.png')
            plt.savefig(save_path)
            plt.show()
            return  # Stop after finding and plotting the first common silent interval
        
        # Move the window by a small increment (e.g., 1 second)
        current_time += pd.Timedelta(seconds=1)

    print("No common silent interval found within the specified criteria.")

def analyze_silent_interval(device_data, axis='acc_x', peak_threshold=0.2, window_minutes=2, save_directory='plots'):
    """
    Finds a common 2-minute silent interval without significant peaks across all devices,
    calculates the PSD using the periodogram, peak-to-peak, and RMS, and plots them.

    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - axis: The axis to analyze ('acc_x', 'acc_y', 'acc_z').
    - peak_threshold: The threshold above which values are considered as peaks.
    - window_minutes: Duration of the window to find (in minutes).
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)
    window_size = pd.Timedelta(minutes=window_minutes)

    # Find the common time range
    start_time = max(df['time'].min() for df in device_data.values())
    end_time = min(df['time'].max() for df in device_data.values())
    
    current_time = start_time
    while current_time + window_size <= end_time:
        common_silent = True
        for device, df in device_data.items():
            window_df = df[(df['time'] >= current_time) & (df['time'] < current_time + window_size)]
            if window_df[axis].abs().max() >= peak_threshold:
                common_silent = False
                break

        if common_silent:
            # Initialize lists to store PSD, P2P, and RMS data
            psd_data = {}
            p2p_data = {}
            rms_data = {}

            # Plot PSD for each device and calculate P2P and RMS
            plt.figure(figsize=(12, 8))
            plt.title(f'PSD of Silent 2-Minute Interval in {axis} for All Devices (no peaks > {peak_threshold} g)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (g^2/Hz)')

            for device, df in device_data.items():
                # Extract the silent interval data
                window_df = df[(df['time'] >= current_time) & (df['time'] < current_time + window_size)]
                signal = window_df[axis].values

                # Compute PSD using the periodogram method
                freqs, psd = periodogram(signal, fs=125)
                psd_data[device] = (freqs, psd)

                # Calculate peak-to-peak (P2P) and RMS
                p2p_data[device] = np.ptp(signal)
                rms_data[device] = np.sqrt(np.mean(signal**2))

                # Plot PSD
                plt.plot(freqs, psd, linewidth=0.5, label=f'{device}')

            plt.legend()
            plt.savefig(os.path.join(save_directory, f'common_silent_psd_{axis}.png'))
            plt.show()

            # Display P2P and RMS values
            print(f"Peak-to-Peak and RMS for silent interval in {axis}:")
            for device in device_data.keys():
                print(f"Device: {device}")
                print(f"  Peak-to-Peak (P2P): {p2p_data[device]*1000:.6f} mg")
                print(f"  RMS: {rms_data[device]*1000:.6f} mg")

            return

        current_time += pd.Timedelta(seconds=1)

    print("No common silent interval found within the specified criteria.")

def plot_full_acceleration(device_data, save_directory='plots'):
    """
    Plots the entire acceleration time series for each device.

    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    for axis in ['acc_x', 'acc_y', 'acc_z']:
        plt.figure(figsize=(12, 8))
        plt.title(f'Full Acceleration Time Series for {axis} Across All Devices')
        plt.xlabel('Time')
        plt.ylabel(f'{axis} acceleration (g)')
        
        # Plot the full time series for each device for the current axis
        for device, df in device_data.items():
            plt.plot(df['time'], df[axis], label=device, linewidth=0.5)

        # Display legend and save plot
        plt.legend()
        save_path = os.path.join(save_directory, f'full_acceleration_{axis}.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Plot saved to {save_path}")

def plot_ssta_slta_ratio_all_devices(device_data, axis='acc_x', nSTA=125*3, nLTA=125*30, save_directory='plots'):
    """
    Calculates and plots the ratio sSTA/sLTA for all devices together in a given direction.

    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - axis: The axis to analyze ('acc_x', 'acc_y', 'acc_z').
    - nSTA: Number of samples for the short time window (sSTA).
    - nLTA: Number of samples for the long time window (sLTA).
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(14, 8))

    for device, df in device_data.items():
        signal_squared = df[axis]**2  # Squared acceleration signal

        # Initialize sSTA and sLTA
        sSTA = [signal_squared.iloc[0]]  # Start with the first value
        sLTA = [signal_squared.iloc[0]]

        # Calculate sSTA
        for i in range(1, len(signal_squared)):
            if i < nSTA:
                sSTA.append(sSTA[-1] + (signal_squared.iloc[i] - signal_squared.iloc[i - 1]) / (i + 1))
            else:
                sSTA.append(sSTA[-1] + (signal_squared.iloc[i] - signal_squared.iloc[i - nSTA]) / nSTA)

        # Calculate sLTA
        for i in range(1, len(signal_squared)):
            if i < nLTA:
                sLTA.append(sLTA[-1] + (signal_squared.iloc[i] - signal_squared.iloc[i - 1]) / (i + 1))
            else:
                sLTA.append(sLTA[-1] + (signal_squared.iloc[i] - signal_squared.iloc[i - nLTA]) / nLTA)

        # Convert lists to numpy arrays for plotting
        sSTA = np.sqrt(sSTA)
        sLTA = np.sqrt(sLTA)

        # Calculate sSTA/sLTA ratio
        ratio = np.array(sSTA) / np.array(sLTA)

        # Plot the ratio for this device
        plt.plot(df['time'], ratio, label=f'{device} sSTA/sLTA', linewidth=0.5)

    plt.title(f'sSTA/sLTA Ratio for {axis} - All Devices')
    plt.xlabel('Time')
    plt.ylabel('sSTA/sLTA Ratio')
    plt.legend()
    plt.grid()

    # Save the plot
    save_path = os.path.join(save_directory, f'ssta_slta_ratio_all_devices_{axis}.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

def get_data():
    device_data = {}

    # Read data for each device
    for device_folder in os.listdir(data_directory):
        device_folder_path = os.path.join(data_directory, device_folder)
        device_df = pd.DataFrame()

        if os.path.isdir(device_folder_path):
            for file_name in os.listdir(device_folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(device_folder_path, file_name)
                    # Read the file and parse only the necessary columns
                    #df = pd.read_fwf(file_path, usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df = pd.read_csv(file_path, delimiter='\t',
                                usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']  # Rename columns for ease of use

                    # Convert time column to datetime format for easier filtering
                    df['time'] = pd.to_datetime(df['time'])
                    # Convert acceleration from milli-g to g
                    df['acc_x'] = df['acc_x'] / 1000
                    df['acc_y'] = df['acc_y'] / 1000
                    df['acc_z'] = df['acc_z'] / 1000
                    # Apply baseline correction by subtracting the mean of each axis
                    df['acc_x'] = df['acc_x'] - df['acc_x'].mean()
                    df['acc_y'] = df['acc_y'] - df['acc_y'].mean()
                    df['acc_z'] = df['acc_z'] - df['acc_z'].mean()

                    # Store the DataFrame in the dictionary with device name as key
                    device_df = pd.concat([device_df, df], axis=0, ignore_index=True)
                    device_data[device_folder] = device_df
    
    return device_data


def butterworth_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a 4th-order Butterworth filter to the data.
    
    Parameters:
    - data: The signal to filter.
    - lowcut: Low cutoff frequency (Hz).
    - highcut: High cutoff frequency (Hz).
    - fs: Sampling frequency (Hz).
    - order: Order of the filter.
    
    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


if __name__ == '__main__':
    device_data = get_data()



    #plot_full_acceleration(device_data)
    # find_common_silent_interval(device_data, axis='acc_x', peak_threshold=0.001, window_minutes=1)
    # find_common_silent_interval(device_data, axis='acc_y', peak_threshold=0.001, window_minutes=1)
    # find_common_silent_interval(device_data, axis='acc_z', peak_threshold=0.001, window_minutes=1)
    # analyze_silent_interval(device_data, axis='acc_x',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    # analyze_silent_interval(device_data, axis='acc_y',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    # analyze_silent_interval(device_data, axis='acc_z',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    plot_ssta_slta_ratio_all_devices(device_data, axis='acc_x', nSTA=125*3, nLTA=125*30, save_directory='plots')
    plot_ssta_slta_ratio_all_devices(device_data, axis='acc_y', nSTA=125*3, nLTA=125*30, save_directory='plots')
    plot_ssta_slta_ratio_all_devices(device_data, axis='acc_z', nSTA=125*3, nLTA=125*30, save_directory='plots')
