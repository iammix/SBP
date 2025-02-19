import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
from scipy.signal import periodogram, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig

class DynamicModeDecomposition:
    def __init__(self, data_matrix, dt):
        """
        Initialize the DMD class.
        
        :param data_matrix: A 2D numpy array where each row represents acceleration data from one accelerometer.
        :param dt: Time step between recordings.
        """
        self.data_matrix = data_matrix
        self.dt = dt
        self.eigenvalues = None
        self.modes = None
        self.frequencies = None

    def fit(self):
        """
        Perform Dynamic Mode Decomposition.
        """
        X = self.data_matrix[:, :-1]
        X_prime = self.data_matrix[:, 1:]

        # Step 1: SVD of X
        U, S, Vh = svd(X, full_matrices=False)

        # Step 2: Reduced representation
        r = len(S)  # Can truncate to fewer modes by selecting r < len(S)
        U_r = U[:, :r]
        S_r = np.diag(S[:r])
        V_r = Vh[:r, :]

        # Step 3: Build the reduced A_tilde matrix
        A_tilde = U_r.T @ X_prime @ V_r.T @ np.linalg.inv(S_r)

        # Step 4: Compute eigenvalues and eigenvectors of A_tilde
        eigvals, eigvecs = eig(A_tilde)

        # Step 5: Map back to the high-dimensional space
        self.eigenvalues = eigvals
        self.modes = U_r @ eigvecs

        # Step 6: Compute frequencies from eigenvalues
        self.frequencies = np.angle(self.eigenvalues) / (2 * np.pi * self.dt)

    def plot_modes(self):
        """
        Plot the real and imaginary parts of the DMD modes.
        """
        if self.modes is None:
            raise ValueError("DMD has not been fitted yet. Run `fit` first.")

        for i, mode in enumerate(self.modes.T):
            plt.figure()
            plt.title(f"DMD Mode {i+1}")
            plt.plot(np.real(mode), label='Real Part')
            plt.plot(np.imag(mode), label='Imaginary Part')
            plt.legend()
            plt.show()

    def plot_frequencies(self):
        """
        Plot the frequencies of the DMD modes.
        """
        if self.frequencies is None:
            raise ValueError("Frequencies have not been computed. Run `fit` first.")
        
        plt.figure()
        plt.title("DMD Frequencies")
        plt.stem(self.frequencies, use_line_collection=True)
        plt.xlabel("Mode index")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)
        plt.show()


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
                rms_data[device] = np.sqrt(np.mean(signal ** 2))

                # Plot PSD
                plt.plot(freqs, psd, linewidth=0.5, label=f'{device}')

            plt.legend()
            plt.savefig(os.path.join(save_directory, f'common_silent_psd_{axis}.png'))
            plt.show()

            # Display P2P and RMS values
            print(f"Peak-to-Peak and RMS for silent interval in {axis}:")
            for device in device_data.keys():
                print(f"Device: {device}")
                print(f"  Peak-to-Peak (P2P): {p2p_data[device] * 1000:.6f} mg")
                print(f"  RMS: {rms_data[device] * 1000:.6f} mg")

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

def plot_ssta_slta_ratio_all_devices(device_data, axis='acc_x', nSTA=125 * 3, nLTA=125 * 30, save_directory='plots'):
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
        signal_squared = df[axis] ** 2  # Squared acceleration signal

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
                    # df = pd.read_fwf(file_path, usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df = pd.read_csv(file_path, delimiter='\t',
                                     usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']  # Rename columns for ease of use

                    # Convert time column to datetime format for easier filtering
                    df['time'] = pd.to_datetime(df['time'])
                    # Convert acceleration from milli-g to g
                    df['acc_x'] = df['acc_x'] / 1000
                    df['acc_y'] = df['acc_y'] / 1000
                    df['acc_z'] = df['acc_z'] / 1000

                    df = baseline_correction(df, linear=True)

                    # Store the DataFrame in the dictionary with device name as key
                    device_df = pd.concat([device_df, df], axis=0, ignore_index=True)
                    device_data[device_folder] = device_df

    return device_data

def get_data_by_file(data_directory='Data/T7/sbp'):
    """
    Reads and organizes acceleration data from each file in the directory.

    Parameters:
    - data_directory: The root directory containing device folders with files.

    Returns:
    - file_data: A dictionary where each key is a filename, and the value is its corresponding DataFrame.
    """
    file_data = {}

    # Traverse through each device folder
    for device_folder in os.listdir(data_directory):
        device_folder_path = os.path.join(data_directory, device_folder)

        if os.path.isdir(device_folder_path):
            for file_name in os.listdir(device_folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(device_folder_path, file_name)
                    
                    # Read the file and parse columns
                    df = pd.read_csv(file_path, delimiter='\t',
                                     usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                    df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']  # Rename columns for consistency
                    
                    # Convert time column to datetime format
                    df['time'] = pd.to_datetime(df['time'])
                    
                    # Convert acceleration from milli-g to g
                    df['acc_x'] = df['acc_x'] / 1000
                    df['acc_y'] = df['acc_y'] / 1000
                    df['acc_z'] = df['acc_z'] / 1000

                    df = baseline_correction(df, linear=True)

                    # Store the DataFrame with the file name as key
                    file_data[file_name] = df

    return file_data

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

def baseline_correction(df, linear=False):
    """
    Applies zero baseline correction or linear baseline correction to a single DataFrame.

    Parameters:
    - df: A pandas DataFrame containing acceleration columns: 'acc_x', 'acc_y', 'acc_z', and a 'time' column.
    - linear: If True, applies linear baseline correction; otherwise, zero baseline correction.

    Returns:
    - df_corrected: A DataFrame with corrected acceleration data.
    """
    # Ensure the 'time' column exists
    if 'time' not in df.columns:
        raise KeyError("'time' column is missing in the DataFrame.")

    # Ensure 'time' is in datetime format
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception as e:
            raise ValueError(f"Failed to convert 'time' column to datetime. Error: {e}")

    df_corrected = df.copy()

    if linear:
        # Apply linear baseline correction
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis not in df_corrected.columns:
                raise KeyError(f"'{axis}' column is missing in the DataFrame.")

            time_numeric = (df_corrected['time'] - df_corrected['time'].iloc[0]).dt.total_seconds()
            trend = np.polyfit(time_numeric, df_corrected[axis], 1)  # Fit a linear trend
            linear_baseline = np.polyval(trend, time_numeric)        # Evaluate the trend
            df_corrected[axis] = df_corrected[axis] - linear_baseline  # Subtract the linear baseline
    else:
        # Apply zero baseline correction (subtract mean)
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis not in df_corrected.columns:
                raise KeyError(f"'{axis}' column is missing in the DataFrame.")
            df_corrected[axis] = df_corrected[axis] - df_corrected[axis].mean()

    return df_corrected


def get_data_with_gaps(threshold_minutes=1):
    device_data = {}
    gap_info = {}

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
                    # Convert acceleration from milli-g to g
                    df['acc_x'] = df['acc_x'] / 1000
                    df['acc_y'] = df['acc_y'] / 1000
                    df['acc_z'] = df['acc_z'] / 1000
                    # Apply baseline correction by subtracting the mean of each axis
                    df = baseline_correction(df, linear=True)

                    # Store the DataFrame in the dictionary with device name as key
                    device_df = pd.concat([device_df, df], axis=0, ignore_index=True)

            # Calculate time gaps
            if not device_df.empty:
                device_df['time_diff'] = device_df['time'].diff().dt.total_seconds() / 60  # Time difference in minutes
                gaps = device_df[device_df['time_diff'] > threshold_minutes]
                if not gaps.empty:
                    gap_info[device_folder] = gaps[['time', 'time_diff']]

            device_data[device_folder] = device_df

    return device_data, gap_info

def plot_recursive_sta_lta(device_data, nSTA=125 * 3, nLTA=125 * 30, save_directory='plots'):
    """
    Calculates and plots recursive STA/LTA ratio vs. time for each direction (x, y, z) separately,
    including all devices in each direction.

    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - nSTA: Number of samples for the short-term average (STA) window.
    - nLTA: Number of samples for the long-term average (LTA) window.
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(14, 8))
    plt.title(f'Recursive STA/LTA Ratio vs Time for All Devices')
    plt.xlabel('Time')
    plt.ylabel('rSTA/rLTA Ratio')

    for device, df in device_data.items():
        # Compute acceleration vector (magnitude)
        vec2 = df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
        # Initialize recursive STA (rSTA) and recursive LTA (rLTA)
        rSTA = np.zeros_like(vec2)
        rLTA = np.zeros_like(vec2)
        rSTA[0] = vec2[0]
        rLTA[0] = vec2[0]
        # Calculate recursive STA
        for i in range(1, len(vec2)):
            if i < nSTA:
                rSTA[i] = rSTA[i - 1] + (vec2[i] - rSTA[i - 1]) / (i + 1)
            else:
                rSTA[i] = rSTA[i - 1] + (vec2[i] - rSTA[i - 1]) / nSTA
        # Calculate recursive LTA
        for i in range(1, len(vec2)):
            if i < nLTA:
                rLTA[i] = rLTA[i - 1] + (vec2[i] - rLTA[i - 1]) / (i + 1)
            else:
                rLTA[i] = rLTA[i - 1] + (vec2[i] - rLTA[i - 1]) / nLTA
        # Avoid division by zero
        rLTA[rLTA == 0] = np.nan
        # Compute rSTA/rLTA ratio
        rsta_rlta_ratio = rSTA / rLTA
        # Plot rSTA/rLTA ratio for this device
        plt.plot(df['time'], rsta_rlta_ratio, label=device, linewidth=0.5)

    # Save and show the plot
    plt.legend()
    plt.ylim(0, 10)
    plt.grid(True)
    save_path = os.path.join(save_directory, f'recursive_sta_lta_ratio_all.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot for all saved to {save_path}")

def plot_sta_lta_together(device_data, nSTA=125 * 3, nLTA=125 * 30, save_directory='plots'):
    """
    Calculates and plots STA and LTA together for all devices in each direction (x, y, z).

    Parameters:
    - device_data: Dictionary of DataFrames for each device.
    - nSTA: Number of samples for the short-term average (STA) window.
    - nLTA: Number of samples for the long-term average (LTA) window.
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    for axis in ['acc_x', 'acc_y', 'acc_z']:
        plt.figure(figsize=(14, 8))
        plt.title(f'STA and LTA for {axis.upper()} (All Devices)')
        plt.xlabel('Time')
        plt.ylabel('Value')

        for device, df in device_data.items():
            # Compute acceleration vector (magnitude)
            vec2 = df[axis] ** 2

            # Initialize STA and LTA arrays
            sSTA = np.zeros_like(vec2)
            sLTA = np.zeros_like(vec2)

            # Calculate STA
            for i in range(1, len(vec2)):
                if i < nSTA:
                    sSTA[i] = sSTA[i - 1] + (vec2.iloc[i] - sSTA[i - 1]) / (i + 1)
                else:
                    sSTA[i] = sSTA[i - 1] + (vec2.iloc[i] - vec2.iloc[i - nSTA]) / nSTA

            # Calculate LTA
            for i in range(1, len(vec2)):
                if i < nLTA:
                    sLTA[i] = sLTA[i - 1] + (vec2.iloc[i] - sLTA[i - 1]) / (i + 1)
                else:
                    sLTA[i] = sLTA[i - 1] + (vec2.iloc[i] - vec2.iloc[i - nLTA]) / nLTA

            # Plot STA and LTA for this device
            plt.plot(df['time'], np.sqrt(sSTA), label=f'{device} STA', linewidth=0.5, linestyle='--')
            plt.plot(df['time'], np.sqrt(sLTA), label=f'{device} LTA', linewidth=0.5)

        # Add legend and save the plot
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_directory, f'sta_lta_together_{axis}.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Plot for {axis} saved to {save_path}")

def plot_recursive_sta_lta_by_file(file_data, nSTA=125 * 3, nLTA=125 * 30, save_directory='plots'):
    """
    Computes and plots rSTA/rLTA ratio for each file separately.

    Parameters:
    - file_data: Dictionary of DataFrames, where each key is a filename and the value is its DataFrame.
    - nSTA: Number of samples for the short-term average (STA) window.
    - nLTA: Number of samples for the long-term average (LTA) window.
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    for file_name, df in file_data.items():
        plt.figure(figsize=(14, 8))
        plt.title(f'Recursive STA/LTA Ratio for {file_name}')
        plt.xlabel('Time')
        plt.ylabel('rSTA/rLTA Ratio')

        # Compute acceleration vector for the current axis
        vec2 = df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
        # Initialize recursive STA (rSTA) and recursive LTA (rLTA)
        rSTA = np.zeros_like(vec2)
        rLTA = np.zeros_like(vec2)
        rSTA[0] = vec2[0]
        rLTA[0] = vec2[0]
        # Compute rSTA
        for i in range(1, len(vec2)):
            if i < nSTA:
                rSTA[i] = rSTA[i - 1] + (vec2[i] - rSTA[i - 1]) / (i + 1)
            else:
                rSTA[i] = rSTA[i - 1] + (vec2[i] - rSTA[i - 1]) / nSTA
        # Compute rLTA
        for i in range(1, len(vec2)):
            if i < nLTA:
                rLTA[i] = rLTA[i - 1] + (vec2[i] - rLTA[i - 1]) / (i + 1)
            else:
                rLTA[i] = rLTA[i - 1] + (vec2[i] - rLTA[i - 1]) / nLTA
        # Avoid division by zero
        rLTA[rLTA == 0] = np.nan
        # Compute rSTA/rLTA ratio
        #rsta_rlta_ratio = np.sqrt(rSTA) / np.sqrt(rLTA)
        # Plot rSTA/rLTA ratio for this axis
        plt.plot(df['time'], rSTA/rLTA, label='rSTA/rLTA', linewidth=0.5)

        # Save the plot
        plt.legend()
        #plt.ylim(0, 10)
        plt.grid(True)
        save_path = os.path.join(save_directory, f'{file_name}_rSTA_rLTA.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Plot for {file_name} saved to {save_path}")

def plot_standard_sta_lta_by_file(file_data, nSTA=125 * 3, nLTA=125 * 30, save_directory='plots'):
    """
    Computes and plots standard STA (sSTA) and LTA (sLTA) for all devices within each file.

    Parameters:
    - file_data: Dictionary where each key is a filename and the value is its DataFrame.
    - devices: List of device names (e.g., ['device1', 'device2', ...]) to extract from each file.
               Each device corresponds to a set of columns like 'device1_acc_x', 'device1_acc_y', etc.
    - nSTA: Number of samples for the short-term average (STA) window.
    - nLTA: Number of samples for the long-term average (LTA) window.
    - save_directory: Directory to save the plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    for file_name, df in file_data.items():
        plt.figure(figsize=(14, 8))
        plt.title(f'Recursive STA/LTA Ratio for {file_name}')
        plt.xlabel('Time')
        plt.ylabel('rSTA/rLTA Ratio')

        # Compute acceleration vector for the current axis
        vec2 = df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
        # Initialize recursive STA (rSTA) and recursive LTA (rLTA)
        if file_name == '2024_10_23_1346_RISE_02.txt':
            x=1
        sSTA = np.zeros_like(vec2)
        sLTA = np.zeros_like(vec2)
        sSTA[0] = vec2[0]
        sLTA[0] = vec2[0]
        # Compute rSTA
        for i in range(1, len(vec2)):
            if i < nSTA:
                sSTA[i] = sSTA[i - 1] + (vec2[i] - sSTA[i - 1]) / (i + 1)
            else:
                sSTA[i] = sSTA[i - 1] + (vec2[i] - vec2[i - nSTA]) / nSTA
                # Compute rLTA
        for i in range(1, len(vec2)):
            
            if i < nLTA:
                sLTA[i] = sLTA[i - 1] + (vec2[i] - sLTA[i - 1]) / (i + 1)
            else:
                sLTA[i] = sLTA[i - 1] + (vec2[i] - vec2[i - nLTA]) / nLTA
        # Compute rSTA/rLTA ratio
        #rsta_rlta_ratio = np.sqrt(rSTA) / np.sqrt(rLTA)
        # Plot rSTA/rLTA ratio for this axis
        plt.plot(df['time'], sSTA/sLTA, label='sSTA/sLTA', linewidth=0.5)

        # Save the plot
        plt.legend()
        plt.ylim(0, 10)
        plt.grid(True)
        save_path = os.path.join(save_directory, f'{file_name}_sSTA_sLTA.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Plot for {file_name} saved to {save_path}")

def extract_and_compare_frequencies(device_data, dt=1/125, save_directory='plots'):
    """
    Extracts frequencies using DMD from acceleration data and compares them with FEM model frequencies.

    Parameters:
    - device_data: Dictionary containing acceleration data for each device.
    - dt: Time step between recordings (assumed 1/125 for 125 Hz sampling rate).
    - save_directory: Directory to save plots.
    """
    os.makedirs(save_directory, exist_ok=True)

    for axis in ['acc_x', 'acc_y', 'acc_z']:
        # Initialize a list to store DMD frequencies for each device
        dmd_frequencies_per_device = {}

        for device, df in device_data.items():
            # Extract the acceleration data for the given axis and convert to a 2D NumPy array (each row represents data from one device)
            data_matrix = df[axis].values.reshape(1, -1)

            # Perform DMD on the acceleration data
            dmd = DynamicModeDecomposition(data_matrix, dt)
            dmd.fit()

            # Store the frequencies obtained from DMD
            dmd_frequencies_per_device[device] = dmd.frequencies

            # Plot DMD frequencies for each device
            plt.figure()
            plt.stem(dmd.frequencies)
            plt.title(f"DMD Frequencies for {axis} - Device {device}")
            plt.xlabel("Mode Index")
            plt.ylabel("Frequency (Hz)")
            plt.grid()
            plt.savefig(os.path.join(save_directory, f'dmd_frequencies_{axis}_{device}.png'))
            plt.show()

            # Print the DMD frequencies for comparison
            print(f"Device: {device}, Axis: {axis}, DMD Frequencies (Hz): {dmd.frequencies}")

        # Compare frequencies across all devices for the current axis
        # (You can extend this comparison to your FEM model frequencies)

    print("Frequency extraction and comparison complete.")



if __name__ == '__main__':
    #device_data, gap_info = get_data_with_gaps()

    # plot_full_acceleration(device_data)
    # find_common_silent_interval(device_data, axis='acc_x', peak_threshold=0.001, window_minutes=1)
    # find_common_silent_interval(device_data, axis='acc_y', peak_threshold=0.001, window_minutes=1)
    # find_common_silent_interval(device_data, axis='acc_z', peak_threshold=0.001, window_minutes=1)
    # analyze_silent_interval(device_data, axis='acc_x',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    # analyze_silent_interval(device_data, axis='acc_y',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    # analyze_silent_interval(device_data, axis='acc_z',peak_threshold=0.001, window_minutes=1, save_directory='plots')
    # plot_ssta_slta_ratio_all_devices(device_data, axis='acc_x', nSTA=125 * 3, nLTA=125 * 30, save_directory='plots')
    # plot_ssta_slta_ratio_all_devices(device_data, axis='acc_y', nSTA=125 * 3, nLTA=125 * 30, save_directory='plots')
    # plot_ssta_slta_ratio_all_devices(device_data, axis='acc_z', nSTA=125 * 3, nLTA=125 * 30, save_directory='plots')
    # Assuming device_data is already obtained from your `get_data` function or equivalent
    device_data = get_data_by_file()

    #plot_recursive_sta_lta(device_data)
    plot_recursive_sta_lta_by_file(device_data)
    plot_standard_sta_lta_by_file(device_data)
