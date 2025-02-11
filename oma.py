import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.signal import welch, butter, filtfilt, find_peaks


def sbp_parser(data_directory, file_extension=".txt"):
    """
    Parses SBP acceleration data from the given directory.

    Parameters:
    - data_directory: Path to the root directory containing device folders with SBP data files.
    - file_extension: Extension of the data files to parse (default: ".txt").

    Returns:
    - device_data: Dictionary with device folder names as keys and parsed DataFrames as values.
    """
    device_data = {}

    # Iterate over each device folder
    for device_folder in os.listdir(data_directory):
        device_folder_path = os.path.join(data_directory, device_folder)

        if os.path.isdir(device_folder_path):
            combined_df = pd.DataFrame()  # DataFrame to store all data for this device

            for file_name in os.listdir(device_folder_path):
                if file_name.endswith(file_extension):
                    file_path = os.path.join(device_folder_path, file_name)

                    # Parse the file
                    try:
                        df = pd.read_csv(file_path, delimiter='\t', usecols=['Time', 'Acc x', 'Acc y', 'Acc z'])
                        df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']  # Rename columns for consistency

                        # Convert time column to datetime
                        df['time'] = pd.to_datetime(df['time'])

                        # Convert acceleration from milli-g to g
                        df['acc_x'] = df['acc_x'] / 1000
                        df['acc_y'] = df['acc_y'] / 1000
                        df['acc_z'] = df['acc_z'] / 1000

                        # Baseline correction (subtract mean from each axis)
                        for axis in ['acc_x', 'acc_y', 'acc_z']:
                            # linear
                            df[axis] = df[axis] - df[axis].mean()

                        # Append to the combined DataFrame for the device
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

            if not combined_df.empty:
                # Store the combined DataFrame in the dictionary
                device_data[device_folder] = combined_df

    return device_data


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Designs a Butterworth bandpass filter.

    Parameters:
    - lowcut: Low cutoff frequency (Hz).
    - highcut: High cutoff frequency (Hz).
    - fs: Sampling frequency (Hz).
    - order: Order of the filter.

    Returns:
    - b, a: Filter coefficients.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass filter to the acceleration data.

    Parameters:
    - data: DataFrame containing acceleration data (columns are channels).
    - lowcut: Low cutoff frequency (Hz).
    - highcut: High cutoff frequency (Hz).
    - fs: Sampling frequency (Hz).
    - order: Order of the filter.

    Returns:
    - filtered_data: DataFrame with filtered acceleration data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_data = data.copy()
    for column in filtered_data.columns:
        filtered_data[column] = filtfilt(b, a, filtered_data[column])
    return filtered_data


def compute_psd_matrix(data, fs=125, nperseg=1024):
    """
    Computes the Power Spectral Density (PSD) matrix using Welch's method.

    Parameters:
    - data: DataFrame with acceleration data (columns are channels).
    - fs: Sampling frequency in Hz.
    - nperseg: Number of samples per segment for Welch's method.

    Returns:
    - freqs: Array of frequency bins.
    - psd_matrix: PSD matrix of shape (num_channels, num_channels, len(freqs)).
    """
    num_channels = data.shape[1]
    psd_matrix = []

    for i in range(num_channels):
        psd_row = []
        for j in range(num_channels):
            freqs, psd = welch(data.iloc[:, i] * data.iloc[:, j], fs=fs, nperseg=nperseg)
            psd_row.append(psd)
        psd_matrix.append(psd_row)

    psd_matrix = np.array(psd_matrix)
    return freqs, psd_matrix


def perform_fdd(psd_matrix):
    """
    Performs Frequency Domain Decomposition (FDD) on the PSD matrix.

    Parameters:
    - psd_matrix: PSD matrix of shape (num_channels, num_channels, len(freqs)).

    Returns:
    - singular_values: Array of singular values for each frequency.
    """
    num_freqs = psd_matrix.shape[-1]
    singular_values = []

    for i in range(num_freqs):
        psd_at_freq = psd_matrix[:, :, i]
        _, S, _ = svd(psd_at_freq)
        singular_values.append(S)

    singular_values = np.array(singular_values)
    return singular_values


def plot_singular_values_with_peaks(freqs, singular_values):
    """
    Plots singular values against frequency on a logarithmic x-axis, identifies peaks, 
    and annotates their values.

    Parameters:
    - freqs: Array of frequency bins.
    - singular_values: Array of singular values for each frequency.
    """
    plt.figure(figsize=(10, 6))
    peak_freqs = []  # To store frequencies of the peaks
    peak_values = []  # To store peak values

    for i in range(singular_values.shape[1]):
        plt.plot(freqs, singular_values[:, i], label=f"SV {i + 1}")

        # Identify peaks for the current singular value series
        peaks, _ = find_peaks(singular_values[:, i])
        peak_freqs.extend(freqs[peaks])
        peak_values.extend(singular_values[peaks, i])

        # Annotate peaks on the plot
        for peak, freq in zip(singular_values[peaks, i], freqs[peaks]):
            plt.scatter(freq, peak, color='red')  # Mark the peak
            plt.annotate(f"{freq:.2f}", xy=(freq, peak), xytext=(freq, peak * 1.1),
                         arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

    plt.xscale('log')  # Use a logarithmic scale for the x-axis
    plt.xlabel("Frequency (Hz, Log Scale)")
    plt.ylabel("Singular Values")
    plt.title("Singular Values with Peaks (FDD)")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

    # Print the peaks
    peak_data = pd.DataFrame({'Frequency (Hz)': peak_freqs, 'Singular Value': peak_values})
    print("\nPeaks Detected:")
    print(peak_data.sort_values(by='Frequency (Hz)'))

    return peak_data


def compute_average_normalized_singular_values(device_data, sampling_frequency, lowcut, highcut, order):
    """
    Computes the average normalized singular values of spectral density matrices across all recordings.

    Parameters:
    - device_data: Dictionary of DataFrames for each device's data.
    - sampling_frequency: Sampling frequency in Hz.
    - lowcut: Low cutoff frequency (Hz) for bandpass filter.
    - highcut: High cutoff frequency (Hz) for bandpass filter.
    - order: Order of the Butterworth filter.

    Returns:
    - freqs: Array of frequency bins.
    - avg_normalized_singular_values: Average normalized singular values for each frequency.
    """
    singular_values_list = []
    freqs = None

    for device, df in device_data.items():
        print(f"Processing device: {device}")

        # Select acceleration columns (e.g., acc_x, acc_y, acc_z)
        acceleration_data = df[['acc_x', 'acc_y', 'acc_z']]

        # Apply Butterworth bandpass filter
        filtered_data = apply_bandpass_filter(acceleration_data, lowcut, highcut, sampling_frequency, order)

        # Compute PSD matrix
        freqs, psd_matrix = compute_psd_matrix(filtered_data, fs=sampling_frequency)

        # Perform FDD to get singular values
        singular_values = perform_fdd(psd_matrix)

        # Normalize singular values for each frequency
        normalized_singular_values = singular_values / np.max(singular_values, axis=1, keepdims=True)

        # Append to list
        singular_values_list.append(normalized_singular_values)

    # Average normalized singular values across all recordings
    avg_normalized_singular_values = np.mean(singular_values_list, axis=0)

    return freqs, avg_normalized_singular_values


def plot_average_normalized_singular_values(freqs, avg_normalized_singular_values):
    """
    Plots the average normalized singular values against frequency.

    Parameters:
    - freqs: Array of frequency bins.
    - avg_normalized_singular_values: Array of average normalized singular values for each frequency.
    """
    plt.figure(figsize=(10, 6))
    for i in range(avg_normalized_singular_values.shape[1]):
        plt.plot(freqs, avg_normalized_singular_values[:, i], label=f"Avg Normalized SV {i + 1}")

    plt.xscale('log')  # Logarithmic x-axis
    plt.xlabel("Frequency (Hz, Log Scale)")
    plt.ylabel("Average Normalized Singular Values")
    plt.title("Average Normalized Singular Values of Spectral Density Matrices")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()


def compute_normalized_singular_values_for_each_device(device_data, sampling_frequency, lowcut, highcut, order):
    """
    Computes normalized singular values for spectral density matrices separately for each device.

    Parameters:
    - device_data: Dictionary of DataFrames for each device's data.
    - sampling_frequency: Sampling frequency in Hz.
    - lowcut: Low cutoff frequency (Hz) for bandpass filter.
    - highcut: High cutoff frequency (Hz) for bandpass filter.
    - order: Order of the Butterworth filter.

    Returns:
    - results: Dictionary where keys are device names, and values are tuples of (freqs, normalized_singular_values).
    """
    results = {}

    for device, df in device_data.items():
        print(f"Processing device: {device}")

        # Select acceleration columns (e.g., acc_x, acc_y, acc_z)
        acceleration_data = df[['acc_x', 'acc_y', 'acc_z']]

        # Apply Butterworth bandpass filter
        filtered_data = apply_bandpass_filter(acceleration_data, lowcut, highcut, sampling_frequency, order)

        # Compute PSD matrix
        freqs, psd_matrix = compute_psd_matrix(filtered_data, fs=sampling_frequency)

        # Perform FDD to get singular values
        singular_values = perform_fdd(psd_matrix)

        # Normalize singular values for each frequency
        normalized_singular_values = singular_values / np.max(singular_values, axis=1, keepdims=True)

        # Store results for the device
        results[device] = (freqs, normalized_singular_values)

    return results


def plot_normalized_singular_values_per_device(results):
    """
    Plots normalized singular values separately for each device.

    Parameters:
    - results: Dictionary where keys are device names, and values are tuples of (freqs, normalized_singular_values).
    """
    for device, (freqs, normalized_singular_values) in results.items():
        plt.figure(figsize=(10, 6))
        for i in range(normalized_singular_values.shape[1]):
            plt.plot(freqs, normalized_singular_values[:, i], label=f"Normalized SV {i + 1}")

        plt.xscale('log')  # Logarithmic x-axis
        plt.xlabel("Frequency (Hz, Log Scale)")
        plt.ylabel("Normalized Singular Values")
        plt.title(f"Normalized Singular Values for Device: {device}")
        plt.legend()
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.show()


from scipy.linalg import svd, eig


def construct_hankel_matrix(data, l):
    """
    Constructs a Hankel matrix from the given time-series data.

    Parameters:
    - data: 2D numpy array (rows = channels, columns = time samples).
    - l: Number of block rows (corresponds to model order).

    Returns:
    - H: Hankel matrix.
    """
    num_channels, num_samples = data.shape
    n_columns = num_samples - 2 * l + 1
    if n_columns < 1:
        raise ValueError("Not enough data samples for the specified block size.")

    # Construct the Hankel matrix
    H = np.zeros((l * num_channels, n_columns))
    for i in range(l):
        H[i * num_channels:(i + 1) * num_channels, :] = data[:, i:i + n_columns]

    return H


def stochastic_subspace_identification(data, l, model_orders):
    """
    Applies Stochastic Subspace Identification (SSI) to estimate modal parameters.

    Parameters:
    - data: 2D numpy array (rows = channels, columns = time samples).
    - l: Number of block rows (corresponds to model order).
    - model_orders: List of model orders to analyze.

    Returns:
    - results: Dictionary containing natural frequencies, damping ratios, and mode shapes for each model order.
    """

    H = construct_hankel_matrix(data, l)
    U, S, Vh = svd(H)  # Singular Value Decomposition

    results = {}
    for model_order in model_orders:
        # Reduce to the desired model order
        U_r = U[:, :model_order]
        S_r = np.diag(S[:model_order])
        V_r = Vh[:model_order, :]

        # Compute the system matrix
        O = U_r @ np.sqrt(S_r)  # Observability matrix
        C = O[:data.shape[0], :]  # Output matrix
        A_hat = np.linalg.pinv(O[:-data.shape[0], :]) @ O[data.shape[0]:, :]  # System matrix

        # Eigen decomposition to estimate natural frequencies and damping ratios
        eigvals, eigvecs = eig(A_hat)
        frequencies = np.abs(np.angle(eigvals)) / (2 * np.pi)
        damping_ratios = -np.real(eigvals) / np.abs(eigvals)

        # Normalize mode shapes
        mode_shapes = C @ eigvecs
        mode_shapes = mode_shapes / np.abs(mode_shapes).max(axis=0)

        # Store results
        results[model_order] = {
            "frequencies": frequencies,
            "damping_ratios": damping_ratios,
            "mode_shapes": mode_shapes,
        }

    return results


def plot_ssi_results(results, sampling_frequency):
    """
    Plots the natural frequencies and damping ratios from SSI results.

    Parameters:
    - results: Dictionary containing natural frequencies and damping ratios for each model order.
    - sampling_frequency: Sampling frequency in Hz.
    """
    for model_order, result in results.items():
        frequencies = result["frequencies"] * sampling_frequency
        damping_ratios = result["damping_ratios"]

        plt.figure(figsize=(10, 6))
        plt.scatter(frequencies, damping_ratios, label=f"Model Order {model_order}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Damping Ratio")
        plt.title(f"Modal Parameters for Model Order {model_order}")
        plt.grid(True)
        plt.legend()
        plt.show()


def ssi():
    # Path to the root directory with SBP data
    data_directory = 'Data/T7/sbp'
    parsed_data = sbp_parser(data_directory)

    # Bandpass filter parameters
    lowcut = 0.1
    highcut = 20.0
    sampling_frequency = 125
    order = 4

    # Block size and model orders for SSI
    block_size = 50  # Adjust based on system complexity
    model_orders = [10, 20, 30, 40]  # Example model orders to analyze

    # Example: Process a specific device's data
    for device, df in parsed_data.items():
        print(f"Processing device: {device}")

        # Select acceleration columns (e.g., acc_x, acc_y, acc_z)
        acceleration_data = df[['acc_x', 'acc_y', 'acc_z']].to_numpy().T

        # Apply Butterworth bandpass filter
        filtered_data = apply_bandpass_filter(pd.DataFrame(acceleration_data.T), lowcut, highcut, sampling_frequency,
                                              order).to_numpy().T

        # Perform SSI
        ssi_results = stochastic_subspace_identification(filtered_data, block_size, model_orders)

        # Plot SSI results
        plot_ssi_results(ssi_results, sampling_frequency)


def fdd():
    # Path to the root directory with SBP data
    data_directory = 'Data/T7/sbp/'
    parsed_data = sbp_parser(data_directory)

    # Bandpass filter parameters
    lowcut = 0.1
    highcut = 20.0
    sampling_frequency = 125  # Adjust if different
    order = 4

    # Compute normalized singular values for each device
    results = compute_normalized_singular_values_for_each_device(parsed_data, sampling_frequency, lowcut, highcut,
                                                                 order)

    # Plot normalized singular values for each device
    plot_normalized_singular_values_per_device(results)


def baseline_correction_hnd(rec_a, rec_t):
    # linear baseline correction
    rec_t_sec = np.array([(t - rec_t[0]).total_seconds() for t in rec_t])

    sps = len(rec_t_sec) / (rec_t_sec[-1] - rec_t_sec[0])

    rec_a_cor = np.empty_like(rec_a)
    for i in range(3):
        signal = rec_a[:, i]
        mean_signal = np.mean(signal)
        mean_time = np.mean(rec_t_sec)
        numerator = np.sum((rec_t_sec - mean_time) * (signal - mean_signal))
        denominator = np.sum((rec_t_sec - mean_time) ** 2)
        slope = numerator / denominator
        intercept = mean_signal - slope * mean_time
        baseline = slope * rec_t_sec + intercept
        rec_a_cor[:, i] = signal - baseline

    return rec_a_cor

def baseline_correction_(df, linear=True):
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


if __name__ == '__main__':
    fdd()
