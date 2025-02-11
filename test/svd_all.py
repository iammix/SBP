import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt
import sys

sys.path.append(r'A:\Projects\SBP')

from oma import baseline_correction_

def butterworth_bandpass_filter(data, lowcut=0.1, highcut=18.0, fs=100.0, order=4):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
        data: DataFrame, input signal data with multiple columns (e.g., acc_x, acc_y, acc_z).
        lowcut: float, low cut-off frequency in Hz (default: 0.1 Hz).
        highcut: float, high cut-off frequency in Hz (default: 20 Hz).
        fs: float, sampling frequency in Hz (default: 100 Hz).
        order: int, order of the filter (default: 4).

    Returns:
        filtered_data: DataFrame, filtered signal data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create Butterworth filter coefficients
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to each column of the data
    filtered_data = data.apply(lambda col: filtfilt(b, a, col), axis=0)
    
    return filtered_data

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['Date'] + " " + df['Time'])
    df = df.rename(columns={"Acc x": "acc_x", "Acc y": "acc_y", "Acc z": "acc_z"})
    return df[['time', 'acc_x', 'acc_y', 'acc_z']]


def compute_psd(data, fs):
    """
    Compute PSD for each channel in the signal data.
    Parameters:
        data: array-like (NumPy array or pandas DataFrame), the input signal data.
        fs: int, sampling frequency of the data.
    Returns:
        freqs: array, frequency bins.
        psd_matrix: array, PSD values for each channel.
    """
    psd_list = []

    # Check if the input is a DataFrame or NumPy array
    if isinstance(data, np.ndarray):
        for i in range(data.shape[1]):  # Loop through each channel
            freqs, psd = welch(data[:, i], fs, nperseg=256)
            psd_list.append(psd)
    elif hasattr(data, "columns"):  # Check for DataFrame-like structure
        for column in data.columns:
            freqs, psd = welch(data[column], fs, nperseg=256)
            psd_list.append(psd)
    else:
        raise ValueError("Input data must be a NumPy array or a pandas DataFrame.")

    psd_matrix = np.array(psd_list).T  # Transpose to align dimensions
    return freqs, psd_matrix


device1 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_01.txt')
device2 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_02.txt')
device3 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_03.txt')
device4 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_04.txt')
device6 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_06.txt')
device7 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_07.txt')
device8 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_08.txt')
device9 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_09.txt')
device10 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_10.txt')
device11 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_11.txt')
device12 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_12.txt')
device14 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_14.txt')
device15 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_15.txt')
device16 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_16.txt')

device1 = baseline_correction_(device1)
device2 = baseline_correction_(device2)
device3 = baseline_correction_(device3)
device4 = baseline_correction_(device4)
device6 = baseline_correction_(device6)
device7 = baseline_correction_(device7)
device8 = baseline_correction_(device8)
device9 = baseline_correction_(device9)
device10 = baseline_correction_(device10)
device11 = baseline_correction_(device11)
device12 = baseline_correction_(device12)
device14 = baseline_correction_(device14)
device15 = baseline_correction_(device15)
device16 = baseline_correction_(device16)

# Generate synthetic data for 4 devices
device_data = [device1, device2, device3, device4, device6, device7, device8, device9, device10, device11, device12, device14, device15, device16]
sampling_frequencies = []
filtered_device_data = []

for device in device_data:
    t = pd.to_datetime(device['time'])
    time_diffs = t.diff().dt.total_seconds()
    avg_sampling_interval = time_diffs.mean()
    fs = 1/ avg_sampling_interval
    sampling_frequencies.append(fs)

    # Filter the data
    numeric_columns = device[['acc_x', 'acc_y', 'acc_z']]
    filtered_data = butterworth_bandpass_filter(numeric_columns, fs=fs)
    filtered_device_data.append(filtered_data)

# Replace device_data with the filtered data for further processing
device_data = filtered_device_data


def svd_reconstruct():
    # Compute PSD for a single device
    psd_results = []
    for i, (df, fs) in enumerate(zip(device_data, sampling_frequencies)):
        freqs, psd_matrix = compute_psd(df, fs)
        psd_results.append((freqs, psd_matrix))

    # Normalize to a common frequency grid
    common_freqs = np.linspace(0, min(sampling_frequencies) / 2,
                               256)  # Common frequency range (Nyquist frequency of smallest fs)
    normalized_psd_matrices = []

    for freqs, psd_matrix in psd_results:
        interp_func = interp1d(freqs, psd_matrix, axis=0, bounds_error=False, fill_value=0)  # Interpolate PSD
        normalized_psd = interp_func(common_freqs)  # Interpolate to common frequency grid
        normalized_psd_matrices.append(normalized_psd)

    # Combine normalized PSDs into a single matrix
    combined_psd_matrix = np.hstack(normalized_psd_matrices)  # Combine: [common_freq_bins, 4 devices * 3 channels]

    # Perform SVD on the combined PSD matrix
    U, Sigma, VT = np.linalg.svd(combined_psd_matrix, full_matrices=False)

    # Plot singular values
    plt.figure(figsize=(8, 5))
    plt.plot(Sigma, marker='o', label='Singular Values')
    plt.title('Singular Values of Combined PSD Matrix')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    # Reconstruct the PSD matrix using top k singular values
    k = 12  # Retain top 2 singular values
    Sigma_reduced = np.zeros_like(Sigma)
    Sigma_reduced[:k] = Sigma[:k]
    psd_matrix_reconstructed = (U[:, :k] * Sigma_reduced[:k]) @ VT[:k, :]

    # Plot original vs reconstructed PSD for the first device/channel
    plt.figure(figsize=(10, 6))
    plt.plot(common_freqs, combined_psd_matrix[:, 0], label="Original PSD (Device 1, x-axis)")
    plt.plot(common_freqs, psd_matrix_reconstructed[:, 0], '--', label="Reconstructed PSD (Device 1, x-axis)")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency')
    plt.legend()
    plt.title(f'Original vs Reconstructed PSD (Top {k} Singular Values)')
    plt.show()


def svd_feature_extract():
    # Compute PSD for each device
    psd_results = []
    for i, (df, fs_i) in enumerate(zip(device_data, sampling_frequencies)):
        freqs, psd_matrix = compute_psd(df, fs_i)
        psd_results.append((freqs, psd_matrix))

    # Normalize PSDs to a common frequency grid
    common_freqs = np.linspace(0, min(sampling_frequencies) / 2, 256)  # Common frequency range
    normalized_psd_matrices = []
    for freqs, psd_matrix in psd_results:
        interp_func = interp1d(freqs, psd_matrix, axis=0, bounds_error=False, fill_value=0)  # Interpolate PSD
        normalized_psd = interp_func(common_freqs)  # Interpolate to common frequency grid
        normalized_psd_matrices.append(normalized_psd)

    # Combine PSDs into a single matrix for feature extraction
    combined_psd_matrix = np.hstack(normalized_psd_matrices)  # Combine: [freq_bins, 4 devices * 3 channels]

    # Perform SVD
    U, Sigma, VT = np.linalg.svd(combined_psd_matrix, full_matrices=False)
    # Plot singular values
    plt.figure(figsize=(8, 5))
    plt.plot(Sigma, marker='o', label='Singular Values')
    plt.title('Singular Values of Combined PSD Matrix')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    k = 3  # Number of features
    # Extract Features
    features = U[:, :k]  # Top 3 spectral features (based on singular values)
    feature_importance = Sigma[:k]  # Importance of each feature

    # Visualize the feature importance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(feature_importance) + 1), feature_importance,
            tick_label=[f"Feature {i + 1}" for i in range(len(feature_importance))])
    plt.title('Feature Importance (Singular Values)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (Magnitude)')
    plt.show()

    # Visualize the first feature (example)
    plt.figure(figsize=(10, 6))
    for i in range(0, k):
        plt.plot(common_freqs, features[:, i], label=f"Feature {i + 1}")
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Feature Magnitude')
    plt.title('Top Spectral Feature (Feature 1)')
    plt.legend()
    plt.show()

    # Compare final PSD vs original PSD for each device
    for i, (freqs, psd_matrix) in enumerate(psd_results):
        # Interpolate original PSD to common frequency grid
        interp_func_original = interp1d(freqs, psd_matrix, axis=0, bounds_error=False, fill_value=0)
        original_psd_on_common_grid = interp_func_original(common_freqs)

        # Get the normalized PSD for the current device
        normalized_psd = normalized_psd_matrices[i]

        # Plot comparison
        plt.figure(figsize=(12, 6))
        for channel in range(psd_matrix.shape[1]):
            plt.plot(common_freqs, original_psd_on_common_grid[:, channel],
                     label=f"Original PSD (Channel {channel + 1})", linestyle='--')
            plt.plot(common_freqs, normalized_psd[:, channel], label=f"Normalized PSD (Channel {channel + 1})",
                     alpha=0.7)

        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Device {i + 1}: Original vs Normalized PSD')
        plt.legend()
        plt.show()


def lanczos_kernel(x, a=3):
    """
    Lanczos kernel function.
    Parameters:
        x: array-like, input distance values.
        a: int, Lanczos window size.
    Returns:
        Lanczos kernel values.
    """
    x = np.array(x)
    kernel = np.sinc(x) * np.sinc(x / a)
    kernel[np.abs(x) >= a] = 0  # Zero out values beyond the window size
    return kernel


def lanczos_resample(signal, src_rate, target_rate, a=3):
    """
    Resample a signal using Lanczos interpolation.
    Parameters:
        signal: array-like, the input signal to resample.
        src_rate: float, the original sampling rate of the signal.
        target_rate: float, the desired sampling rate.
        a: int, the Lanczos window size (default 3).
    Returns:
        Resampled signal.
    """
    # Compute the resampling ratio
    ratio = target_rate / src_rate

    # Generate new time points for the resampled signal
    n_samples = int(len(signal) * ratio)
    new_time = np.linspace(0, len(signal) - 1, n_samples)

    # Compute the resampled signal using the Lanczos kernel
    resampled_signal = np.zeros_like(new_time)
    for i, t in enumerate(new_time):
        # Compute distances from the current point
        distances = np.arange(len(signal)) - t

        # Compute the Lanczos weights
        weights = lanczos_kernel(distances, a=a)

        # Apply the weights to the original signal
        resampled_signal[i] = np.sum(weights * signal)

    return resampled_signal


def svd_feature_extract_resample():
    # Resample signals and compute PSD for each device
    psd_results = []
    target_rate = 100
    for i, (df, fs_i) in enumerate(zip(device_data, sampling_frequencies)):
        # Ensure only numeric data is resampled
        numeric_columns = df.select_dtypes(include=[np.number])

        resampled_data = np.zeros((int(len(numeric_columns) * target_rate / fs_i), numeric_columns.shape[1]))
        for j in range(numeric_columns.shape[1]):
            resampled_data[:, j] = lanczos_resample(numeric_columns.iloc[:, j].values, fs_i, target_rate)

        # Compute PSD for the resampled data
        freqs, psd_matrix = compute_psd(resampled_data, target_rate)
        psd_results.append((freqs, psd_matrix))

    # Combine PSDs into a single matrix for feature extraction
    combined_psd_matrix = np.hstack([psd_matrix for _, psd_matrix in psd_results])

    # Perform SVD
    U, Sigma, VT = np.linalg.svd(combined_psd_matrix, full_matrices=False)
    # Plot singular values
    plt.figure(figsize=(8, 5))
    plt.plot(Sigma, marker='o', label='Singular Values')
    plt.title('Singular Values of Combined PSD Matrix')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    k = 12 # Number of features
    # Extract Features
    features = U[:, :k]
    feature_importance = Sigma[:k]

    # Visualize the feature importance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(feature_importance) + 1), feature_importance,
            tick_label=[f"Feature {i + 1}" for i in range(len(feature_importance))])
    plt.title('Feature Importance (Singular Values)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (Magnitude)')
    plt.show()

    # Visualize the first feature (example)
    plt.figure(figsize=(10, 6))
    for i in range(0, k):
        plt.plot(freqs, features[:, i], label=f"Feature {i + 1}")
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Feature Magnitude')
    plt.title('Top Spectral Feature (Feature 1)')
    plt.legend()
    plt.show()

    # # Compare final PSD vs original PSD for each device
    # for i, (freqs, psd_matrix) in enumerate(psd_results):
    #     # Get the PSD for the current device
    #     plt.figure(figsize=(12, 6))
    #     for channel in range(psd_matrix.shape[1]):
    #         plt.plot(freqs, psd_matrix[:, channel], label=f"PSD (Channel {channel + 1})", alpha=0.7)

    #     plt.xscale('log')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power Spectral Density')
    #     plt.title(f'Device {i + 1}: PSD')
    #     plt.legend()
    #     plt.show()
    # ----- New plotting code: 3 subplots for the 3 channels -----

    # Reconstruct the PSD matrix using top k singular values
    k = 6  # Retain top 2 singular values
    Sigma_reduced = np.zeros_like(Sigma)
    Sigma_reduced[:k] = Sigma[:k]
    psd_matrix_reconstructed = (U[:, :k] * Sigma_reduced[:k]) @ VT[:k, :]

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Loop over channels (assuming order: 0=acc_x, 1=acc_y, 2=acc_z)
    for ch in range(3):
        for i, (freqs, psd_matrix) in enumerate(psd_results):
            axs[ch].plot(freqs, psd_matrix_reconstructed[:, ch], label=f"Device {i + 1}", alpha=0.7)
        axs[ch].set_xscale('log')
        axs[ch].set_xlabel('Frequency (Hz)')
        axs[ch].set_ylabel('Power Spectral Density')
        axs[ch].set_title(f'Channel {ch + 1} PSD across Devices')
        axs[ch].legend()

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #svd_reconstruct()
    svd_feature_extract_resample()
