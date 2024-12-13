import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt
from oma import baseline_correction_
from scipy.interpolate import interp1d


def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['Date'] + " " + df['Time'])
    df = df.rename(columns={"Acc x": "acc_x", "Acc y": "acc_y", "Acc z": "acc_z"})
    return df[['time', 'acc_x', 'acc_y', 'acc_z']]

def compute_psd(df, fs):
    freqs, psd_x = welch(df['acc_x'], fs, nperseg=256)
    _, psd_y = welch(df['acc_y'], fs, nperseg=256)
    _, psd_z = welch(df['acc_z'], fs, nperseg=256)
    return freqs, np.vstack([psd_x, psd_y, psd_z]).T  # PSD matrix: [freq_bins, 3 channels]



device1 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_01.txt')
device1 = baseline_correction_(device1)
device2 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_02.txt')
device2 = baseline_correction_(device2)
device3 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_03.txt')
device3 = baseline_correction_(device3)
device4 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_04.txt')
device4 = baseline_correction_(device4)

# Generate synthetic data for 4 devices
device_data = [device1, device2, device3, device4]
sampling_frequencies = []
for device in device_data:
    t = pd.to_datetime(device['time'])
    time_diffs = t.diff().dt.total_seconds()
    avg_sampling_interval = time_diffs.mean()
    sampling_frequencies.append(1 / avg_sampling_interval)


def svd_reconstruct():
    # Compute PSD for a single device
    psd_results = []
    for i, (df, fs) in enumerate(zip(device_data, sampling_frequencies)):
        freqs, psd_matrix = compute_psd(df, fs)
        psd_results.append((freqs, psd_matrix))

    # Normalize to a common frequency grid
    common_freqs = np.linspace(0, min(sampling_frequencies) / 2, 256)  # Common frequency range (Nyquist frequency of smallest fs)
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

    k = 3 # Number of features
    # Extract Features
    features = U[:, :k]  # Top 3 spectral features (based on singular values)
    feature_importance = Sigma[:k]  # Importance of each feature

    # Visualize the feature importance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(feature_importance) + 1), feature_importance, tick_label=[f"Feature {i+1}" for i in range(len(feature_importance))])
    plt.title('Feature Importance (Singular Values)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (Magnitude)')
    plt.show()

    # Visualize the first feature (example)
    plt.figure(figsize=(10, 6))
    for i in range(0, k):
        plt.plot(common_freqs, features[:, i], label=f"Feature {i+1}")
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
            plt.plot(common_freqs, original_psd_on_common_grid[:, channel], label=f"Original PSD (Channel {channel+1})", linestyle='--')
            plt.plot(common_freqs, normalized_psd[:, channel], label=f"Normalized PSD (Channel {channel+1})", alpha=0.7)

        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Device {i+1}: Original vs Normalized PSD')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    svd_feature_extract()