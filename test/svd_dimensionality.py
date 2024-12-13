import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import pandas as pd
from oma import baseline_correction_

def svd_dimensionality_reduction():
    device1 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_01.txt')
    device1 = baseline_correction_(device1)
    device2 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_02.txt')
    device3 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_03.txt')
    device4 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_04.txt')

    # Generate synthetic multi-channel signal (e.g., x, y, z axes)
    np.random.seed(42)
    #fs = 100  # Sampling frequency
    #t = np.linspace(0, 10, fs * 10)  # 10 seconds
    #x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.normal(size=len(t))
    #y = np.sin(2 * np.pi * 20 * t) + 0.5 * np.random.normal(size=len(t))
    #z = np.sin(2 * np.pi * 30 * t) + 0.5 * np.random.normal(size=len(t))


    t = device1['time']
    x = device1['acc_x']
    y = device1['acc_y']
    z = device1['acc_z']

    t = pd.to_datetime(t)
    time_diffs = t.diff().dt.total_seconds()
    avg_sampling_interval = time_diffs.mean()
    fs = 1 / avg_sampling_interval

    # Generate synthetic multi-dimensional acceleration data
    np.random.seed(42)
    #time = np.linspace(0, 10, 500)  # 10 seconds, 500 samples
    #x = np.sin(2 * np.pi * 1 * time) + 0.1 * np.random.normal(size=len(time))
    #y = np.cos(2 * np.pi * 1 * time) + 0.1 * np.random.normal(size=len(time))
    #z = 0.5 * np.sin(2 * np.pi * 2 * time) + 0.1 * np.random.normal(size=len(time))
    #z = 0.5 * np.sin(2 * np.pi * 2 * time) + 0.1 * np.random.normal(size=len(time))



    # Stack into a matrix (rows: time samples, columns: x, y, z axes)
    acceleration_matrix = np.vstack([x, y, z]).T

    # Perform SVD
    U, Sigma, VT = np.linalg.svd(acceleration_matrix, full_matrices=False)

    # Reduce dimensions to k=1 (retain only the most significant singular value)
    k = 2
    Sigma_reduced = np.diag(Sigma[:k])  # Keep only the top k singular values
    U_reduced = U[:, :k]
    VT_reduced = VT[:k, :]

    # Reconstruct data in reduced-dimensional space
    reduced_data = U_reduced @ Sigma_reduced  # This is the reduced representation
    reconstructed_data = reduced_data @ VT_reduced  # Approximation in original space

    # Plot the original and reconstructed data (x, y, z axes)
    plt.figure(figsize=(12, 8))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(t, acceleration_matrix[:, 0], label='Original x')
    plt.plot(t, reconstructed_data[:, 0], '--', label='Reconstructed x')
    plt.title('Dimensionality Reduction: X Axis')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, acceleration_matrix[:, 1], label='Original y')
    plt.plot(t, reconstructed_data[:, 1], '--', label='Reconstructed y')
    plt.title('Dimensionality Reduction: Y Axis')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, acceleration_matrix[:, 2], label='Original z')
    plt.plot(t, reconstructed_data[:, 2], '--', label='Reconstructed z')
    plt.title('Dimensionality Reduction: Z Axis')
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['Date'] + " " + df['Time'])
    df = df.rename(columns={"Acc x": "acc_x", "Acc y": "acc_y", "Acc z": "acc_z"})
    return df[['time', 'acc_x', 'acc_y', 'acc_z']]


def svd_on_psd():
    device1 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_01.txt')
    device1 = baseline_correction_(device1)
    device2 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_02.txt')
    device3 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_03.txt')
    device4 = load_data(r'A:\Projects\SBP\test\2024_10_23_1305_RISE_04.txt')

    # Generate synthetic multi-channel signal (e.g., x, y, z axes)
    np.random.seed(42)
    #fs = 100  # Sampling frequency
    #t = np.linspace(0, 10, fs * 10)  # 10 seconds
    #x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.normal(size=len(t))
    #y = np.sin(2 * np.pi * 20 * t) + 0.5 * np.random.normal(size=len(t))
    #z = np.sin(2 * np.pi * 30 * t) + 0.5 * np.random.normal(size=len(t))


    t = device1['time']
    x = device1['acc_x']
    y = device1['acc_y']
    z = device1['acc_z']

    t = pd.to_datetime(t)
    time_diffs = t.diff().dt.total_seconds()
    avg_sampling_interval = time_diffs.mean()
    fs = 1 / avg_sampling_interval



    # Plot the original and reconstructed data (x, y, z axes)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='Original x')
    plt.subplot(3, 1, 2)
    plt.plot(t, y, label='Original y')
    plt.subplot(3, 1, 3)
    plt.plot(t, z, label='Original z')
    plt.show()

    # Compute PSD for each channel
    frequencies, psd_x = welch(x, fs, nperseg=256)
    _, psd_y = welch(y, fs, nperseg=256)
    _, psd_z = welch(z, fs, nperseg=256)

    # Stack PSDs into a matrix (frequency bins x channels)
    psd_matrix = np.vstack([psd_x, psd_y, psd_z]).T

    # Perform SVD on the PSD matrix
    U, Sigma, VT = np.linalg.svd(psd_matrix, full_matrices=False)

    # Reconstruct PSD matrix with reduced dimensions (retain top k singular values)
    k = 1  # Retain only the dominant spectral feature
    Sigma_reduced = np.zeros_like(Sigma)
    Sigma_reduced[:k] = Sigma[:k]
    psd_matrix_reconstructed = (U[:, :k] * Sigma_reduced[:k]) @ VT[:k, :]

    # Plot original and reconstructed PSD (x-axis channel as an example)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd_matrix[:, 0], label="Original PSD (x-axis)")
    plt.plot(frequencies, psd_matrix_reconstructed[:, 0], '--', label="Reconstructed PSD (x-axis)")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency')
    plt.legend()
    plt.title('SVD on PSD: Dimensionality Reduction')
    plt.show()

def svd_all_devices():
    pass


if __name__ == '__main__':
    svd_dimensionality_reduction()