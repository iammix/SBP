import pandas as pd
import pytest
from oma import baseline_correction_, baseline_correction_hnd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['Date'] + " " + df['Time'])
    df = df.rename(columns={"Acc x": "acc_x", "Acc y": "acc_y", "Acc z": "acc_z"})
    return df[['time', 'acc_x', 'acc_y', 'acc_z']]

@pytest.fixture
def sample_data():
    file_path = r'A:\Projects\SBP\test\2024_10_23_1305_RISE_01.txt'
    df = load_data(file_path)
    rec_t = df['time'].tolist()
    rec_a = df[['acc_x', 'acc_y', 'acc_z']].values
    return rec_a, rec_t, df

def test_baseline_correction(sample_data):
    rec_a, rec_t, df = sample_data

    # Method 1: Using baseline_correction_hnd
    rec_a_corrected_hnd = np.empty_like(rec_a)
    rec_a_cor = baseline_correction_hnd(rec_a, rec_t)  # Ensure this function modifies rec_a_corrected_hnd

    # Method 2: Using baseline_correction_
    df_corrected = baseline_correction_(df, linear=True)

    # Validate outputs
    for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
        assert np.allclose(
            rec_a_cor[:, i],
            df_corrected[axis].values,
            atol=1e-12
        ), f"Mismatch in {axis} axis correction"


