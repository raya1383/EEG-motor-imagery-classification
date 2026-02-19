import os
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(X, fs, lowf, highf, order = 4):
    nyq = 0.5 * fs
    low = lowf / nyq
    high = highf / nyq

    b, a = butter(order, [low,high], btype = "bandpass")
    X_filt = np.zeros_like(X, dtype = np.float64)

    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_filt[i, ch] = filtfilt(b, a, X[i, ch])

    return X_filt

def main():
    in_path = os.path.join("data", "processed", "ds1a_extracted.npz")
    out_path = os.path.join("data", "processed", "ds1a_filtered_bands.npz")

    data = np.load(in_path, allow_pickle = True)

    X = data["X"]
    y = data["y"]
    fs = float(data["fs"])
    print("Loaded dataset:", X.shape)
    
    print("Filtering 8-30 Hz...")
    X_8_30 = bandpass_filter(X, fs, 8, 30)

    print("Filering 8-13 Hz...")
    X_mu = bandpass_filter(X, fs, 8, 13)

    print("Filtering 13-30 Hz...")
    X_beta = bandpass_filter(X, fs, 13, 30)

    np.savez(
        out_path,
        X_8_30 = X_8_30,
        X_mu = X_mu,
        y = y,
        fs = fs

    )

    print("Saved filtered datasets to:", out_path)

if __name__ == "__main__":
        main()