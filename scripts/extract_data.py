import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

cue_duration = 4.0
channels_plot = [0, 15, 30, 45, 58]

def load_mat(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found")
    
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    cnt = mat["cnt"]
    mrk = mat["mrk"]
    nfo = mat["nfo"]

    return cnt, mrk, nfo

def extract_trials(cnt: np.ndarray, mrk, fs: float, cue_duration: float):
    window_length = int(round(cue_duration * float(fs)))
    pos = np.array(mrk.pos, dtype=int).reshape(-1)
    y = np.array(mrk.y).reshape(-1)
    unique_labels = np.unique(y)

    # Mapping the labels to 0 and 1
    if set(unique_labels.tolist()) == {-1, 1}:
        y_mapped = (y == 1).astype(int)
        class_map = {-1: 0, 1: 1}
    elif set(unique_labels.tolist()) == {1, 2}:
        y_mapped = (y == 2).astype(int)
        class_map = {1: 0, 2: 1}
    else:
        y_mapped = y
        class_map = {int(k): int(k) for k in unique_labels}

    if cnt.ndim != 2:
        raise ValueError(f"Expected cnt to be 2D(samples, channels). Got shape {cnt.shape}")
    
    n_samples, n_channels = cnt.shape

    trials = []
    kept_labels = []

    for i, start in enumerate(pos):
        end = start + window_length
        if start < 0 or end > n_samples:
            continue
        seg = cnt[start:end, :]      # (samples, channels)
        trials.append(seg.T)         # (channels, samples)
        kept_labels.append(y_mapped[i])

    if not trials:
        raise RuntimeError("No trials were extracted, check your data format again")
    
    X = np.stack(trials, axis=0)  # trials, channels, samples
    y_out = np.array(kept_labels)

    return X, y_out, window_length, class_map

def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.75, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n_train = int(round(train_ratio * len(idx)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def plot_sample(X: np.ndarray, fs: float, channels_plot: list[int], sample_index: int = 0):
    # X: (trials, channels, samples)
    x = X[sample_index]  # (channels, samples)
    n_ch, n_samp = x.shape
    t = np.arange(n_samp) / float(fs)

    plt.figure()
    for ch in channels_plot:  # <-- 0-based indices
        if ch < 0 or ch >= n_ch:
            continue
        plt.plot(t, x[ch], label=f"ch {ch}")

    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title(f"one trial (index {sample_index}) - selected channels")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = os.path.join("data", "raw", "BCICIV_calib_ds1a.mat")
    cnt, mrk, nfo = load_mat(file_path)

    fs = float(nfo.fs)
    print("fs:", fs)
    print("cnt shape:", cnt.shape)

    X, y, window_length, class_map = extract_trials(cnt, mrk, fs, cue_duration)

    print("window_length (samples):", window_length)
    print("X shape:", X.shape)  # (trials, channels, samples)
    print("y shape:", y.shape)
    print("class_map:", class_map)

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75, seed=42)
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    plot_sample(X_train, fs, channels_plot, sample_index=0)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    np.savez(
        os.path.join("data", "processed", "ds1a_extracted.npz"),
        X=X, y=y,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        fs=fs, window_length=window_length,
        class_map=class_map
    )
