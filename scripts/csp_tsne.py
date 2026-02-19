import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BAND_KEY = "X_8_30"


N_PAIRS = 6
REG = 1e-10
SEED = 42

def cov_trace_norm(trial):
    c = trial @ trial.T
    tr = np.trace(c)
    return c / tr if tr > 0 else c

def mean_cov(X):
    return np.mean([cov_trace_norm(tr) for tr in X], axis = 0)

def whitener(C, eps = 1e-12):
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, eps)
    D_inv_sqrt = np.diag(1.0/np.sqrt(evals))
    P = D_inv_sqrt @ evecs.T
    return P

def fit_csp(X, y, n_pairs = 4, reg = 1e-10):
    X0 = X[y == 0]
    X1 = X[y == 1]
    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("Need trials from both classes to fit CSP")
    
    C0 = mean_cov(X0) + reg * np.eye(X.shape[1])
    C1 = mean_cov(X1) + reg * np.eye(X.shape[1])
    C = C0 + C1

    P = whitener(C)
    S0 = P @ C0 @ P.T

    evals, B = np.linalg.eigh(S0)

    idx = np.argsort(evals)
    B = B[:, idx]

    W_full = (B.T @ P)
    
    picks = np.r_[0:n_pairs, -n_pairs:0]
    W = W_full[picks, :].T

    return W

def transform_csp(X, W):
    F = np.zeros((X.shape[0], W.shape[1]), dtype = np.float64)
    for i in range(X.shape[0]):
        Z = W.T  @ X[i]
        var = np.var(Z, axis = 1)
        F[i] = np.log(var + 1e-12)
    return F

def train_test_split_idx(n, train_ratio=0.75, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train_ratio * n))
    return idx[:n_train], idx[n_train:]


def tsne_2d(features, seed=42):
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto"
    ).fit_transform(features)


def plot_tsne(Z, y, title):
    plt.figure()
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=0.7, label="class 0")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=0.7, label="class 1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load filtered data
    in_path = os.path.join("data", "processed", "ds1a_filtered_bands.npz")
    d = np.load(in_path, allow_pickle=True)

    X = d[BAND_KEY]            # (trials, channels, samples)
    X_mu = d["X_mu"]
    X_beta = d["X_beta"]
    y = d["y"].astype(int)     # (trials,)
    
    print("Loaded:", BAND_KEY, X.shape, "y:", y.shape)
    print("Loaded: X_mu", X_mu.shape, "X_beta", X_beta.shape)

    # Split first to avoid leakage
    train_idx, test_idx = train_test_split_idx(len(y), train_ratio=0.75, seed=SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_mu_train = X_mu[train_idx]
    X_mu_test = X_mu[test_idx]
    X_beta_train = X_beta[train_idx]
    X_beta_test = X_beta[test_idx]

    # ---------------- t-SNE BEFORE CSP ----------------

    F_before = np.log(np.var(X_train, axis=2) + 1e-12)
    Z_before = tsne_2d(F_before, seed=SEED)
    plot_tsne(Z_before, y_train, f"t-SNE BEFORE CSP ({BAND_KEY}: per-channel log-variance)")

    # ---------------- CSP on 8-30  ----------------
    W = fit_csp(X_train, y_train, n_pairs=N_PAIRS, reg=REG)
    F_train = transform_csp(X_train, W)
    F_test = transform_csp(X_test, W)

    print("CSP 8-30 features:")
    print("F_train:", F_train.shape, "F_test:", F_test.shape)
    #----------------- CSP on mu and beta seperate + concatenate ---------
    W_mu = fit_csp(X_mu_train, y_train, n_pairs=N_PAIRS, reg=REG)
    W_beta = fit_csp(X_beta_train, y_train, n_pairs=N_PAIRS, reg=REG)

    F_mu_train = transform_csp(X_mu_train, W_mu)
    F_mu_test = transform_csp(X_mu_test, W_mu)

    F_beta_train = transform_csp(X_beta_train, W_beta)
    F_beta_test = transform_csp(X_beta_test, W_beta)

    F_mubeta_train = np.concatenate([F_mu_train, F_beta_train], axis=1)
    F_mubeta_test = np.concatenate([F_mu_test, F_beta_test], axis=1)

    print("CSP Mu+Beta concatenated features:")
    print("F_mubeta_train:", F_mubeta_train.shape, "F_mubeta_test:", F_mubeta_test.shape)

    Z_mubeta = tsne_2d(F_mubeta_train, seed=SEED)
    plot_tsne(Z_mubeta, y_train, "t-SNE AFTER CSP (Mu CSP + Beta CSP concatenated)")
    # ---------------- t-SNE AFTER CSP ----------------
    Z_after = tsne_2d(F_train, seed=SEED)
    plot_tsne(Z_after, y_train, f"t-SNE AFTER CSP ({BAND_KEY}: CSP log-variance features)")

    # Save CSP features for next step (classification)
    out_path = os.path.join("data", "processed", "ds1a_csp_features.npz")
    np.savez(
        out_path,
        band_key=BAND_KEY,
        train_idx=train_idx,
        test_idx=test_idx,
        y=y,
        F_train=F_train,
        y_train=y_train,
        F_test=F_test,
        y_test=y_test,
        F_mubeta_train=F_mubeta_train,
        F_mubeta_test=F_mubeta_test,
        n_pairs=N_PAIRS
    )
    print("Saved CSP features to:", out_path)


if __name__ == "__main__":
    main()