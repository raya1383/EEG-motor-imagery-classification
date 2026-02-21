import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 42

def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    W = evecs[:, idx[:2]]
    return Xc @ W

def init_centroids_kmeanspp(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    i0 = rng.integers(0, n)
    centroids[0] = X[i0]
    d2 = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / np.sum(d2)
        ci = rng.choice(n, p=probs)
        centroids[i] = X[ci]
        new_d2 = np.sum((X - centroids[i]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return centroids

def assign_labels(X, centroids):
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1), dists

def recompute_centroids(X, labels, k, rng):
    d = X.shape[1]
    centroids = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        mask = (labels == c)
        if np.any(mask):
            centroids[c] = X[mask].mean(axis=0)
        else:
            centroids[c] = X[rng.integers(0, X.shape[0])]
    return centroids

def kmeans_fit(X, k, n_init=10, max_iter=300, tol=1e-6, seed=42):
    rng_master = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        centroids = init_centroids_kmeanspp(X, k, rng)

        prev_inertia = np.inf
        for _ in range(max_iter):
            labels, dists = assign_labels(X, centroids)
            inertia = float(np.sum(dists[np.arange(X.shape[0]), labels]))
            centroids_new = recompute_centroids(X, labels, k, rng)

            if abs(prev_inertia - inertia) <= tol * max(1.0, prev_inertia):
                centroids = centroids_new
                break
            prev_inertia = inertia
            centroids = centroids_new

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia

def silhouette_score_numpy(X, labels):
    n = X.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.nan

    D = np.sqrt(np.maximum(0.0, np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)))

    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = labels[i]
        same = (labels == ci)
        same_count = int(np.sum(same))
        if same_count <= 1:
            s[i] = 0.0
            continue

        a = np.sum(D[i, same]) / (same_count - 1)

        b = np.inf
        for cj in uniq:
            if cj == ci:
                continue
            other = (labels == cj)
            b = min(b, np.mean(D[i, other]))

        denom = max(a, b)
        s[i] = (b - a) / denom if denom > 0 else 0.0

    return float(np.mean(s))

def plot_wcss(k_values, wcss, title):
    plt.figure()
    plt.plot(k_values, wcss, marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_silhouette(k_values, sil, title):
    plt.figure()
    plt.plot(k_values, sil, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_clusters_2d(Z, labels, title):
    plt.figure()
    for c in np.unique(labels):
        plt.scatter(Z[labels == c, 0], Z[labels == c, 1], alpha=0.7, label=f"cluster {c}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_kmeans_section(X_features, name, k_min=2, k_max=10):
    Xs = standardize(X_features.astype(np.float64))

    k_values = np.arange(k_min, k_max + 1)
    wcss = []
    sil = []

    for k in k_values:
        labels, centroids, inertia = kmeans_fit(Xs, int(k), n_init=10, max_iter=300, tol=1e-6, seed=SEED)
        wcss.append(inertia)
        sil.append(silhouette_score_numpy(Xs, labels))

    wcss = np.array(wcss, dtype=np.float64)
    sil = np.array(sil, dtype=np.float64)

    plot_wcss(k_values, wcss, f"WCSS (Elbow) - {name}")
    plot_silhouette(k_values, sil, f"Silhouette score - {name}")

    k_best = int(k_values[np.nanargmax(sil)])
    print(f"\n{name} -> best k by silhouette: {k_best}")

    labels_best, _, _ = kmeans_fit(Xs, k_best, n_init=20, max_iter=500, tol=1e-7, seed=SEED)

    Z = pca_2d(Xs)
    plot_clusters_2d(Z, labels_best, f"K-Means clusters on PCA 2D (k={k_best}) - {name}")

def main():
    path = os.path.join("data", "processed", "ds1a_csp_features.npz")
    d = np.load(path, allow_pickle=True)

    F_train = d["F_train"]
    F_mubeta_train = d["F_mubeta_train"]

    run_kmeans_section(F_train, "Feature set A: CSP on 8-30 Hz", k_min=2, k_max=10)
    run_kmeans_section(F_mubeta_train, "Feature set B: Mu CSP + Beta CSP (concatenated)", k_min=2, k_max=10)

if __name__ == "__main__":
    main()