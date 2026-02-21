import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    RocCurveDisplay,
    classification_report
)
@dataclass
class Standardizer:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    # K(x,z) = exp(-gamma * ||x-z||^2)
    # Efficient pairwise squared distances:
    X1_sq = np.sum(X1 * X1, axis=1, keepdims=True)      # (n1,1)
    X2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T    # (1,n2)
    dist2 = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
    return np.exp(-gamma * dist2)


@dataclass
class SVMRBF_SM0:
    C: float = 1.0
    gamma: float = 0.1
    tol: float = 1e-3
    max_passes: int = 10
    max_iters: int = 20000
    seed: int = 42

    # learned params
    alphas_: Optional[np.ndarray] = None
    b_: float = 0.0
    X_sv_: Optional[np.ndarray] = None
    y_sv_: Optional[np.ndarray] = None
    alpha_sv_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        y = y.astype(float)

        # precompute full kernel matrix
        K = rbf_kernel(X, X, self.gamma)

        alphas = np.zeros(n, dtype=float)
        b = 0.0

        def f(i):
            return np.sum(alphas * y * K[:, i]) + b

        passes = 0
        iters = 0

        while passes < self.max_passes and iters < self.max_iters:
            num_changed = 0
            for i in range(n):
                Ei = f(i) - y[i]

       
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
          
                    j = i
                    while j == i:
                        j = rng.integers(0, n)
                    Ej = f(j) - y[j]

                    ai_old = alphas[i]
                    aj_old = alphas[j]

               
                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if abs(H - L) < 1e-12:
                        continue

    
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    aj_new = aj_old - (y[j] * (Ei - Ej)) / eta
                    aj_new = np.clip(aj_new, L, H)

                    if abs(aj_new - aj_old) < 1e-5:
                        continue
                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                    b1 = b - Ei - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]

                    if 0 < ai_new < self.C:
                        b = b1
                    elif 0 < aj_new < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    alphas[i] = ai_new
                    alphas[j] = aj_new
                    num_changed += 1

                iters += 1
                if iters >= self.max_iters:
                    break

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        sv_mask = alphas > 1e-8
        self.alphas_ = alphas
        self.b_ = b
        self.X_sv_ = X[sv_mask]
        self.y_sv_ = y[sv_mask]
        self.alpha_sv_ = alphas[sv_mask]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K = rbf_kernel(self.X_sv_, X, self.gamma)  # (n_sv, n)
        scores = (self.alpha_sv_ * self.y_sv_) @ K + self.b_
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)  # map to {0,1}

def evaluate_binary(y_true01: np.ndarray, y_pred01: np.ndarray, scores: np.ndarray, title_prefix: str):
    acc = accuracy_score(y_true01, y_pred01)
    prec = precision_score(y_true01, y_pred01, zero_division=0)
    rec = recall_score(y_true01, y_pred01, zero_division=0)
    f1 = f1_score(y_true01, y_pred01, zero_division=0)

    print(f"\n[{title_prefix}] Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true01, y_pred01, digits=4, zero_division=0))

    cm = confusion_matrix(y_true01, y_pred01)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{title_prefix} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_true01, scores)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"{title_prefix} - ROC (AUC={roc_auc:.4f})")
    plt.tight_layout()
    plt.show()

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": roc_auc}


def simple_grid_search_custom_svm(
    Xtr: np.ndarray, ytr01: np.ndarray,
    C_list=(0.1, 1.0, 10.0, 100.0),
    gamma_list=(0.01, 0.1, 1.0, 10.0),
    seed=42
) -> Tuple[float, float, Dict]:

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr01))
    rng.shuffle(idx)
    split = int(round(0.8 * len(idx)))
    fit_idx, val_idx = idx[:split], idx[split:]

    X_fit, y_fit = Xtr[fit_idx], ytr01[fit_idx]
    X_val, y_val = Xtr[val_idx], ytr01[val_idx]

    y_fit_pm = np.where(y_fit == 1, 1.0, -1.0)

    best = {"acc": -1.0, "C": None, "gamma": None}
    for C in C_list:
        for gamma in gamma_list:
            model = SVMRBF_SM0(C=C, gamma=gamma, seed=seed, max_passes=10, max_iters=20000)
            model.fit(X_fit, y_fit_pm)
            val_scores = model.decision_function(X_val)
            val_pred = (val_scores >= 0).astype(int)
            acc = accuracy_score(y_val, val_pred)
            if acc > best["acc"]:
                best.update({"acc": acc, "C": C, "gamma": gamma})

    return best["C"], best["gamma"], best

def run_one_branch(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    print("\n" + "=" * 80)
    print(f"BRANCH: {name}")
    print("=" * 80)

    # 1) Standardize (fit on train only)
    scaler = Standardizer()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # 2) Hyperparam selection on train only (simple holdout inside train)
    C_best, gamma_best, best_info = simple_grid_search_custom_svm(Xtr, y_train, seed=42)
    print(f"\n[{name}] Best (custom grid): C={C_best}, gamma={gamma_best}, val_acc={best_info['acc']:.4f}")

    # 3) Train CUSTOM SVM from scratch on full train
    ytr_pm = np.where(y_train == 1, 1.0, -1.0)
    custom = SVMRBF_SM0(C=C_best, gamma=gamma_best, seed=42, max_passes=15, max_iters=50000)
    custom.fit(Xtr, ytr_pm)

    custom_scores = custom.decision_function(Xte)         
    custom_pred = (custom_scores >= 0).astype(int)

    custom_metrics = evaluate_binary(y_test, custom_pred, custom_scores, f"{name} | Custom RBF-SVM (SMO)")

    # 4) Train sklearn SVC with SAME params and compare
    sk = SVC(kernel="rbf", C=C_best, gamma=gamma_best)
    sk.fit(Xtr, y_train)

    sk_scores = sk.decision_function(Xte)
    sk_pred = sk.predict(Xte).astype(int)

    sk_metrics = evaluate_binary(y_test, sk_pred, sk_scores, f"{name} | scikit-learn SVC(RBF)")

    # 5) Compare side-by-side summary
    print(f"\n[{name}] SUMMARY COMPARISON (Test)")
    for k in ["acc", "prec", "rec", "f1", "auc"]:
        print(f"{k.upper():>4} | custom={custom_metrics[k]:.4f}   sklearn={sk_metrics[k]:.4f}")

    return {
        "best_C": C_best,
        "best_gamma": gamma_best,
        "custom": custom_metrics,
        "sklearn": sk_metrics
    }


def main():
  
    path = os.path.join("data", "processed", "ds1a_csp_features.npz")
    d = np.load(path, allow_pickle=True)

    # A) 8-30 CSP features
    F_train = d["F_train"]
    y_train = d["y_train"].astype(int)
    F_test = d["F_test"]
    y_test = d["y_test"].astype(int)

    # B) mu+beta concatenated CSP features
    F_mb_train = d["F_mubeta_train"]
    F_mb_test = d["F_mubeta_test"]

    print("Loaded features:")
    print("  F_train:", F_train.shape, "y_train:", y_train.shape)
    print("  F_test :", F_test.shape, "y_test :", y_test.shape)
    print("  F_mb_train:", F_mb_train.shape, "F_mb_test:", F_mb_test.shape)

    # Run BOTH branches exactly as you asked
    res_A = run_one_branch("A) CSP on 8-30Hz", F_train, y_train, F_test, y_test)
    res_B = run_one_branch("B) CSP on (mu+beta) concat", F_mb_train, y_train, F_mb_test, y_test)

    print("\n" + "#" * 80)
    print("FINAL RESULT (Test) - both branches")
    print("#" * 80)
    print("A) 8-30Hz  : custom acc={:.4f}, sklearn acc={:.4f}".format(res_A["custom"]["acc"], res_A["sklearn"]["acc"]))
    print("B) mu+beta : custom acc={:.4f}, sklearn acc={:.4f}".format(res_B["custom"]["acc"], res_B["sklearn"]["acc"]))
    print("#" * 80)


if __name__ == "__main__":
    main()
