import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42

def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

def run_one_feature_set(title, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print(title)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    lda = LinearDiscriminantAnalysis()

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED)

    evaluate("LDA", lda, X_train, y_train, X_test, y_test)
    evaluate("Logistic Regression", logreg, X_train, y_train, X_test, y_test)
    evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)

def main():
    path = os.path.join("data", "processed", "ds1a_csp_features.npz")
    d = np.load(path, allow_pickle=True)

    y_train = d["y_train"].astype(int)
    y_test = d["y_test"].astype(int)

    F_train = d["F_train"]
    F_test = d["F_test"]

    F_mubeta_train = d["F_mubeta_train"]
    F_mubeta_test = d["F_mubeta_test"]

    run_one_feature_set("Feature set A: CSP on 8-30 Hz", F_train, y_train, F_test, y_test)
    run_one_feature_set("Feature set B: CSP on Mu + CSP on Beta (concatenated)", F_mubeta_train, y_train, F_mubeta_test, y_test)

if __name__ == "__main__":
    main()