"""
Train five classifiers on the UCI Dry Bean dataset.
Run this script in BITS Virtual Lab before launching the Streamlit app.
It creates model/*.joblib, test_data.csv, metrics_summary.csv and model_metadata.json.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20

def metric_values(y_true, y_pred, y_score, n_classes):
    # Weighted averaging is used for multiclass precision/recall/F1.
    # OvR macro AUC gives each class equal importance.
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_score,
            multi_class="ovr", average="macro",
            labels=list(range(n_classes))
        ),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def main():
    print("Fetching UCI Dry Bean dataset (UCI repository ID: 602)...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.iloc[:, 0].astype(str)

    # Validate assignment dataset constraints.
    if X.shape[1] < 12 or len(X) < 500:
        raise ValueError("Dataset does not meet the assignment minimums.")
    if X.isnull().any().any() or y_raw.isnull().any():
        raise ValueError("Unexpected missing values. Add a documented imputation step.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
    }

    rows = []
    for name, estimator in models.items():
        print(f"Training {name}...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        values = metric_values(y_test, predictions, probabilities, len(label_encoder.classes_))
        rows.append({"ML Model": name, **values})
        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "KNN": "knn",
            "Naive Bayes": "naive_bayes",
            "Support Vector Machine": "support_vector_machine",
        }[name]
        joblib.dump(estimator, MODEL_DIR / f"{safe_name}.joblib")

    # Keep actual target labels in the uploaded CSV; the app encodes them consistently.
    test_output = X_test.reset_index(drop=True).copy()
    test_output["Class"] = label_encoder.inverse_transform(y_test)
    test_output.to_csv(ROOT / "test_data.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "metrics_summary.csv", index=False)
    metadata = {
        "dataset": "UCI Dry Bean Dataset",
        "uci_id": 602,
        "target_column": "Class",
        "feature_columns": list(X.columns),
        "class_names": list(label_encoder.classes_),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_instances": int(len(X)),
        "auc_method": "One-vs-Rest macro AUC",
        "averaging": "weighted for precision, recall and F1",
    }
    (ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    print("\nEvaluation results:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCreated model files, test_data.csv, metrics_summary.csv and model_metadata.json.")
    print("Review the results and generated files before submission.")

if __name__ == "__main__":
    main()
