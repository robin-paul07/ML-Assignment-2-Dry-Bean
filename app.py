from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classification Studio", page_icon="🫘", layout="wide")
st.title("Dry Bean Classification Studio")
st.caption("Compare five trained machine-learning classifiers on the UCI Dry Bean dataset.")

metadata_path = ROOT / "model_metadata.json"
if not metadata_path.exists():
    st.error("Trained artifacts are missing. Run `python train_models.py` first.")
    st.stop()

metadata = json.loads(metadata_path.read_text())
encoder_path = MODEL_DIR / "label_encoder.joblib"
if not encoder_path.exists():
    st.error("Label encoder is missing. Run `python train_models.py` first.")
    st.stop()

label_encoder = joblib.load(encoder_path)
feature_columns = metadata["feature_columns"]
class_names = metadata["class_names"]
n_classes = len(class_names)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.subheader("1. Upload test dataset")
st.write("Upload a CSV containing all 16 required input features and the actual `Class` target column.")
uploaded = st.file_uploader("Choose a test CSV file", type=["csv"])

if uploaded is None:
    st.info("For a first run, use the generated `test_data.csv` in this project folder.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

required = feature_columns + [metadata["target_column"]]
missing = [col for col in required if col not in data.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_test = data[feature_columns].copy()
y_text = data[metadata["target_column"]].astype(str)
unknown = sorted(set(y_text) - set(class_names))
if unknown:
    st.error(f"Unknown target labels found: {unknown}. Expected: {class_names}")
    st.stop()
if X_test.isnull().any().any():
    st.error("Missing values found in input features. Please clean the test CSV.")
    st.stop()
try:
    X_test = X_test.astype(float)
    y_true = label_encoder.transform(y_text)
except Exception as exc:
    st.error(f"Invalid feature or target values: {exc}")
    st.stop()

st.success(f"Loaded {len(data):,} test rows with {len(feature_columns)} features.")
st.subheader("2. Select a classifier")
selected_name = st.selectbox("Choose model", list(MODEL_FILES.keys()))

def evaluate(model, X, y):
    pred = model.predict(X)
    scores = model.predict_proba(X)
    return pred, scores, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores, multi_class="ovr", average="macro",
                             labels=list(range(n_classes))),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

selected_model = load_model(MODEL_DIR / MODEL_FILES[selected_name])
try:
    predictions, probabilities, scores = evaluate(selected_model, X_test, y_true)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader(f"3. Evaluation metrics — {selected_name}")
cols = st.columns(6)
for col, (metric, value) in zip(cols, scores.items()):
    col.metric(metric, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, predictions, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(n_classes), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, predictions, labels=list(range(n_classes)),
        target_names=class_names, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("4. Compare all five models on this uploaded test data")
comparison = []
for name, filename in MODEL_FILES.items():
    model = load_model(MODEL_DIR / filename)
    _, _, model_scores = evaluate(model, X_test, y_true)
    comparison.append({"ML Model": name, **model_scores})
comparison_df = pd.DataFrame(comparison).set_index("ML Model")
st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)
st.bar_chart(comparison_df[["Accuracy", "AUC", "F1", "MCC"]])

st.subheader("5. Sample predictions")
preview = X_test.head(10).copy()
preview["Actual Class"] = label_encoder.inverse_transform(y_true[:len(preview)])
preview["Predicted Class"] = label_encoder.inverse_transform(predictions[:len(preview)])
st.dataframe(preview, use_container_width=True)

st.download_button(
    "Download predictions CSV",
    data=preview.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions_preview.csv",
    mime="text/csv"
)
st.caption("Models are loaded from saved Joblib files. Uploaded test data is evaluated only; no retraining occurs.")
