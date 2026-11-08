# Machine Learning Assignment 2 — Dry Bean Classification

**BITS ID:** 2026AK05139  
**Programme:** M.Tech (AIML), BITS Pilani WILP  
**Course:** Machine Learning  
**Dataset:** UCI Dry Bean Dataset (UCI ID: 602)  
**Problem type:** Seven-class classification


## 1. Problem statement

The objective is to develop and compare five supervised classification algorithms to identify the variety of a dry bean from geometric measurements extracted from bean images. The five classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine.

## 2. Dataset description

The UCI Dry Bean dataset contains 13,611 instances and 16 numerical input features. The target is `Class`, with seven categories: Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira. The UCI repository reports no missing values. The dataset meets the assignment requirements of at least 12 features and 500 instances.

Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
Citation: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B  
License: CC BY 4.0. Please retain attribution.

## 3. GitHub repository link


`https://github.com/robin-paul07/ML-Assignment-2-Dry-Bean`

## 4. Live Streamlit application link


`https://dry-bean-classification-2026.streamlit.app`


## 5. Models used and evaluation metrics

The implementation calculates the six required measures for each model:

- Accuracy
- AUC Score (multiclass One-vs-Rest, macro average)
- Precision (weighted average)
- Recall (weighted average)
- F1 Score (weighted average)
- Matthews Correlation Coefficient (MCC)


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.920676 | 0.994768 | 0.921533 | 0.920676 | 0.920873 | 0.904167 |
| Decision Tree | 0.888358 | 0.953088 | 0.888243 | 0.888358 | 0.887862 | 0.865178 |
| KNN | 0.916636 | 0.983317 | 0.917409 | 0.916636 | 0.916805 | 0.899204 |
| Naive Bayes | 0.897907 | 0.991614 | 0.900702 | 0.897907 | 0.898075 | 0.877263 |
| Support Vector Machine | 0.922145 | 0.994927 | 0.922426 | 0.922145 | 0.922177 | 0.905870 |

Recall equals Accuracy in every row because weighted-average recall is mathematically the same as accuracy.

### Observations on model performance


| Model | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy of 0.9207 is within 0.0015 of the best model (SVM), and its AUC (0.9948) is nearly identical. A linear model doing this well suggests the classes are largely separable once features are scaled. Its confusion matrix shows Bombay classified perfectly (104 of 104), while Dermason and Sira are confused most (53 Dermason predicted as Sira, 43 Sira as Dermason). Sira has the lowest per-class F1 (0.8625). |
| Decision Tree | Lowest on every metric (accuracy 0.8884, MCC 0.8652), about 3.4 percentage points below the best model. Its AUC (0.9531) is far below the others. A single tree gives coarse probability estimates, so it ranks classes less well. |
| KNN | Accuracy of 0.9166 is only about 0.4 points below Logistic Regression, but its AUC (0.9833) is clearly lower. Being distance-based, it relies on the feature scaling in the pipeline and is less accurate where classes overlap. |
| Naive Bayes | Accuracy is low (0.8979) but AUC is high (0.9916), so it ranks classes well yet makes weaker final decisions. A likely cause is that Area, Perimeter, ConvexArea, EquivDiameter and MajorAxisLength are strongly correlated, which breaks its independence assumption. |
| Support Vector Machine | Best on accuracy (0.9221), AUC (0.9949), F1 (0.9222) and MCC (0.9059). Its lead over Logistic Regression is 0.0015 in accuracy, about 4 of 2,723 test samples, so the two are practically tied. |
| Overall winner | **Support Vector Machine** on all headline metrics. Logistic Regression is a near-equal and simpler alternative and both clearly beat the Decision Tree and Naive Bayes. |


## 6. Repository contents

- `train_models.py` — downloads the UCI dataset, performs a stratified train/test split, trains and evaluates all five models, and saves artifacts.
- `app.py` — Streamlit user interface for test CSV upload, model selection, evaluation, confusion matrix, classification report and model comparison.
- `requirements.txt` — Python dependencies.
- `model/` — saved fitted models and label encoder (created by training).
- `test_data.csv` — held-out test records with target labels (created by training).
- `metrics_summary.csv` — training-script evaluation output (created by training).
- `model_metadata.json` — feature names and label metadata (created by training).
- `ML_Assignment_2.ipynb`- Jupyter notebook used to run the training and evaluation in BITS Virtual Lab

## 7. How to run in BITS Virtual Lab

1. Open a Python terminal or notebook environment in BITS Virtual Lab.
2. Upload/clone this repository and open its project directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Train and save models: `python train_models.py`
5. Inspect `metrics_summary.csv`, `test_data.csv` and saved model files.
6. Start the application: `streamlit run app.py`
7. Upload `test_data.csv` in the app and test each classifier.

The training script uses a stratified 80:20 train/test split with random state 42. StandardScaler is included inside the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is used without scaling. All preprocessing required at inference time is stored within the fitted pipeline.

## 8. Streamlit application

The application implements the assignment's required features:

1. CSV upload for held-out test data.
2. Dropdown selection among all five saved classifiers.
3. Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix and classification report.

It also compares all five classifiers on the uploaded test CSV and displays sample predictions. The application validates required columns and known class labels. It loads saved models and does not retrain on uploaded data.

**Important:** The CSV must contain all 16 input feature columns and the `Class` target column. Use the `test_data.csv` generated by `train_models.py` as the reference format.


## 9. BITS Virtual Lab execution evidence

The BITS Virtual Lab execution screenshot is included in the submitted PDF.


## 10. Reproducibility and limitations

- The split is stratified and reproducible using random state 42.
- AUC is computed using multiclass One-vs-Rest macro averaging.
- Precision, Recall and F1 use weighted averaging.
- MCC is computed using the multiclass implementation in scikit-learn.
- Model metrics are calculated on the held-out test data.
- The dataset is fetched from UCI when `train_models.py` is run, so the environment needs internet access for the initial download.
- All results were generated by running the code in BITS Virtual Lab.
