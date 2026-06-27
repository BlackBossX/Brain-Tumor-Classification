# 🧠 Brain Tumor Type Classification using Neural Networks

An end-to-end deep learning project for multi-class brain tumor classification using a Feed-Forward Neural Network (Multilayer Perceptron). This project demonstrates the complete machine learning workflow, from data cleaning and pipeline serialization to model training, evaluation, and inference on unseen test data.

---

## 📌 Overview

Brain tumor diagnosis is a critical task in healthcare where accurate classification can support medical professionals in making informed decisions. This project aims to classify patient cases into one of three tumor types using a Multilayer Perceptron (MLP).

### Target Classes (Three-Class Classification)
*   **Glioma** (42.2% of dataset)
*   **Meningioma** (34.3% of dataset)
*   **Pituitary** (23.4% of dataset)

The project includes the complete pipeline required for a production-ready machine learning solution, including preprocessing, model training, evaluation, hyperparameter tuning, and inference on unseen data.

---

## 🔑 Core ML Principles & Safeguards

This project is built around the fundamental rule of machine learning:
> **Anything you learn from data must only be learned from training data, and then applied unchanged elsewhere.**

To respect this rule and prevent **Data Leakage**, the following architecture is implemented:
1.  **Split Before Processing:** The dataset is partitioned into Train, Validation, and Test subsets *before* fitting any preprocessing transformations.
2.  **Fit on Train Only:** Imputers, encoders, and scalers are fitted **only** on the training partition.
3.  **Apply Unchanged:** Validation and Test sets are transformed using the pre-fitted parameters without recalculating medians, means, or category lists.
4.  **Pipeline Serialization:** All fitted preprocessors are serialized to disk (`.pkl`) and loaded in the inference script `predict.py` to ensure reproducibility on unseen examiner data.

---

## 📊 Preprocessing & Data Cleaning Details

Based on direct profiling of the raw `brain_tumor_data.csv` dataset, several data quality issues must be resolved during preprocessing:

### 1. Identifier Column
*   `patient_id` (e.g. `PT100001`): Unique identifier for each row. **This must be dropped before modeling** because it carries no biological signal and risks causing the model to overfit by memorizing patient IDs.

### 2. Missing-Value Markers
While standard empty CSV cells are automatically parsed as `NaN` by pandas, two categorical columns contain non-standard missing-value markers:
*   **`alcohol_consumption`**: Contains the literal string `"Unknown"` in **381 rows**.
*   **`genetic_marker_status`**: Contains the literal string `"? "` / `"?"` in **247 rows**.

These must be replaced with `np.nan` before calculating final missing counts and applying the imputers:
```python
df['alcohol_consumption'] = df['alcohol_consumption'].replace('Unknown', np.nan)
df['genetic_marker_status'] = df['genetic_marker_status'].replace('?', np.nan)
```

### 3. Casing Inconsistency
*   **`gender`**: Contains four distinct strings: `'Male'`, `'Female'`, `'male'`, and `'female'`. These must be normalized to prevent redundant one-hot encoding columns:
```python
df['gender'] = df['gender'].str.capitalize()
```

### 4. True Missing Value Summary (After Normalizing Markers)
*   `ethnicity`: 281 missing
*   `bmi`: 436 missing (numeric)
*   `smoking_status`: 381 missing
*   `alcohol_consumption`: 3,804 missing (381 `"Unknown"` + 3,423 blanks)
*   `family_history`: 356 missing
*   `tumor_growth_rate`: 540 missing (numeric)
*   `headache_severity`: 457 missing (numeric)
*   `mri_intensity`: 356 missing (numeric)
*   `ct_density`: 434 missing (numeric)
*   `edema_grade`: 283 missing (numeric)
*   `contrast_enhancement`: 1,136 missing
*   `ki67_index`: 664 missing (numeric)
*   `bp_diastolic`: 197 missing (numeric)
*   `wbc_count`: 253 missing (numeric)
*   `crp_level`: 445 missing (numeric)
*   `genetic_marker_status`: 247 missing (all from the `"?"` marker)

---

## 🛠 Preprocessing Pipeline Strategy

```text
       Raw Data Row
            │
            ▼
      String Cleanup
 ┌──────────────────────┐
 │ • Gender Casing      │
 │ • Replace 'Unknown'  │
 │ • Replace '?'        │
 └──────────┬───────────┘
            │
            ├────────────────────────┐
            ▼                        ▼
     [Numeric Path]          [Categorical Path]
 ┌─────────────────────┐  ┌─────────────────────┐
 │ Median Imputation   │  │ Mode Imputation     │
 └──────────┬──────────┘  └──────────┬──────────┘
            │                        │
            │                        ├────────────────────────┐
            │                        ▼                        ▼
            │               [Nominal Features]       [Ordinal Features]
            │             ┌─────────────────────┐  ┌─────────────────────┐
            │             │ One-Hot Encoding    │  │ Custom Ordinal Map  │
            │             │ (Ignore Unseen)     │  │ (None < Mid < Max)  │
            │             └──────────┬──────────┘  └──────────┬──────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     ▼
                              Feature Concat
                                     │
                                     ▼
                              StandardScaler
                                     │
                                     ▼
                             Final Input Vector
```

### Feature Encoding Choice Justifications
*   **One-Hot Encoding (Nominal):** Applied to `gender`, `ethnicity`, `region`, `smoking_status`, `family_history`, `tumor_location`, and `genetic_marker_status` because they have no intrinsic order. Binary variables (e.g. Yes/No symptoms) can be mapped directly to `0`/`1`.
    *   *Important:* Use `OneHotEncoder(handle_unknown='ignore')` so the model does not crash if it encounters unknown categories in validation or test data.
*   **Ordinal Encoding (Ordinal):** Applied to:
    *   `alcohol_consumption`: `None` (0) < `Moderate` (1) < `Heavy` (2)
    *   `contrast_enhancement`: `None` (0) < `Mild` (1) < `Moderate` (2) < `Strong` (3)
    *   *Note:* `edema_grade` (0, 1, 2, 3) is already numeric and ordinal.
*   **Target Encoding:** Encode `tumor_type` into integers (`0` = Glioma, `1` = Meningioma, `2` = Pituitary) and compile using `sparse_categorical_crossentropy` loss. Alternatively, one-hot encode target labels and use `categorical_crossentropy`.

### Imputation & Scaling
*   **Numeric Imputer:** Use a `SimpleImputer(strategy='median')`. The median is robust to outliers and skewed distributions present in biological markers like `ki67_index` (0.1–60) and `crp_level`.
*   **Categorical Imputer:** Use a `SimpleImputer(strategy='most_frequent')` or replace missing values with a constant label like `'Missing'`.
*   **Scaling:** Scale numeric features with `StandardScaler` (zero mean, unit variance). Scaling prevents features with larger units (e.g., `mri_intensity` ranging from 0–255) from dominating gradients over features with smaller ranges (e.g., `headache_severity` ranging from 0–10).

---

## 🧠 Neural Network Model Architecture

The feed-forward neural network is designed with the following structure:
*   **Input Layer:** Dimension matches the number of features after one-hot expansion.
*   **Hidden Layer 1:** 64 units with ReLU activation, followed by `Dropout(0.3)`.
*   **Hidden Layer 2:** 32 units with ReLU activation, followed by `Dropout(0.3)`.
*   **Output Layer:** 3 units with **Softmax** activation to output a probability distribution over the 3 classes.
*   **Loss Function:** `sparse_categorical_crossentropy` (assuming integer target labels).

### Overfitting Prevention Strategy
1.  **Dropout:** Standardizes feature dependency by randomly disabling 30% of neurons during training batches.
2.  **Early Stopping:** Monitors `val_loss` with a patience of 5–10 epochs and restores the best weights.
3.  **Weight Regularization:** Optional L2 regularization (`kernel_regularizer=l2(0.001)`) to penalize large weights.

---

## 📊 Splitting & Training Protocol

*   **Data Partitioning:** 70% Train, 15% Validation, 15% Test.
*   **Stratified Splits:** Splitting is stratified by `tumor_type` (`stratify=y` in scikit-learn) to preserve the 42/34/23 class ratio across all splits.
*   **Optimization:** `Adam` optimizer with a default learning rate of `0.001` and a batch size of `32`.
*   **Tuning Pipeline:** Perform experiments by varying the number of hidden layers, neuron counts, learning rate, batch size, and dropout rate. **Configurations are compared using Validation Performance only.**

---

## 📈 Evaluation & Reflection

### 1. Test Performance
The held-out test subset is evaluated **exactly once** at the very end of the project to report:
*   Accuracy
*   Precision, Recall, and F1-Score per class
*   **Macro F1-average** (treats classes equally, highlight performance on the minority class) and **Weighted F1-average** (accounts for class imbalance).

### 2. Confusion Matrix Analysis
*   Analyze class confusions. Expect the model to occasionally confuse Glioma and Meningioma due to overlapping clinical indicators, whereas Pituitary should be highly distinct due to anatomical specificities (e.g., location restricted to `Sellar`).

### 3. Baseline Comparison
Compare test performance against a traditional machine learning classifier (e.g., `RandomForestClassifier` or `LogisticRegression`) to evaluate if the neural network adds value relative to its complexity.

---

## 📂 Project Structure

```text
Brain-Tumor-Type-Classification/
│
├── dataset/
│   ├── brain_tumor_data.csv            # Raw dataset (9,000 cases)
│   └── data_dictionary.txt             # Original data dictionary
│
├── docs/                               # Assignment documentation and background guides
│   ├── 00_big_picture.md
│   ├── 01_your_data_explained.md
│   ├── 02_workflow_roadmap.md
│   ├── 03_overfitting_playbook.md
│   └── 04_keras_concepts_for_beginners.md
│
├── models/
│   ├── brain_tumor_model.h5            # Saved Keras model
│   └── preprocessing_pipeline.pkl      # Saved preprocessors (imputers, encoders, scalers)
│
├── notebooks/
│   └── studentID_NN.ipynb              # Runnable coursework notebook
│
├── predict.py                          # Predict script for raw held-out data
├── requirements.txt                    # Project requirements
└── README.md                           # Project documentation
```

---

## 🚀 Running the Project

### 1. Install Dependencies
Make sure you have created your virtual environment, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Launch Jupyter Notebook to inspect or run the pipeline:
```bash
jupyter notebook notebooks/studentID_NN.ipynb
```

### 3. Run Predictions on New Raw Data
To predict classes on raw patient files (in the same format as `brain_tumor_data.csv`):
```bash
python predict.py --input path/to/new_patient_data.csv
```

---

## 🔮 Model Inference (`predict.py`)

The inference script `predict.py` takes a raw CSV path, processes it without manual intervention or data leakage, and outputs class labels:
1.  Loads the pre-saved preprocessing pipeline containing the fitted imputers, encoders, and scalers.
2.  Normalizes column anomalies (casing, `"?"`, `"Unknown"` markers).
3.  Transforms raw features using the pre-saved preprocessing pipeline.
4.  Loads the trained neural network model (`studentID_model.h5`).
5.  Executes the forward pass and returns predictions.
