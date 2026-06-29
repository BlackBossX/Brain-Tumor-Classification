# 🧠 Brain Tumor Classification

A machine learning project that classifies brain tumors into three categories — **Glioma**, **Meningioma**, and **Pituitary** — using a Feed-Forward Neural Network (Multilayer Perceptron) trained on synthetic clinical and demographic data.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Key ML Principles](#key-ml-principles)
- [Dependencies](#dependencies)

---

## Project Overview

| Property | Detail |
|---|---|
| **Task** | Multi-class classification (3 classes) |
| **Model** | Feed-Forward Neural Network (MLP) |
| **Framework** | TensorFlow / Keras |
| **Dataset** | Synthetic — 9,000 records, 27 features |
| **Test Accuracy** | **94.37%** |

---

## Project Structure

```
Brain Tumor Classification/
├── dataset/
│   ├── brain_tumor_data.csv       # Synthetic dataset (9,000 records)
│   └── data_dictionary.txt        # Feature descriptions
├── models/
│   ├── brain_tumor_model.h5       # Trained Keras model
│   ├── numeric_imputer.pkl        # Median imputer for numeric columns
│   ├── categorical_imputer.pkl    # Mode imputer for categorical columns
│   ├── onehot_encoder.pkl         # One-Hot encoder for nominal columns
│   ├── standard_scaler.pkl        # Standard scaler for numeric columns
│   └── label_encoder.pkl          # Serialized LabelEncoder for target decoding
├── notebooks/
│   └── BrainTumorClassification.ipynb  # Full training notebook
├── docs/
│   └── MODEL_DOCUMENTATION.md    # Detailed technical documentation
├── predict.py                     # CLI script for predicting new data
└── README.md
```

---

## Dataset

- **File:** `dataset/brain_tumor_data.csv`
- **Records:** 9,000 (synthetic)
- **Raw Features:** 27 clinical, imaging, and demographic features
- **Target:** `tumor_type` — one of `Glioma`, `Meningioma`, `Pituitary`

### Feature Categories

| Category | Examples |
|---|---|
| Demographics | `age`, `gender`, `ethnicity`, `region` |
| Lifestyle | `smoking_status`, `alcohol_consumption`, `bmi` |
| Clinical / Genetic | `family_history`, `genetic_marker_status`, `ki67_index` |
| Symptoms | `headache_severity`, `nausea`, `seizures`, `vision_problems`, `memory_loss`, `balance_issues` |
| Imaging | `mri_intensity`, `ct_density`, `tumor_size_mm`, `tumor_growth_rate`, `tumor_location`, `edema_grade`, `contrast_enhancement` |
| Lab Values | `bp_systolic`, `bp_diastolic`, `wbc_count`, `crp_level` |

> See `dataset/data_dictionary.txt` for full feature descriptions.

---

## Preprocessing Pipeline

All preprocessing steps follow **strict anti-data-leakage** principles.

### Data Split (Before Any Transformation)

```
Train : 70% → 6,300 samples
Val   : 15% → 1,350 samples
Test  : 15% → 1,350 samples
```

All encoders and scalers are **fit exclusively on training data** and applied via `.transform()` to validation and test sets.

### Pipeline Steps

1. **Manual Cleaning** — Normalize inconsistent markers (`'Unknown'`, `'?'` → `NaN`; gender casing)
2. **Imputation** — Median for numeric features (13 columns, including `edema_grade`), mode for categorical features (`SimpleImputer`)
3. **Encoding:**
   - Binary mapping for symptom flags and boolean columns
   - Ordinal mapping for severity/ranked features (e.g. `contrast_enhancement`)
   - One-Hot Encoding for nominal categoricals (`ethnicity`, `tumor_location`, `smoking_status`, `region`, `genetic_marker_status`)
4. **Standard Scaling** — Z-score normalization of numeric features (12 columns, excluding `edema_grade` which is kept ordinal)

After One-Hot Encoding, features expand from **27 → 43**.

### Serialized Pipeline Artifacts

The fitted preprocessing objects are serialized individually to the `models/` directory using `joblib`. This modularity allows the inference script to apply exact transformations to new samples without risking data shifts.

---

## Model Architecture

To combat overfitting, the model employs L2 regularization, Batch Normalization, and Dropout layers:

```
          Input (43 features)
               │
    Dense(64, ReLU, L2 = 0.001)
               │
       BatchNormalization
               │
          Dropout(0.3)
               │
    Dense(32, ReLU, L2 = 0.001)
               │
       BatchNormalization
               │
          Dropout(0.2)
               │
        Dense(3, Softmax)
               │
Output (Glioma | Meningioma | Pituitary)
```

### Training Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| **Optimizer** | Adam | Adaptive learning rate |
| **Loss** | Sparse Categorical Crossentropy | Integer-encoded class targets |
| **Max Epochs** | 100 | Ceiling; halted early by callbacks |
| **Batch Size** | 32 | Number of samples per batch |
| **Early Stopping** | patience=8 | Halts training if validation loss plateaus |
| **LR Scheduler** | patience=4, factor=0.5 | Halves learning rate on validation loss plateaus |
| **Regularization** | L2 (0.001), Dropout (0.3/0.2) | Mitigates overfitting |

---

## Results

### Test Set Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **94.37%** |
| Test Loss | 0.4569 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.99 | 0.95 | **0.97** |
| Meningioma | 0.90 | 0.95 | 0.93 |
| Pituitary | 0.94 | 0.91 | 0.92 |
| **Macro Avg** | 0.94 | 0.94 | **0.94** |

> For the full training history, learning curves, and comprehensive model metrics, see [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md).

---

## Getting Started

### Prerequisites

Install all required packages:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

### Training the Model

Open and run the Jupyter notebook end-to-end:

```bash
jupyter notebook notebooks/BrainTumorClassification.ipynb
```

This will:
1. Load and clean the dataset.
2. Split data into train/val/test sets.
3. Fit and serialize the preprocessing steps.
4. Train the regularized MLP model with `EarlyStopping` and `ReduceLROnPlateau` callbacks.
5. Save the trained model (`brain_tumor_model.h5`) and individual sklearn estimators to `models/`.

### Running Inference

An end-to-end CLI prediction script is provided in `predict.py` to classify raw CSV datasets.

```bash
# Run predictions on a raw dataset (e.g., test.csv)
python predict.py dataset/test.csv
```

This script automatically:
1. Loads the target CSV.
2. Applies the saved numeric imputer, categorical imputer, one-hot encoder, and standard scaler.
3. Feeds the processed features to the Keras model.
4. Outputs the predictions to stdout and exports a `_predictions.csv` file containing:
   - `predicted_label` (integer index)
   - `predicted_class` (name of the tumor class)
   - `confidence` (probability score)

---

## Key ML Principles

| Principle | Implementation |
|---|---|
| **No Data Leakage** | Splits are performed before any transformation. Estimators are fit exclusively on training data. |
| **Overfitting Defense** | Addressed using Dropout, Batch Normalization, L2 regularization, and Early Stopping. |
| **Reproducibility** | Set fixed random states (`random_state=42`) and serialized all fitting artifacts. |
| **Robust Imputation** | Imputes missing values using median (numerics) and mode (categoricals) derived from training data only. |
| **Consistent Inference** | The `predict.py` pipeline loads exact training estimators via `joblib`, matching the training data distribution exactly. |

---

## Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| TensorFlow / Keras | 2.20.0 |
| scikit-learn | 1.6.1 |
| pandas | Latest |
| numpy | Latest |
| matplotlib | Latest |
| seaborn | Latest |
| joblib | Latest |
