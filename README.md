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
│   ├── preprocessing_pipeline.pkl # Serialized sklearn preprocessing pipeline
│   └── label_encoder.pkl          # Serialized LabelEncoder for target decoding
├── notebooks/
│   └── BrainTumorClassification.ipynb  # Full training notebook
├── docs/
│   └── MODEL_DOCUMENTATION.md    # Detailed technical documentation
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
2. **Imputation** — Median for numerics, mode for categoricals (`SimpleImputer`)
3. **Encoding:**
   - Binary mapping for symptom flags and boolean columns
   - Ordinal mapping for severity/ranked features
   - One-Hot Encoding for nominal categoricals (`ethnicity`, `tumor_location`, `smoking_status`, `region`, `genetic_marker_status`)
4. **Standard Scaling** — Z-score normalization of numeric features (`StandardScaler`)

After One-Hot Encoding, features expand from **27 → 43**.

### Serialized Pipeline

The fitted pipeline is saved to `models/preprocessing_pipeline.pkl` using `joblib`. This ensures identical transformations during inference.

---

## Model Architecture

```
Input (43 features)
     │
  Dense(64, ReLU)     ← 2,816 params
     │
  Dense(32, ReLU)     ← 2,080 params
     │
  Dense(3, Softmax)   ← 99 params
     │
Output (Glioma | Meningioma | Pituitary)

Total Trainable Params: 4,995
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |

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

> For the full training history, confusion matrix, and detailed analysis, see [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md).

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
1. Load and clean the dataset
2. Split data into train/val/test sets
3. Build and serialize the preprocessing pipeline
4. Train the MLP model for 50 epochs
5. Evaluate on the test set
6. Save `brain_tumor_model.h5` and `preprocessing_pipeline.pkl` to `models/`

### Running Inference

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load saved artifacts
pipeline = joblib.load('models/preprocessing_pipeline.pkl')
model    = load_model('models/brain_tumor_model.h5')
le       = joblib.load('models/label_encoder.pkl')

# Prepare raw input (DataFrame with the original 27 feature columns)
X_new_processed = pipeline.transform(X_new_raw)

# Predict
probs          = model.predict(X_new_processed)
predicted_idx  = np.argmax(probs, axis=1)
predicted_labels = le.inverse_transform(predicted_idx)
# → ['Glioma', 'Pituitary', ...]
```

> ⚠️ Always use the saved `preprocessing_pipeline.pkl` — never re-fit or manually scale data for inference.

---

## Key ML Principles

This project demonstrates several best practices for real-world ML pipelines:

| Principle | Implementation |
|---|---|
| **No Data Leakage** | Split performed before any transformation; encoders fit only on training data |
| **Reproducibility** | Fixed `random_state=42`; serialized pipeline and model artifacts |
| **Robust Imputation** | Median/mode imputation handles missing values without bias |
| **Ordinal Awareness** | Severity features encoded with correct order instead of arbitrary integers |
| **Pipeline Serialization** | `joblib` ensures inference-time transformations match training exactly |

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
