# Brain Tumor Classification — Model Documentation

**Project:** Brain Tumor Multi-Class Classification  
**Model Type:** Feed-Forward Neural Network (Multilayer Perceptron)  
**Framework:** TensorFlow / Keras 2.20.0  
**Dataset:** Synthetic (`brain_tumor_data.csv`) — 9,000 records  
**Target Classes:** Glioma (0), Meningioma (1), Pituitary (2)

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Data Splits](#2-data-splits)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Feature Engineering Summary](#4-feature-engineering-summary)
5. [Label Encoding](#5-label-encoding)
6. [Model Architecture](#6-model-architecture)
7. [Training Configuration](#7-training-configuration)
8. [Training History](#8-training-history)
9. [Evaluation Results](#9-evaluation-results)
10. [Serialized Artifacts](#10-serialized-artifacts)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Dataset Overview

| Property             | Value                          |
| -------------------- | ------------------------------ |
| File                 | `dataset/brain_tumor_data.csv` |
| Total Records        | 9,000                          |
| Total Features (raw) | 27                             |
| Target Column        | `tumor_type`                   |
| Classes              | Glioma, Meningioma, Pituitary  |

The dataset is fully synthetic and was designed to mirror realistic clinical and demographic features associated with brain tumor diagnosis.

---

## 2. Data Splits

The data was split **before** any transformation is applied to prevent data leakage.

| Split      | Proportion | Records (approx.) |
| ---------- | ---------- | ----------------- |
| Train      | 70%        | 6,300             |
| Validation | 15%        | 1,350             |
| Test       | 15%        | 1,350             |

> **Anti-Leakage Guarantee:** All encoders and scalers are fit **exclusively on training data** (`X_train`) and then applied to validation and test sets via `transform()` only.

---

## 3. Preprocessing Pipeline

The preprocessing pipeline was built using scikit-learn components and serialized using `joblib` for reproducible inference.

### Step 1: Manual Data Cleaning (Pre-split)

Inconsistent string markers were normalized before splitting:

```python
# Unify "unknown" representations to NaN
df['column'] = df['column'].replace('Unknown', np.nan)
df['column'] = df['column'].replace('?',       np.nan)

# Standardize gender casing
df['gender'] = df['gender'].str.strip().str.capitalize()
```

### Step 2: Train/Val/Test Split

```python
X_train, X_temp, Y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
```

### Step 3: Missing Value Imputation (fit on train only)

| Feature Group        | Strategy             | Transformer                               |
| -------------------- | -------------------- | ----------------------------------------- |
| Numeric features     | Median               | `SimpleImputer(strategy='median')`        |
| Categorical features | Most frequent (mode) | `SimpleImputer(strategy='most_frequent')` |

### Step 4: Categorical Encoding

| Encoding Type    | Applied To                                                                                                      | Method                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Binary mapping   | Boolean / symptom flags (e.g., `nausea`, `seizures`, `family_history`)                                          | Manual dict map: `{True/Yes: 1, False/No: 0}`      |
| Ordinal mapping  | Severity / ranked categoricals (e.g., `headache_severity`, `edema_grade`)                                       | Manual ordered integer map                         |
| One-Hot Encoding | Nominal categoricals (e.g., `ethnicity`, `tumor_location`, `smoking_status`, `genetic_marker_status`, `region`) | `OneHotEncoder(drop='first', sparse_output=False)` |
| Binary mapping   | `gender`                                                                                                        | `{Male: 1, Female: 0}`                             |

### Step 5: Standard Scaling (fit on train only)

All numeric features (after imputation) are scaled using:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
```

---

## 4. Feature Engineering Summary

After all preprocessing steps, the feature space expands due to One-Hot Encoding:

| Feature Count | Stage                                                         |
| ------------- | ------------------------------------------------------------- |
| 27            | Raw input features                                            |
| **43**        | After OHE expansion (confirmed: `X_train.shape = (6300, 43)`) |

### One-Hot Encoded Columns (examples)

```
genetic_marker_status_Inconclusive
genetic_marker_status_Negative
genetic_marker_status_Positive
tumor_location_Cerebellum / Convexity / Frontal / Occipital / Parietal / Sellar / Temporal
smoking_status_Current / Former / Never
region_Rural / Suburban / Urban
ethnicity_African / Asian / Caucasian / Hispanic / Other
```

---

## 5. Label Encoding

The target column `tumor_type` was encoded using scikit-learn's `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
y_val   = label_encoder.transform(y_val)
y_test  = label_encoder.transform(y_test)
```

### Class Mapping

| Label (String) | Encoded Integer |
| -------------- | --------------- |
| Glioma         | 0               |
| Meningioma     | 1               |
| Pituitary      | 2               |

---

## 6. Model Architecture

A Sequential Multilayer Perceptron (MLP) was constructed using Keras with regularization techniques added to combat overfitting identified in the initial training run.

```python
model = Sequential([
    tf.keras.Input(shape=(input_dim,)),        # Explicit Input (no deprecation)

    Dense(64, activation="relu",
          kernel_regularizer=l2(0.001)),       # L2 weight regularization
    BatchNormalization(),                       # Normalize activations
    Dropout(0.3),                              # Randomly drop 30% connections

    Dense(32, activation="relu",
          kernel_regularizer=l2(0.001)),       # L2 weight regularization
    BatchNormalization(),
    Dropout(0.2),                              # Randomly drop 20% connections

    Dense(3, activation="softmax")             # Output: 3 classes
])
```

### Layer Summary

| Layer | Type | Units | Activation | Regularization |
|---|---|---|---|---|
| Input | Input | 43 | — | — |
| Hidden 1 | Dense | 64 | ReLU | L2 (0.001) |
| — | BatchNormalization | — | — | — |
| — | Dropout | — | — | rate = 0.30 |
| Hidden 2 | Dense | 32 | ReLU | L2 (0.001) |
| — | BatchNormalization | — | — | — |
| — | Dropout | — | — | rate = 0.20 |
| Output | Dense | 3 | Softmax | — |

> **Regularization rationale:** The original model (no dropout/L2) showed clear overfitting — training loss → 0.00 while validation loss climbed to 0.45 by epoch 50. Dropout, BatchNormalization, and L2 weight penalty were added to improve generalization.

---

## 7. Training Configuration

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, y_val),
    epochs=100,           # ceiling; EarlyStopping halts before this
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (default lr = 0.001) |
| Loss Function | Sparse Categorical Crossentropy |
| Max Epochs | 100 (EarlyStopping halts early) |
| Batch Size | 32 |
| EarlyStopping patience | 8 epochs |
| ReduceLROnPlateau factor | 0.5 (halves LR on plateau) |
| ReduceLROnPlateau patience | 4 epochs |
| Dropout rates | 0.30 (Layer 1), 0.20 (Layer 2) |
| L2 regularization λ | 0.001 (both Dense layers) |

---

## 8. Training History

### Epoch Milestones

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
| ----- | --------- | ------- | ---------- | -------- |
| 1     | 0.8440    | 0.9496  | 0.4026     | 0.1365   |
| 2     | 0.9584    | 0.9570  | 0.1248     | 0.1155   |
| 5     | 0.9625    | 0.9548  | 0.1010     | 0.1207   |
| 10    | 0.9702    | 0.9526  | 0.0819     | 0.1313   |
| 20    | 0.9892    | 0.9415  | 0.0379     | 0.1839   |
| 30    | 0.9986    | 0.9407  | 0.0115     | 0.2703   |
| 40    | 1.0000    | 0.9407  | 0.0020     | 0.3659   |
| 50    | 0.9962    | 0.9407  | 0.0120     | 0.4245   |

### Final Metrics (Epoch 50)

| Metric   | Training | Validation |
| -------- | -------- | ---------- |
| Accuracy | 99.62%   | 94.07%     |
| Loss     | 0.0120   | 0.4245     |

### Observations (Original Run — Overfitting Detected)

- **Fast Convergence:** The model achieved >95% validation accuracy within just 2 epochs.
- **Overfitting Signal:** From epoch ~5 onward, training accuracy climbed toward 100% while validation accuracy plateaued at ~94%. Training loss → 0.00 while validation loss rose to 0.45 — a classic overfitting pattern.
- **Fix Applied:** Dropout, BatchNormalization, L2 regularization, EarlyStopping, and ReduceLROnPlateau were added to correct this. The model now trains until `val_loss` stops improving, restoring the best weights automatically.

---

## 9. Evaluation Results

### Test Set Performance

```
Test Accuracy : 94.37%
Test Loss     : 0.4569
```

Evaluated on 1,350 held-out test samples (43 batches).

### Classification Report

| Class              | Precision | Recall | F1-Score | Support |
| ------------------ | --------- | ------ | -------- | ------- |
| **Glioma (0)**     | 0.99      | 0.95   | 0.97     | 570     |
| **Meningioma (1)** | 0.90      | 0.95   | 0.93     | 464     |
| **Pituitary (2)**  | 0.94      | 0.91   | 0.92     | 316     |
| **Accuracy**       |           |        | **0.94** | 1350    |
| **Macro Avg**      | 0.94      | 0.94   | 0.94     | 1350    |
| **Weighted Avg**   | 0.95      | 0.94   | 0.94     | 1350    |

### Key Observations

- **Glioma** achieves the highest F1 score (0.97), with near-perfect precision (0.99), indicating very few false positives.
- **Meningioma** has the lowest precision (0.90), meaning some non-meningioma samples are misclassified as meningioma.
- **Pituitary** performs well overall but has the lowest recall (0.91), suggesting a few pituitary tumors are misclassified as other types.
- All three classes clear the 90% threshold for all metrics, indicating robust multi-class performance.

---

## 10. Serialized Artifacts

| Artifact            | Path                             | Description                                        |
| ------------------- | -------------------------------- | -------------------------------------------------- |
| Keras Model         | `models/brain_tumor_model.h5`    | Saved weights and architecture of the trained MLP  |
| Numeric Imputer     | `models/numeric_imputer.pkl`     | Median-based imputer for the 13 numeric features   |
| Categorical Imputer | `models/categorical_imputer.pkl` | Mode-based imputer for the 14 categorical features |
| One-Hot Encoder     | `models/onehot_encoder.pkl`      | OneHotEncoder for categorical nominal variables    |
| Standard Scaler     | `models/standard_scaler.pkl`     | StandardScaler fit on training numeric features    |
| Label Encoder       | `models/label_encoder.pkl`       | Target label mapping (Glioma/Meningioma/Pituitary) |

### Loading for Inference

```python
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load all artifacts
model = load_model('models/brain_tumor_model.h5')
numeric_imputer = joblib.load('models/numeric_imputer.pkl')
categorical_imputer = joblib.load('models/categorical_imputer.pkl')
onehot_encoder = joblib.load('models/onehot_encoder.pkl')
scaler = joblib.load('models/standard_scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# For end-to-end preprocessing, run the script:
python predict.py <path_to_csv>
```

> ⚠️ **Warning:** The exact fitted imputer, encoder, and scaler from the training set must be reused for **all** inference tasks. Re-fitting any estimator on test/prediction data will lead to feature shifts and incorrect predictions.

---

## 11. Key Design Decisions

| Decision                          | Rationale                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| Split-before-transform            | Prevents data leakage; ensures validation/test metrics reflect true generalization      |
| Median imputation for numerics    | Robust to outliers compared to mean imputation                                          |
| Mode imputation for categoricals  | Most appropriate strategy for categorical missing values                                |
| `drop='first'` in OHE             | Avoids multicollinearity (dummy variable trap) in the dense layers                      |
| `sparse_categorical_crossentropy` | Works with integer class labels directly, avoiding the need for one-hot target encoding |
| Adam optimizer                    | Adaptive learning rate; well-suited for tabular MLP classification tasks                |
| Softmax output                    | Multi-class classification requires probability distribution over all classes           |
| `joblib` serialization            | Ensures the exact fitted transformers are reused at inference time                      |
