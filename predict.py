"""
predict.py — Brain Tumor Classification Inference Script
=========================================================
Loads saved preprocessing artifacts and the trained Keras model,
applies the full preprocessing pipeline to a raw CSV file, and
outputs predicted tumor types for each row.

Usage:
    python predict.py <path_to_csv>

Example:
    python predict.py dataset/brain_tumor_dataset_test.csv

Output labels:
    0 → Glioma
    1 → Meningioma
    2 → Pituitary
"""

import sys
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

MODEL_PATH          = MODELS_DIR / "brain_tumor_model.h5"
NUMERIC_IMP_PATH    = MODELS_DIR / "numeric_imputer.pkl"
CATEGORICAL_IMP_PATH= MODELS_DIR / "categorical_imputer.pkl"
ONEHOT_PATH         = MODELS_DIR / "onehot_encoder.pkl"
SCALER_PATH         = MODELS_DIR / "standard_scaler.pkl"
LABEL_ENC_PATH      = MODELS_DIR / "label_encoder.pkl"

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}

# ── Column definitions (must match training) ───────────────────────────────────
# Cols used by the numeric imputer (13 cols, includes edema_grade)
IMPUTER_NUMERIC_COLS = [
    "age", "bmi", "tumor_size_mm", "tumor_growth_rate",
    "headache_severity", "mri_intensity", "ct_density",
    "edema_grade", "ki67_index", "bp_systolic", "bp_diastolic",
    "wbc_count", "crp_level",
]

# Cols used by StandardScaler (12 cols, edema_grade excluded — kept as ordinal 0-3)
SCALER_NUMERIC_COLS = [
    "age", "bmi", "tumor_size_mm", "tumor_growth_rate",
    "headache_severity", "mri_intensity", "ct_density",
    "ki67_index", "bp_systolic", "bp_diastolic",
    "wbc_count", "crp_level",
]

CATEGORICAL_COLS = [
    "gender", "ethnicity", "region", "smoking_status",
    "alcohol_consumption", "family_history", "tumor_location",
    "nausea", "vision_problems", "seizures", "memory_loss",
    "balance_issues", "contrast_enhancement", "genetic_marker_status",
]

# Binary / ordinal mappings (hardcoded from training; these are semantic
# definitions, not statistics learned from data, so they are safe to embed here)
BINARY_MAPS = {
    "gender":             {"Male": 0, "Female": 1},
    "nausea":             {"Yes": 1, "No": 0},
    "vision_problems":    {"Yes": 1, "No": 0},
    "seizures":           {"Yes": 1, "No": 0},
    "memory_loss":        {"Yes": 1, "No": 0},
    "balance_issues":     {"Yes": 1, "No": 0},
    "family_history":     {"Yes": 1, "No": 0},
    "alcohol_consumption":{"Moderate": 0, "Heavy": 1},
    "contrast_enhancement":{"Mild": 0, "Moderate": 1, "Strong": 2},
}

ONE_HOT_COLS = [
    "genetic_marker_status", "tumor_location",
    "smoking_status", "region", "ethnicity",
]


def load_artifacts():
    """Load all saved preprocessing objects and the Keras model."""
    missing = [
        p for p in [MODEL_PATH, NUMERIC_IMP_PATH, CATEGORICAL_IMP_PATH,
                    ONEHOT_PATH, SCALER_PATH, LABEL_ENC_PATH]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing artifact(s):\n  " + "\n  ".join(str(p) for p in missing) +
            "\nRun the notebook first to generate these files."
        )

    numeric_imputer    = joblib.load(NUMERIC_IMP_PATH)
    categorical_imputer= joblib.load(CATEGORICAL_IMP_PATH)
    onehot_encoder     = joblib.load(ONEHOT_PATH)
    scaler             = joblib.load(SCALER_PATH)
    label_encoder      = joblib.load(LABEL_ENC_PATH)
    model              = tf.keras.models.load_model(MODEL_PATH)

    return numeric_imputer, categorical_imputer, onehot_encoder, scaler, label_encoder, model


def preprocess(df: pd.DataFrame,
               numeric_imputer,
               categorical_imputer,
               onehot_encoder,
               scaler) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a raw dataframe.

    Steps (must mirror the notebook exactly):
      1. Impute missing values
      2. Binary / ordinal mapping
      3. One-Hot Encoding
      4. Standard scaling of numeric features

    Returns a NumPy array ready for model.predict().
    """
    df = df.copy()
    DROP_COLS = [
    "patient_id",
    "Patient_ID",
    "PatientID",
    "id",
    "ID"
    ]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Drop target column if present in the input file
    if "tumor_type" in df.columns:
        df = df.drop(columns=["tumor_type"])

    # ── Step 1: Impute ──────────────────────────────────────────────────────
    df[IMPUTER_NUMERIC_COLS] = numeric_imputer.transform(df[IMPUTER_NUMERIC_COLS])
    df[CATEGORICAL_COLS]     = categorical_imputer.transform(df[CATEGORICAL_COLS])

    # ── Step 2: Binary / ordinal mapping ───────────────────────────────────
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    # ── Step 3: One-Hot Encoding ────────────────────────────────────────────
    ohe_array  = onehot_encoder.transform(df[ONE_HOT_COLS])
    ohe_cols   = onehot_encoder.get_feature_names_out(ONE_HOT_COLS)
    ohe_df     = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)

    df = df.drop(columns=ONE_HOT_COLS)
    df = pd.concat([df, ohe_df], axis=1)

    # ── Step 4: Standard scaling (12 cols only, edema_grade kept as ordinal) ──
    df[SCALER_NUMERIC_COLS] = scaler.transform(df[SCALER_NUMERIC_COLS])

    return df.to_numpy(dtype=np.float32)


def predict(csv_path: str) -> pd.DataFrame:
    """
    End-to-end prediction for a raw CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file (same schema as the training dataset,
        minus the target column which is optional).

    Returns
    -------
    pd.DataFrame with columns:
        predicted_label  (int)  : 0, 1, or 2
        predicted_class  (str)  : "Glioma", "Meningioma", or "Pituitary"
        confidence       (float): softmax probability of the predicted class
    """
    print(f"[INFO] Loading data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"[INFO] Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")

    print("[INFO] Loading artifacts …")
    numeric_imp, cat_imp, onehot, scaler, label_enc, model = load_artifacts()

    print("[INFO] Preprocessing …")
    X = preprocess(df_raw, numeric_imp, cat_imp, onehot, scaler)

    print("[INFO] Running inference …")
    probs   = model.predict(X, verbose=0)          # shape (n, 3)
    labels  = np.argmax(probs, axis=1)             # int labels
    confs   = probs[np.arange(len(labels)), labels] # confidence scores

    results = pd.DataFrame({
        "predicted_label": labels,
        "predicted_class": [CLASS_NAMES[l] for l in labels],
        "confidence":      confs.round(4),
    })
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    results = predict(csv_path)

    print("\n── Predictions ──────────────────────────────")
    print(results.to_string(index=False))
    print(f"\nClass distribution:")
    print(results["predicted_class"].value_counts().to_string())

    out_path = os.path.splitext(csv_path)[0] + "_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
