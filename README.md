# 🧠 Brain Tumor Type Classification using Neural Networks

An end-to-end deep learning project for multi-class brain tumor classification using a Feed-Forward Neural Network (Multilayer Perceptron). This project demonstrates the complete machine learning workflow, from data preprocessing to model deployment, following industry best practices for reproducible AI development.

---

## 📌 Overview

Brain tumor diagnosis is a critical task in healthcare where accurate classification can support medical professionals in making informed decisions. This project aims to classify patient cases into one of three tumor types using a Multilayer Perceptron (MLP).

### Target Classes

- Glioma
- Meningioma
- Pituitary

The project includes the complete pipeline required for a production-ready machine learning solution, including preprocessing, model training, evaluation, hyperparameter tuning, and inference on unseen data.

---

## ✨ Features

- Comprehensive data exploration and cleaning
- Missing value detection and imputation
- Categorical feature encoding
- Feature scaling
- Class imbalance analysis
- Feed-Forward Neural Network (MLP)
- Hyperparameter tuning
- Regularization techniques
- Training and validation visualization
- Performance evaluation using multiple metrics
- Confusion matrix generation
- Baseline model comparison
- Saved preprocessing pipeline
- Saved trained model (`.h5`)
- Prediction script for new patient data

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```text
Brain-Tumor-Type-Classification/
│
├── dataset/
│   ├── brain_tumor_data.csv
│   └── data_dictionary.txt
│
├── notebooks/
│   └── BrainTumorClassification.ipynb
│
├── models/
│   ├── brain_tumor_model.h5
│   └── preprocessing_pipeline.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── feature_importance.png
│   └── classification_report.csv
│
├── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Machine Learning Pipeline

```text
Raw Dataset
      │
      ▼
Data Inspection
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Imputation
      │
      ▼
Categorical Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Neural Network Training
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Model Evaluation
      │
      ▼
Baseline Comparison
      │
      ▼
Save Model & Preprocessing
      │
      ▼
Prediction on New Data
```

---

## 📊 Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1
- Weighted F1
- Confusion Matrix
- Training & Validation Curves

---

## 🧠 Neural Network Architecture

The final model consists of:

- Input Layer
- Multiple Hidden Layers with ReLU activation
- Regularization (Dropout / Early Stopping)
- Softmax Output Layer (3 Classes)

The architecture was selected after comparing multiple hyperparameter configurations using the validation dataset.

---

## 📈 Hyperparameter Tuning

Several experiments were conducted by varying:

- Number of hidden layers
- Number of neurons
- Learning rate
- Batch size
- Dropout rate
- Optimizer
- Number of training epochs

The final model was selected solely based on validation performance.

---

## 📉 Baseline Comparison

To evaluate whether the neural network adds value, its performance is compared against a traditional machine learning model such as:

- Logistic Regression
- Random Forest

This comparison highlights the strengths and limitations of deep learning for structured medical datasets.

---

## 🚀 Running the Project

### Clone the repository

```bash
git clone https://github.com/your-username/Brain-Tumor-Type-Classification.git
cd Brain-Tumor-Type-Classification
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

## 🔮 Predict on New Data

```bash
python predict.py
```

The prediction script automatically:

- Loads the saved preprocessing pipeline
- Processes raw patient data
- Loads the trained neural network
- Predicts the tumor type
- Returns class probabilities

---

## 📁 Outputs

The project generates:

- Trained Model (`.h5`)
- Preprocessing Pipeline (`.pkl`)
- Classification Report
- Confusion Matrix
- Training History
- Performance Plots
