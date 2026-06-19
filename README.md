# 🧪 ML Studio — Interactive End-to-End Machine Learning Pipeline

An interactive AutoML web application built from scratch using Python and Streamlit. Upload any dataset (CSV/Excel) and execute a complete ML pipeline through a clean, multi-page GUI — covering everything from raw data ingestion to model evaluation.

---

## 🎬 Demo

> 📹 *(Add demo video here)*

---

## 🖼️ Screenshots

### 📂 File Upload & Data Overview
![Upload](screenshots/upload.png)

### 📊 Data Visualization
![Visualization](screenshots/visualization.png)

### ⚙️ Preprocessing Pipeline
![Preprocessing](screenshots/preprocessing.png)

### 🤖 Model Training
![Model](screenshots/model_selection.png)

### 📈 Model Evaluation
![Evaluation](screenshots/evaluation.png)

---

## 🧠 Problem Statement

Running a proper ML pipeline requires deep technical knowledge and repetitive boilerplate code. ML Studio solves this by providing a structured, interactive interface that guides users through every stage — from raw data to a trained and evaluated model.

---

## 🏗️ Architecture

The project is split into two layers:

- **Frontend:** Multi-page Streamlit app using Session State to persist data across pages
- **Backend (core/):** Python modules handling all ML logic — cleaning, visualization, training, evaluation

---

## 📂 Pipeline Pages

### 1. 📂 Upload & Explore
- Supports CSV, Excel (.xlsx, .xls)
- Quick metrics: Samples, Features, Missing Values
- 3-tab analysis: Data Types · Statistical Summary · Missing Data Report

### 2. 📊 Visualization
- Scatter Plot — correlation between numeric features with Hue support
- Line Plot — time series and sequential data
- Box Plot — outlier detection and distribution comparison across categories

### 3. ⚙️ Preprocessing (8 Modules)

| Module | Options |
|--------|---------|
| Missing Values | Simple Imputer · KNN Imputer · Iterative Imputer (MICE) |
| Text Cleaning | Lowercase · Strip whitespace · Remove special characters |
| Encoding | Label Encoding · One-Hot Encoding (auto threshold-based) |
| Outlier Handling | IQR · Z-Score · Winsorization · Clipping |
| Feature Transformation | Log · Box-Cox · Yeo-Johnson · Polynomial Features |
| Feature Selection | Correlation filter · RFE (Logistic Regression) · PCA |
| Scaling | Standard Scaler · MinMax Scaler |
| Imbalanced Data | SMOTE (Oversampling) · Random Undersampling |

### 4. 🤖 Model Training

**Classification:**
Decision Tree · Random Forest · KNN · SVM · Logistic Regression · Naive Bayes · Neural Network (MLP)
> Interactive hyperparameter tuning: Max Depth, C, N-Neighbors, Hidden Layers

**Regression:**
Linear Regression · Ridge · Lasso · Decision Tree · Random Forest
> Metrics: R² · MAE · Actual vs Predicted line chart (first 50 rows)

**Clustering:**
K-Means (adjustable K via slider)
> Metrics: Silhouette Score · Inertia · Davies-Bouldin Score · Cluster scatter plot

### 5. 📈 Evaluation
- Classification: Accuracy · Precision · F1-Score · Confusion Matrix · Classification Report
- Regression: R² · MAE · Predicted vs Actual visualization
- Clustering: Silhouette Score · labeled dataset output

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-GUI-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-9cf)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-blue)
![Imbalanced-learn](https://img.shields.io/badge/SMOTE-Imbalanced--learn-yellow)

---

## 📁 Project Structure

```
ml-studio-pipeline/
│
├── app.py                        # Main entry point
├── pages/
│   ├── 00_upload.py              # File upload & data exploration
│   ├── 01_preprocessing.py       # Full preprocessing pipeline
│   ├── 02_visualization.py       # Data visualization
│   └── 03_models.py              # Model training & evaluation
├── core/
│   ├── p_ml.py                   # Preprocessing logic
│   ├── visualization.py          # Plotting functions
│   ├── classification_clustering.py
│   ├── regression.py
│   └── evaluation.py
├── assets/                       # Static files
├── screenshots/                  # App screenshots
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/v7med7elmy-ai/ml-studio-pipeline.git
cd ml-studio-pipeline
pip install -r requirements.txt
streamlit run app.py
```

---

## 👥 Team & My Role

Built by a team of 8 AI Engineering students.

**My Role: Team Leader & Lead Engineer**
- Designed the full app architecture and multi-page folder structure
- Built the entire Streamlit GUI and Session State management
- Implemented: Outlier Handling, Feature Transformation, PCA, RFE, SMOTE modules
- Contributed to Model Training and Evaluation modules
- Managed task distribution, integrated all team code, and resolved critical bugs

---

## 👤 Author

**Ahmed Helmy** — AI Engineering Student | Team Leader
[GitHub](https://github.com/v7med7elmy-ai)
