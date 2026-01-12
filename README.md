# Aegis – Real-time Credit Card Fraud Detection System

**Aegis** is an end-to-end machine learning project designed to identify fraudulent credit card transactions using anomaly detection. The system transitions from a baseline evaluation model to a production-ready API that serves real-time predictions.

## 🚀 Key Features
* **Anomaly Detection:** Utilizes the **Isolation Forest** algorithm to detect fraudulent outliers without needing historical labels.
* **Production Pipeline:** Includes automated data preprocessing using `StandardScaler` to handle feature normalization.
* **Real-time API:** Built with **FastAPI** to provide instant fraud scoring via a RESTful endpoint (`/predict`).
* **Web Interface:** A simple UI powered by Jinja2 templates for manual transaction verification.
* **Scalable Architecture:** Designed to handle high-dimensional PCA-transformed features (V1-V28).

## 📂 Project Structure
* `data_prep.py`: Script to generate synthetic transaction data (`creditcard.csv`).
* `train_baseline.py`: Initial script to evaluate model performance (Precision/Recall).
* `train_aegis.py`: Advanced training script that saves the finalized model and scaler.
* `main.py`: The FastAPI application serving the model.
* `aegis_model.joblib`: The serialized Isolation Forest model.
* `scaler.joblib`: The trained StandardScaler object for consistent inference.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **API Framework:** FastAPI, Uvicorn
* **Model Persistence:** Joblib

