# ❤️ Heart Disease Prediction Web App

<p align="center">
  <img src="https://img.shields.io/badge/Maintained-Yes-green.svg" />
  <img src="https://img.shields.io/badge/Framework-Flask-lightgrey.svg" />
  <img src="https://img.shields.io/badge/AI--ML-Scikit--learn-orange.svg" />
</p>

### 💡 Project Overview
This project is an end-to-end **Machine Learning application** designed to predict the risk of cardiovascular disease. It bridges the gap between data science and user accessibility by wrapping a trained predictive model in a clean, interactive **Flask web interface**. Users can input medical parameters and receive a risk assessment in real-time.

---

## ✨ Key Features

* 🤖 **Algorithm Implementation:** Comparison of Logistic Regression, Random Forest, and Gradient Boosting to find the most accurate predictor.
* 🌐 **Interactive Web UI:** Custom-built frontend using Flask `templates` for seamless user data entry.
* ⚡ **Model Serialization:** Uses a pre-trained `heart_disease_bundle.pkl` for instantaneous inference without retraining.
* 📊 **Deep Metric Analysis:** Evaluation using Confusion Matrices, Classification Reports, and ROC-AUC scores to ensure medical reliability.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask |
| **AI/ML** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Pickle (Model Serialization) |

---

## 📁 Project Structure

```bash
heart-disease-prediction/
├── templates/          # HTML frontend files
├── app.py              # Flask server & prediction logic
├── heart_disease_bundle.pkl  # Serialized ML model & scaler
└── requirements.txt    # Project dependencies
