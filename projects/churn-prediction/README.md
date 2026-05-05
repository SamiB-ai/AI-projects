# Churn Intelligence Platform

> End-to-end machine learning system for customer churn prediction, segmentation, and retention strategy generation.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)

---

## Overview

This project goes beyond a standard churn model. It is a full ML platform that combines **predictive modeling**, **customer segmentation**, **explainability**, and **actionable business recommendations** — all served through a REST API and an interactive UI.

**Key capabilities:**
- Predict churn probability for any customer profile
- Assign customers to behavioral segments via KMeans clustering
- Explain every prediction with SHAP feature impact
- Generate retention actions based on segment × risk level
- Expose predictions via a FastAPI REST endpoint
- Visualize everything through a Streamlit dashboard
- Deploy the full stack with a single Docker command

---

## Architecture

```
churn-platform/
├── data/
│   ├── raw/                  # Original Telco dataset
│   └── processed/            # Enriched dataset with segments
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_eda.ipynb          # Model training + SHAP
│   └── 03_clustering.ipynb   # KMeans segmentation
│
├── src/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic input/output models
│   └── business.py           # ML pipeline + business logic
│
├── app/
│   └── app.py                # Streamlit dashboard
│
├── models/                   # Saved artifacts (.pkl, .json)
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Target distribution analysis (26.5% churn rate)
- Feature correlation with churn: tenure, contract type, monthly charges
- Visualization of key behavioral patterns

### 2. Preprocessing
- `TotalCharges` coerced to numeric, missing values dropped
- Stratified train/test split (80/20)
- One-hot encoding with `drop_first=True`
- Class imbalance handled via `scale_pos_weight`

### 3. Models Trained

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.835 | Baseline |
| Random Forest | 0.820 | Ensemble |
| **XGBoost** | **0.832** | **Production model** |

### 4. Explainability — SHAP
Every prediction is explained using SHAP TreeExplainer. The top 10 features driving each individual prediction are displayed as a waterfall chart in the UI.

**Top features by importance:**
- `InternetService_Fiber optic`
- `Contract_Two year` / `Contract_One year`
- `tenure`
- `MonthlyCharges`
- `PaymentMethod_Electronic check`

### 5. Customer Segmentation — KMeans
Customers are clustered using 8 behavioral features. Optimal K is selected via Elbow method + Silhouette score.

| Segment | Profile | Churn Risk |
|---|---|---|
| At-Risk Newcomers | New, month-to-month contract | 🔴 High |
| High-Value Churners | High spend, still leaving | 🟠 Medium-High |
| Stable Mid-tier | Average profile | 🟡 Medium |
| Loyal Long-term | Long tenure, annual contract | 🟢 Low |

### 6. Business Layer
Retention actions are generated based on `segment × churn_proba`:

```python
# Example output
{
  "SegmentName": "At-Risk Newcomers",
  "churn_proba": 0.87,
  "risk": "High",
  "action": "Offer onboarding support + discount",
  "priority_score": 104.4   # churn_proba × MonthlyCharges
}
```

`priority_score` allows customer success teams to rank intervention efforts by revenue impact.

---

## API — FastAPI

### Run locally
```bash
uvicorn src.main:app --reload
```

### Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch predictions |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 1,
    "MonthlyCharges": 99.75,
    "TotalCharges": 99.75,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

### Example response
```json
{
  "customerID": null,
  "SegmentName": "At-Risk Newcomers",
  "churn_proba": 0.9726,
  "risk": "High",
  "action": "Offer onboarding support + discount",
  "priority_score": 97.23
}
```

Interactive docs available at `http://localhost:8000/docs`

---

## Streamlit Dashboard

The UI provides a full customer profile form with real-time prediction output:

- **Churn probability** with visual progress bar
- **Risk badge** (color-coded: High / Medium / Low)
- **Customer segment** assignment
- **Priority score** for triage
- **Recommended retention action**
- **SHAP waterfall chart** — top 10 features driving the prediction
- **Customer summary** panel

---

## Docker — Full Stack Deployment

### Requirements
- Docker + Docker Compose

### Run
```bash
docker compose up --build
```

### Services

| Service | URL |
|---|---|
| Streamlit UI | `http://localhost:8501` |
| FastAPI | `http://localhost:8000` |
| API Docs | `http://localhost:8000/docs` |

The API container includes a healthcheck. The Streamlit container starts automatically once the API is ready.

---

## Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| ML models | scikit-learn, XGBoost |
| Explainability | SHAP |
| Clustering | scikit-learn KMeans |
| API | FastAPI, Uvicorn, Pydantic |
| UI | Streamlit, Matplotlib, Seaborn |
| Deployment | Docker, Docker Compose |
| Language | Python 3.11 |

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset. 7,043 customers, 21 features, 26.5% churn rate.

---

## Results

- **ROC-AUC: 0.832** on held-out test set
- Model correctly identifies customers with up to **97.3% churn probability**
- Clear separation between churners (mean proba: 0.64) and non-churners (mean proba: 0.19)
- 4 actionable customer segments with distinct retention strategies

---

## Author

Built as a portfolio project to demonstrate end-to-end ML engineering skills:
data pipeline · feature engineering · model training · explainability · REST API · UI · Docker containerized deployment.
