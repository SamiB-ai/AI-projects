import joblib
import pandas as pd
import numpy as np
import json

# load models
model = joblib.load("../models/churn_model.pkl")
kmeans = joblib.load("../models/kmeans_model.pkl")
scaler = joblib.load("../models/scaler_cluster.pkl")
features = joblib.load("../models/features.pkl")

with open("../models/segment_names.json", "r") as f:
    segment_names = json.load(f)

# Features attendues par le scaler_cluster (dans cet ordre exact)
CLUSTER_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "HasTechSupport",
    "HasOnlineSec",
    "IsMonthly",
    "HasMultiLines",
]

def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(0)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


def build_cluster_features(df_raw):
    """Recrée les 8 features custom utilisées lors du fit du scaler_cluster."""
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(0)

    cluster_df = pd.DataFrame()
    cluster_df["tenure"]         = df["tenure"]
    cluster_df["MonthlyCharges"] = df["MonthlyCharges"]
    cluster_df["TotalCharges"]   = df["TotalCharges"]
    cluster_df["SeniorCitizen"]  = df["SeniorCitizen"]
    cluster_df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    cluster_df["HasOnlineSec"]   = (df["OnlineSecurity"] == "Yes").astype(int)
    cluster_df["IsMonthly"]      = (df["Contract"] == "Month-to-month").astype(int)
    cluster_df["HasMultiLines"]  = (df["MultipleLines"] == "Yes").astype(int)

    return cluster_df


def assign_segment(df_raw):
    """Segmentation à partir des données brutes (avant get_dummies)."""
    cluster_df = build_cluster_features(df_raw)
    df_scaled = scaler.transform(cluster_df[CLUSTER_FEATURES])
    clusters = kmeans.predict(df_scaled)
    return clusters


def add_churn_score(df):
    proba = model.predict_proba(df)[:, 1]
    return proba


def risk_level(p):
    if p > 0.7:
        return "High"
    elif p > 0.4:
        return "Medium"
    else:
        return "Low"


def retention_action(segment, churn_proba):
    if churn_proba > 0.7:
        actions = {
            "At-Risk Newcomers":  "Offer onboarding support + discount",
            "High-Value Churners": "Premium retention + dedicated support",
            "Loyal Long-term":    "Loyalty reward + prevent downgrade",
            "Stable Mid-tier":    "Targeted retention campaign",
        }
        return actions.get(segment, "Manual review")

    elif churn_proba > 0.4:
        actions = {
            "At-Risk Newcomers":  "Send welcome offer",
            "High-Value Churners": "Send premium loyalty offer",
            "Loyal Long-term":    "Send renewal reminder",
            "Stable Mid-tier":    "Send personalized offer",
        }
        return actions.get(segment, "Send personalized offer")

    else:
        return "No action"

def run_business_pipeline(df_raw):
    clusters = assign_segment(df_raw)

    df = preprocess(df_raw)

    churn_proba = add_churn_score(df)

    result = df_raw.copy()
    result["Segment"]     = clusters
    result["SegmentName"] = result["Segment"].map(
        {int(k): v for k, v in segment_names.items()}
    )
    result["churn_proba"]    = churn_proba
    result["risk"]           = result["churn_proba"].apply(risk_level)
    result["action"]         = result.apply(
        lambda row: retention_action(row["SegmentName"], row["churn_proba"]), axis=1
    )
    result["priority_score"] = result["churn_proba"] * result["MonthlyCharges"]

    return result


if __name__ == "__main__":
    df = pd.read_csv("../data/raw/telco.csv")
    result = run_business_pipeline(df)
    print(result[[
        "SegmentName", "churn_proba", "risk", "action", "priority_score"
    ]].head())