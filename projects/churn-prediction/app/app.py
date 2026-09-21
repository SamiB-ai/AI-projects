import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import requests
import os


# ══════════════════════════════════════════════════════════════════════════════
# API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://ai-projects-3-qkro.onrender.com"


def predict_churn(customer_data):
    """
    Send customer data to the FastAPI backend on Render.
    """
    response = requests.post(
        f"{API_URL}/predict",
        json=customer_data,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

.stApp {
    background-color: #0d0f14;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2230;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #13151e;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}

.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}

.metric-card .sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.25rem;
}

/* Risk badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.badge-high {
    background: #3b0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
}

.badge-medium {
    background: #431407;
    color: #fb923c;
    border: 1px solid #7c2d12;
}

.badge-low {
    background: #052e16;
    color: #4ade80;
    border: 1px solid #14532d;
}

/* Action box */
.action-box {
    background: #0f1623;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #93c5fd;
    font-family: 'IBM Plex Mono', monospace;
}

/* Section titles */
.section-title {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2230;
}

/* Header */
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}

.header-sub {
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 0.05em;
    font-family: 'IBM Plex Mono', monospace;
}

/* Probability bar */
.prob-bar-wrap {
    background: #1e2230;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 0.5rem;
}

.prob-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Divider */
hr {
    border-color: #1e2230;
}

/* Streamlit overrides */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: #13151e !important;
    border: 1px solid #1e2230 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}

.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #2563eb;
}

[data-testid="stMetric"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD LOCAL ARTIFACTS
# Used only for SHAP visualization.
# The actual prediction comes from FastAPI /predict.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():

    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models")
    )

    model = joblib.load(
        os.path.join(base, "churn_model.pkl")
    )

    features = joblib.load(
        os.path.join(base, "features.pkl")
    )

    return model, features


try:
    model, features = load_artifacts()
    shap_available = True
    shap_error = None

except Exception as e:
    shap_available = False
    shap_error = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FOR SHAP
# This mirrors the preprocessing used by the backend.
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_shap(df_raw):

    df = df_raw.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.fillna(0)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def shap_waterfall(model, df_proc, features_list):

    explainer = shap.TreeExplainer(model)

    shap_vals = explainer.shap_values(df_proc)

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    vals = sv
    names = features_list

    # Top 10 features
    idx = np.argsort(np.abs(vals))[::-1][:10]

    top_vals = vals[idx]
    top_names = [names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 4))

    fig.patch.set_facecolor("#13151e")
    ax.set_facecolor("#13151e")

    colors = [
        "#f87171" if v > 0 else "#4ade80"
        for v in top_vals
    ]

    ax.barh(
        range(len(top_vals)),
        top_vals,
        color=colors,
        height=0.6
    )

    ax.set_yticks(range(len(top_names)))

    ax.set_yticklabels(
        top_names,
        color="#94a3b8",
        fontsize=8,
        fontfamily="monospace"
    )

    ax.set_xlabel(
        "SHAP value (impact on churn probability)",
        color="#475569",
        fontsize=8
    )

    ax.tick_params(
        colors="#475569",
        labelsize=8
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#1e2230")
    ax.spines["bottom"].set_color("#1e2230")

    ax.axvline(
        0,
        color="#1e2230",
        linewidth=1
    )

    red_patch = mpatches.Patch(
        color="#f87171",
        label="↑ increases churn risk"
    )

    green_patch = mpatches.Patch(
        color="#4ade80",
        label="↓ decreases churn risk"
    )

    ax.legend(
        handles=[red_patch, green_patch],
        framealpha=0,
        labelcolor="#94a3b8",
        fontsize=7.5,
        loc="lower right"
    )

    plt.tight_layout()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CUSTOMER INPUT
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown(
        '<div class="header-title">📡 Churn Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="header-sub">v1.0 · Single prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # ACCOUNT
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div class="section-title">Account</div>',
        unsafe_allow_html=True
    )

    tenure = st.slider(
        "Tenure (months)",
        0,
        72,
        12
    )

    monthly_charges = st.slider(
        "Monthly Charges ($)",
        10.0,
        120.0,
        65.0,
        step=0.5
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        value=float(tenure * monthly_charges),
        min_value=0.0
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CONTRACT
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div class="section-title" style="margin-top:1rem">'
        'Contract & Billing'
        '</div>',
        unsafe_allow_html=True
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SERVICES
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div class="section-title" style="margin-top:1rem">'
        'Services'
        '</div>',
        unsafe_allow_html=True
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multi_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["Fiber optic", "DSL", "No"]
    )

    online_sec = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_prot = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # DEMOGRAPHICS
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div class="section-title" style="margin-top:1rem">'
        'Demographics'
        '</div>',
        unsafe_allow_html=True
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    st.markdown("")

    predict_btn = st.button(
        "⚡  Run Prediction"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="header-title">'
    'Churn Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header-sub">'
    'ML-powered churn prediction · customer segmentation · retention actions'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# API STATUS
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔌 API Backend"):

    st.write(
        f"Backend: `{API_URL}`"
    )

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        if health_response.ok:

            st.success(
                "FastAPI backend is online."
            )

        else:

            st.warning(
                f"API returned HTTP {health_response.status_code}"
            )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Cannot reach FastAPI backend: {e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

if predict_btn:

    # ──────────────────────────────────────────────────────────────────────────
    # Build customer payload
    # ──────────────────────────────────────────────────────────────────────────

    customer = {

        "customerID": "DEMO-001",

        "gender": gender,

        "SeniorCitizen": senior,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multi_lines,

        "InternetService": internet,

        "OnlineSecurity": online_sec,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_prot,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless,

        "PaymentMethod": payment,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges,
    }


    # ──────────────────────────────────────────────────────────────────────────
    # Call FastAPI
    # ──────────────────────────────────────────────────────────────────────────

    with st.spinner("Running prediction..."):

        try:

            prediction = predict_churn(customer)

        except requests.exceptions.HTTPError as e:

            st.error(
                f"FastAPI returned an error: {e}"
            )

            try:
                st.code(e.response.text)
            except Exception:
                pass

            st.stop()

        except requests.exceptions.RequestException as e:

            st.error(
                f"Could not connect to FastAPI: {e}"
            )

            st.info(
                "Check that the Render API is awake and available."
            )

            st.stop()

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

            st.stop()


    # ──────────────────────────────────────────────────────────────────────────
    # Read API response
    # ──────────────────────────────────────────────────────────────────────────

    try:

        proba = float(
            prediction["churn_proba"]
        )

        risk = prediction["risk"]

        segment = prediction["SegmentName"]

        action = prediction["action"]

        priority = float(
            prediction["priority_score"]
        )

    except (KeyError, TypeError, ValueError) as e:

        st.error(
            "Invalid response received from FastAPI."
        )

        st.json(prediction)

        st.stop()


    # ══════════════════════════════════════════════════════════════════════════
    # RISK DISPLAY
    # ══════════════════════════════════════════════════════════════════════════

    risk_color = {
        "High": "#f87171",
        "Medium": "#fb923c",
        "Low": "#4ade80"
    }.get(
        risk,
        "#94a3b8"
    )

    badge_cls = {
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low"
    }.get(
        risk,
        "badge-medium"
    )

    bar_color = risk_color


    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — KPIs
    # ══════════════════════════════════════════════════════════════════════════

    col1, col2, col3, col4 = st.columns(4)


    # ──────────────────────────────────────────────────────────────────────────
    # Churn Probability
    # ──────────────────────────────────────────────────────────────────────────

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="label">
                    Churn Probability
                </div>

                <div class="value"
                     style="color:{risk_color}">

                    {proba:.1%}

                </div>

                <div class="prob-bar-wrap">

                    <div class="prob-bar-fill"
                         style="width:{proba*100:.1f}%;
                                background:{bar_color}">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ──────────────────────────────────────────────────────────────────────────
    # Risk Level
    # ──────────────────────────────────────────────────────────────────────────

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="label">
                    Risk Level
                </div>

                <div class="value"
                     style="font-size:1.2rem;
                            padding-top:0.3rem">

                    <span class="badge {badge_cls}">
                        {risk}
                    </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ──────────────────────────────────────────────────────────────────────────
    # Segment
    # ──────────────────────────────────────────────────────────────────────────

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="label">
                    Customer Segment
                </div>

                <div class="value"
                     style="font-size:1rem;
                            color:#93c5fd;
                            padding-top:0.2rem">

                    {segment}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ──────────────────────────────────────────────────────────────────────────
    # Priority
    # ──────────────────────────────────────────────────────────────────────────

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="label">
                    Priority Score
                </div>

                <div class="value"
                     style="color:#e2e8f0">

                    {priority:.1f}

                </div>

                <div class="sub">
                    churn_proba × monthly_charges
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — ACTION + SHAP
    # ══════════════════════════════════════════════════════════════════════════

    col_left, col_right = st.columns([1, 1.6])


    # ──────────────────────────────────────────────────────────────────────────
    # LEFT COLUMN
    # ──────────────────────────────────────────────────────────────────────────

    with col_left:

        st.markdown(
            '<div class="section-title">'
            'Recommended Action'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="action-box">
                → {action}
            </div>
            """,
            unsafe_allow_html=True
        )


        # Customer summary

        st.markdown(
            '<div class="section-title" '
            'style="margin-top:1.5rem">'
            'Customer Summary'
            '</div>',
            unsafe_allow_html=True
        )


        summary_data = {

            "Tenure":
                f"{tenure} months",

            "Monthly Charges":
                f"${monthly_charges:.2f}",

            "Total Charges":
                f"${total_charges:.2f}",

            "Contract":
                contract,

            "Internet":
                internet,

            "Tech Support":
                tech_support,

            "Online Security":
                online_sec,
        }


        for key, value in summary_data.items():

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:0.3rem 0;
                    border-bottom:1px solid #1e2230;
                    font-size:0.82rem;
                ">

                    <span style="
                        color:#64748b;
                        font-family:monospace;
                    ">
                        {key}
                    </span>

                    <span style="
                        color:#cbd5e1;
                    ">
                        {value}
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ──────────────────────────────────────────────────────────────────────────
    # RIGHT COLUMN — SHAP
    # ──────────────────────────────────────────────────────────────────────────

    with col_right:

        st.markdown(
            '<div class="section-title">'
            'SHAP — Feature Impact'
            '</div>',
            unsafe_allow_html=True
        )


        if shap_available:

            try:

                df_raw = pd.DataFrame(
                    [customer]
                )

                df_proc = preprocess_for_shap(
                    df_raw
                )

                fig = shap_waterfall(
                    model,
                    df_proc,
                    features
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            except Exception as e:

                st.warning(
                    f"SHAP unavailable: {e}"
                )

        else:

            st.info(
                "SHAP visualization unavailable. "
                "The prediction itself is still performed "
                "by the FastAPI backend."
            )


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════

else:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:4rem 2rem;
            color:#334155;
        ">

            <div style="
                font-size:3rem;
                margin-bottom:1rem;
            ">
                📡
            </div>

            <div style="
                font-family:monospace;
                font-size:1rem;
                color:#475569;
            ">

                Fill in the customer profile on the left
                <br>

                and click

                <strong style="color:#3b82f6">
                    Run Prediction
                </strong>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        font-family:monospace;
        font-size:0.7rem;
        color:#1e2230;
    ">

        Churn Intelligence Platform ·
        XGBoost + KMeans + SHAP ·
        FastAPI backend available on /predict

    </div>
    """,
    unsafe_allow_html=True
)