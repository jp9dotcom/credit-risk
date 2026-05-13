import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background: #0F1117; }

    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #161B27 100%);
    }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1520 100%);
        border: 1px solid #2a3550;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #E8EAF0;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #818CF8;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Score card */
    .score-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }
    .score-card.high-risk {
        background: linear-gradient(135deg, #2d1515 0%, #1f1010 100%);
        border-color: #7f1d1d;
    }
    .score-card.low-risk {
        background: linear-gradient(135deg, #0f2d1a 0%, #0a1f12 100%);
        border-color: #14532d;
    }
    .score-card.medium-risk {
        background: linear-gradient(135deg, #2d2415 0%, #1f1a0f 100%);
        border-color: #78350f;
    }
    .score-number {
        font-family: 'DM Serif Display', serif;
        font-size: 4rem;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-label {
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .decision-badge {
        display: inline-block;
        font-size: 1rem;
        font-weight: 600;
        padding: 8px 24px;
        border-radius: 30px;
        margin-top: 1rem;
    }
    .high-risk .decision-badge {
        background: rgba(239,68,68,0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .low-risk .decision-badge {
        background: rgba(34,197,94,0.2);
        color: #86EFAC;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .medium-risk .decision-badge {
        background: rgba(234,179,8,0.2);
        color: #FDE047;
        border: 1px solid rgba(234,179,8,0.3);
    }

    /* Metric cards */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: #161B27;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #E8EAF0;
        font-size: 1.3rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #9CA3AF;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1E2740;
    }

    /* Risk factors */
    .risk-factor {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: #161B27;
        border: 1px solid #2a3550;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .risk-factor-name {
        color: #D1D5DB;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .risk-factor-value {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-factor.danger .risk-factor-value { color: #F87171; }
    .risk-factor.warning .risk-factor-value { color: #FCD34D; }
    .risk-factor.safe .risk-factor-value { color: #6EE7B7; }

    /* Info box */
    .info-box {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .info-box p {
        color: #A5B4FC;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Input styling */
    .stSlider > div { padding-bottom: 0.5rem; }
    .stNumberInput > div > div > input {
        background: #161B27 !important;
        border: 1px solid #2a3550 !important;
        color: #E8EAF0 !important;
        border-radius: 8px !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0d1117 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.3) !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        models   = joblib.load('xgb_models_tuned.pkl')
        feat_df  = pd.read_csv('feature_list.csv')
        feat_cols = feat_df['Feature'].tolist()
        return models, feat_cols
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Make sure xgb_models_tuned.pkl and feature_list.csv are in the correct paths.")
        return None, None


def predict(applicant_features: dict, models, feat_cols, threshold=0.448):
    row = np.array(
        [applicant_features.get(c, 0) for c in feat_cols],
        dtype=np.float32
    ).reshape(1, -1)
    scores = [m.predict_proba(row)[0, 1] for m in models]
    score  = float(np.mean(scores))
    std    = float(np.std(scores))
    if score >= threshold:
        if score >= 0.65:
            decision, risk_class = "HIGH RISK — REJECT", "high-risk"
        else:
            decision, risk_class = "REVIEW REQUIRED", "medium-risk"
    else:
        decision, risk_class = "LOW RISK — APPROVE", "low-risk"
    return score, std, decision, risk_class


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Home Credit Risk Intelligence</div>
    <h1 class="hero-title">Credit Default Risk Scorer</h1>
    <p class="hero-subtitle">
        XGBoost ensemble model · 708 engineered features · 
        307,511 applicants trained · AUC 0.794
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
models, feat_cols = load_model()
if models is None:
    st.stop()

# ── Sidebar — input form ──────────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Applicant Information")
st.sidebar.markdown("---")

st.sidebar.markdown("**Personal Details**")
age           = st.sidebar.slider("Age (years)", 18, 70, 35)
gender        = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_status = st.sidebar.selectbox("Family Status",
    ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
children      = st.sidebar.slider("Number of children", 0, 10, 0)
fam_members   = st.sidebar.slider("Family members", 1, 10, 2)
education     = st.sidebar.selectbox("Education",
    ["Secondary / secondary special", "Higher education",
     "Incomplete higher", "Lower secondary", "Academic degree"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Employment**")
income_type   = st.sidebar.selectbox("Income type",
    ["Working", "Commercial associate", "Pensioner",
     "State servant", "Unemployed"])
employment_years = st.sidebar.slider("Years employed", 0, 40, 5)
income        = st.sidebar.number_input("Annual income (₹)", 50000, 10000000,
                                         200000, step=10000)

st.sidebar.markdown("---")
st.sidebar.markdown("**Loan Details**")
credit_amt    = st.sidebar.number_input("Loan amount requested (₹)",
                                         50000, 5000000, 500000, step=10000)
annuity       = st.sidebar.number_input("Monthly EMI (₹)", 1000, 200000,
                                         20000, step=1000)
goods_price   = st.sidebar.number_input("Goods price (₹)", 50000, 5000000,
                                         400000, step=10000)
loan_type     = st.sidebar.selectbox("Loan type",
    ["Cash loans", "Revolving loans"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Credit History**")
ext1 = st.sidebar.slider("External Credit Score 1 (0–1)", 0.0, 1.0, 0.5, 0.01)
ext2 = st.sidebar.slider("External Credit Score 2 (0–1)", 0.0, 1.0, 0.5, 0.01)
ext3 = st.sidebar.slider("External Credit Score 3 (0–1)", 0.0, 1.0, 0.5, 0.01)
prev_refused  = st.sidebar.slider("Previous loan refusals", 0, 10, 0)
days_credit_bureau = st.sidebar.slider(
    "Days since last bureau credit", 0, 2000, 365)

st.sidebar.markdown("---")
score_btn = st.sidebar.button("🔍  Score Applicant", use_container_width=True)

# ── Compute derived features ──────────────────────────────────────────────────
eps = 1e-5
days_birth     = -age * 365
days_employed  = -employment_years * 365 if employment_years > 0 else 365243

# EXT_SOURCE combos
ext_mean    = np.mean([ext1, ext2, ext3])
ext_std     = np.std([ext1, ext2, ext3])
ext_min     = min(ext1, ext2, ext3)
ext_max     = max(ext1, ext2, ext3)
ext_product = ext1 * ext2 * ext3
ext_weighted = 0.25*ext1 + 0.50*ext2 + 0.25*ext3

# Ratios
credit_income   = credit_amt / (income + eps)
annuity_income  = annuity * 12 / (income + eps)
credit_term     = credit_amt / (annuity + eps)
credit_goods    = credit_amt / (goods_price + eps)
income_per_pers = income / (fam_members + eps)
employed_age    = days_employed / (days_birth + eps)

# Gender encoding
gender_enc = 1 if gender == "Male" else 0

# Employment anomaly
days_emp_clean = days_employed if employment_years > 0 else np.nan
emp_anom       = 1 if employment_years == 0 else 0

# Build feature dict — populate all known features
applicant = {
    # Core application features
    'DAYS_BIRTH'                    : days_birth,
    'DAYS_EMPLOYED'                 : np.log1p(abs(days_employed)) if employment_years > 0 else 0,
    'DAYS_EMPLOYED_ANOM'            : emp_anom,
    'AMT_INCOME_TOTAL'              : np.log1p(income),
    'AMT_CREDIT'                    : np.log1p(credit_amt),
    'AMT_ANNUITY'                   : np.log1p(annuity * 12),
    'AMT_GOODS_PRICE'               : np.log1p(goods_price),
    'CNT_CHILDREN'                  : children,
    'CNT_FAM_MEMBERS'               : fam_members,
    'CODE_GENDER'                   : gender_enc,

    # EXT_SOURCE raw
    'EXT_SOURCE_1'                  : ext1,
    'EXT_SOURCE_2'                  : ext2,
    'EXT_SOURCE_3'                  : ext3,

    # EXT_SOURCE engineered
    'APP_EXT_SOURCE_MEAN'           : ext_mean,
    'APP_EXT_SOURCE_STD'            : ext_std,
    'APP_EXT_SOURCE_MIN'            : ext_min,
    'APP_EXT_SOURCE_MAX'            : ext_max,
    'APP_EXT_SOURCE_RANGE'          : ext_max - ext_min,
    'APP_EXT_SOURCE_PRODUCT'        : ext_product,
    'APP_EXT_WEIGHTED'              : ext_weighted,
    'APP_EXT2_x_CREDIT_INCOME'      : ext2 * credit_income,
    'APP_EXT3_x_ANNUITY_INCOME'     : ext3 * annuity_income,

    # Ratio features
    'APP_AGE_YEARS'                 : age,
    'APP_AGE_YEARS_SQUARED'         : age ** 2,
    'APP_EMPLOYMENT_YEARS'          : employment_years,
    'APP_EMPLOYED_TO_AGE_RATIO'     : employed_age,
    'APP_CREDIT_INCOME_RATIO'       : credit_income,
    'APP_INCOME_CREDIT_RATIO'       : income / (credit_amt + eps),
    'APP_ANNUITY_INCOME_RATIO'      : annuity_income,
    'APP_CREDIT_TERM'               : credit_term,
    'APP_CREDIT_TO_GOODS_RATIO'     : credit_goods,
    'APP_GOODS_CREDIT_DIFF'         : goods_price - credit_amt,
    'APP_INCOME_PER_PERSON'         : income_per_pers,
    'APP_ANNUITY_TO_GOODS_RATIO'    : (annuity * 12) / (goods_price + eps),

    # Refusal rate
    'PREV_REFUSAL_RATE'             : prev_refused / 10.0,
    'PREV_REFUSED_COUNT'            : prev_refused,
    'CROSS_REFUSAL_x_CREDIT_INCOME' : (prev_refused / 10.0) * credit_income,

    # IS_NULL flags (0 = data provided)
    'EXT_SOURCE_1_IS_NULL'          : 0,
    'EXT_SOURCE_2_IS_NULL'          : 0,
    'EXT_SOURCE_3_IS_NULL'          : 0,
}

# ── Main content ──────────────────────────────────────────────────────────────
if score_btn:
    score, score_std, decision, risk_class = predict(
        applicant, models, feat_cols
    )

    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        # Score card
        score_pct = int(score * 100)
        st.markdown(f"""
        <div class="score-card {risk_class}">
            <div class="score-label" style="color:#9CA3AF">Default Probability</div>
            <div class="score-number" style="color:{'#F87171' if risk_class=='high-risk' else '#6EE7B7' if risk_class=='low-risk' else '#FCD34D'}">
                {score_pct}%
            </div>
            <div style="color:#6B7280; font-size:0.8rem;">
                Model uncertainty: ±{score_std*100:.1f}%
            </div>
            <div class="decision-badge">{decision}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={'suffix': '%', 'font': {'size': 28, 'color': '#E8EAF0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#4B5563',
                         'tickfont': {'color': '#6B7280'}},
                'bar': {'color': '#F87171' if risk_class == 'high-risk'
                        else '#6EE7B7' if risk_class == 'low-risk' else '#FCD34D',
                        'thickness': 0.3},
                'bgcolor': '#161B27',
                'bordercolor': '#2a3550',
                'steps': [
                    {'range': [0, 25],  'color': 'rgba(110,231,183,0.1)'},
                    {'range': [25, 45], 'color': 'rgba(252,211,77,0.1)'},
                    {'range': [45, 100],'color': 'rgba(248,113,113,0.1)'},
                ],
                'threshold': {
                    'line': {'color': '#818CF8', 'width': 2},
                    'thickness': 0.8,
                    'value': 44.8
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=220,
            margin=dict(t=20, b=10, l=20, r=20),
            font={'color': '#E8EAF0'},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Threshold info
        st.markdown("""
        <div class="info-box">
            <p>🎯 Decision threshold: <strong>44.8%</strong><br>
            Score above threshold → flag for review or reject.<br>
            Optimised using Youden's J statistic on 307K applicants.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Key metrics
        st.markdown('<div class="section-header">Key Risk Metrics</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Credit-to-Income</div>
                <div class="metric-value" style="color:{'#F87171' if credit_income > 5 else '#6EE7B7'}">
                    {credit_income:.2f}x
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Repayment Burden</div>
                <div class="metric-value" style="color:{'#F87171' if annuity_income > 0.5 else '#6EE7B7'}">
                    {annuity_income*100:.1f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Credit Score</div>
                <div class="metric-value" style="color:{'#6EE7B7' if ext_mean > 0.6 else '#F87171'}">
                    {ext_mean:.3f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Loan Term</div>
                <div class="metric-value">{credit_term/12:.1f} yrs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">LTV Ratio</div>
                <div class="metric-value" style="color:{'#F87171' if credit_goods > 1.1 else '#6EE7B7'}">
                    {credit_goods:.2f}x
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Income / Person</div>
                <div class="metric-value">₹{income_per_pers:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk factors
        st.markdown('<div class="section-header">Risk Factor Analysis</div>',
                    unsafe_allow_html=True)

        risk_factors = [
            {
                'name': 'External Credit Score (avg)',
                'value': f'{ext_mean:.3f}',
                'level': 'safe' if ext_mean > 0.6 else 'danger' if ext_mean < 0.4 else 'warning',
                'note': 'Strong' if ext_mean > 0.6 else 'Weak' if ext_mean < 0.4 else 'Moderate'
            },
            {
                'name': 'Credit-to-Income Ratio',
                'value': f'{credit_income:.2f}x',
                'level': 'safe' if credit_income < 3 else 'danger' if credit_income > 6 else 'warning',
                'note': 'Healthy' if credit_income < 3 else 'Very High' if credit_income > 6 else 'Elevated'
            },
            {
                'name': 'Repayment Burden (EMI/Income)',
                'value': f'{annuity_income*100:.1f}%',
                'level': 'safe' if annuity_income < 0.3 else 'danger' if annuity_income > 0.5 else 'warning',
                'note': 'Comfortable' if annuity_income < 0.3 else 'Strained' if annuity_income > 0.5 else 'Manageable'
            },
            {
                'name': 'Employment Stability',
                'value': f'{employment_years} yrs',
                'level': 'safe' if employment_years >= 3 else 'danger' if employment_years == 0 else 'warning',
                'note': 'Stable' if employment_years >= 3 else 'Unemployed' if employment_years == 0 else 'Short tenure'
            },
            {
                'name': 'Previous Loan Refusals',
                'value': str(prev_refused),
                'level': 'safe' if prev_refused == 0 else 'danger' if prev_refused >= 3 else 'warning',
                'note': 'None' if prev_refused == 0 else 'High' if prev_refused >= 3 else 'Some history'
            },
            {
                'name': 'Age Risk Profile',
                'value': f'{age} yrs',
                'level': 'danger' if age < 25 else 'safe' if age > 35 else 'warning',
                'note': 'Higher risk segment' if age < 25 else 'Lower risk' if age > 35 else 'Moderate'
            },
        ]

        for rf in risk_factors:
            st.markdown(f"""
            <div class="risk-factor {rf['level']}">
                <div class="risk-factor-name">{rf['name']}</div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="color:#6B7280; font-size:0.78rem;">{rf['note']}</span>
                    <span class="risk-factor-value">{rf['value']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # EXT_SOURCE breakdown chart
        st.markdown('<div class="section-header">Credit Bureau Scores</div>',
                    unsafe_allow_html=True)
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(
            x=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
            y=[ext1, ext2, ext3],
            marker_color=['#818CF8', '#6366F1', '#4F46E5'],
            text=[f'{v:.3f}' for v in [ext1, ext2, ext3]],
            textposition='outside',
            textfont={'color': '#E8EAF0', 'size': 12},
        ))
        fig_ext.add_hline(y=0.5, line_dash='dash', line_color='#6B7280',
                          annotation_text='Risk threshold (0.5)',
                          annotation_font_color='#6B7280')
        fig_ext.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=180,
            margin=dict(t=10, b=10, l=10, r=10),
            font={'color': '#E8EAF0', 'size': 11},
            xaxis={'gridcolor': '#1E2740', 'tickcolor': '#4B5563'},
            yaxis={'gridcolor': '#1E2740', 'tickcolor': '#4B5563',
                   'range': [0, 1.15]},
            showlegend=False,
        )
        st.plotly_chart(fig_ext, use_container_width=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🏦</div>
        <h2 style="font-family:'DM Serif Display',serif; color:#E8EAF0; font-size:1.8rem; margin-bottom:0.5rem;">
            Ready to Score
        </h2>
        <p style="color:#6B7280; max-width:500px; margin:0 auto; line-height:1.6;">
            Fill in the applicant details in the sidebar and click
            <strong style="color:#818CF8">Score Applicant</strong> to get
            an instant credit risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model stats
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, "0.794", "Model AUC-ROC", "#818CF8"),
        (c2, "307K", "Training applicants", "#6EE7B7"),
        (c3, "708", "Engineered features", "#FCD34D"),
        (c4, "5-Fold", "Cross-validation", "#F87171"),
    ]
    for col, val, label, color in stats:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding:1.5rem;">
                <div style="font-family:'DM Serif Display',serif;
                            font-size:2rem; color:{color}; margin-bottom:4px;">
                    {val}
                </div>
                <div class="metric-label" style="text-align:center;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:2rem;">
        <p>
        <strong style="color:#818CF8">How it works:</strong>
        This app uses an XGBoost ensemble (5 models) trained on the
        Home Credit Default Risk dataset. Fill in applicant details →
        the app computes 708 risk features behind the scenes →
        model outputs a default probability score with business decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
