import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ChurnGuard | Churn Risk Predictor",
                   page_icon="📉", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
       padding: 26px 32px; border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: #FFFFFF; margin: 0; font-size: 1.9rem;}
.hero p {color: #CBD5E1; margin: 6px 0 0 0;}
.risk-card {padding: 18px 20px; border-radius: 12px; border-left: 8px solid;
            margin-bottom: 14px;}
.risk-card h3 {margin: 0 0 4px 0;}
.risk-card p {margin: 0;}
.high {background: #FEE2E2; border-color: #DC2626; color: #7F1D1D;}
.medium {background: #FEF3C7; border-color: #D97706; color: #78350F;}
.low {background: #DCFCE7; border-color: #16A34A; color: #14532D;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 12px 16px;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ---------- Load saved files ----------
@st.cache_resource
def load_files():
    return (joblib.load("models/best_model.pkl"),
            joblib.load("models/scaler.pkl"),
            joblib.load("models/columns.pkl"),
            float(joblib.load("models/threshold.pkl")))

model, scaler, columns, threshold = load_files()

REQUIRED = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]


# ---------- Helper functions ----------
def preprocess(raw):
    data = pd.get_dummies(raw, dtype=int)
    data = data.reindex(columns=columns, fill_value=0)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    data[num_cols] = scaler.transform(data[num_cols])
    return data


def risk_info(prob):
    if prob >= threshold:
        return ("HIGH RISK", "high", "#DC2626",
                "This customer is likely to leave. Contact them with a retention offer.")
    if prob >= 0.4:
        return ("MEDIUM RISK", "medium", "#D97706",
                "Keep an eye on this customer and consider a soft check-in.")
    return "LOW RISK", "low", "#16A34A", "This customer is likely to stay."


def make_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 48}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "#DCFCE7"},
                         {"range": [40, threshold * 100], "color": "#FEF3C7"},
                         {"range": [threshold * 100, 100], "color": "#FEE2E2"}]}))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def get_actions(contract, tech, internet, payment, tenure, monthly):
    tips = []
    if contract == "Month-to-month":
        tips.append("Offer a discount to switch to a 1-year or 2-year contract.")
    if tech == "No" and internet != "No":
        tips.append("Give a free 3-month Tech Support trial.")
    if payment == "Electronic check":
        tips.append("Encourage automatic payment with a small bill credit.")
    if internet == "Fiber optic" and monthly > 80:
        tips.append("Review the plan price and offer a better-value bundle.")
    if tenure < 6:
        tips.append("Schedule an onboarding call within the first 30 days.")
    if not tips:
        tips.append("No special action needed. Keep monitoring this customer.")
    return tips


# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>📉 ChurnGuard</h1>
  <p>AI-powered early warning system for telecom customer retention</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Single Customer", "📂 Bulk Analysis", "📊 Model Insights", "ℹ️ About"])

# ---------- Tab 1: Single customer ----------
with tab1:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
        with c2:
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes"])
            tech = st.selectbox("Tech Support", ["No", "Yes"])
        with c3:
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            st.caption("Total Charges is calculated automatically (tenure x monthly).")

        with st.expander("Advanced details (optional)"):
            a1, a2, a3 = st.columns(3)
            with a1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
            with a2:
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multi = st.selectbox("Multiple Lines", ["No", "Yes"])
                backup = st.selectbox("Online Backup", ["No", "Yes"])
            with a3:
                device = st.selectbox("Device Protection", ["No", "Yes"])
                tv = st.selectbox("Streaming TV", ["No", "Yes"])
                movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Predict Churn Risk",
                                          type="primary", width="stretch")

    if submitted:
        no_net = internet == "No"
        net = lambda v: "No internet service" if no_net else v
        raw = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": "No phone service" if phone == "No" else multi,
            "InternetService": internet, "OnlineSecurity": net(security),
            "OnlineBackup": net(backup), "DeviceProtection": net(device),
            "TechSupport": net(tech), "StreamingTV": net(tv),
            "StreamingMovies": net(movies), "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}])

        prob = float(model.predict_proba(preprocess(raw))[0][1])
        score = round(prob * 100, 1)
        level, css, color, msg = risk_info(prob)

        st.divider()
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_gauge(score, color), width="stretch")
        with right:
            st.markdown(f"""<div class="risk-card {css}">
                <h3>{level}</h3><p>{msg}</p></div>""", unsafe_allow_html=True)
            st.markdown("##### 💡 Recommended retention actions")
            for tip in get_actions(contract, tech, internet, payment, tenure, monthly):
                st.markdown(f"- {tip}")

# ---------- Tab 2: Bulk analysis ----------
with tab2:
    st.subheader("Bulk Customer Risk Analysis")
    st.write("Upload a CSV file of customers (same columns as the Telco dataset). "
             "The system scores every customer and ranks them by churn risk.")
    file = st.file_uploader("Upload customer CSV", type="csv")

    if file is not None:
        df_up = pd.read_csv(file)
        missing = [c for c in REQUIRED if c not in df_up.columns]
        if missing:
            st.error(f"Your file is missing these columns: {', '.join(missing)}")
        else:
            work = df_up[REQUIRED].copy()
            work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce").fillna(0)
            probs = model.predict_proba(preprocess(work))[:, 1]

            out = pd.DataFrame({
                "Customer": df_up["customerID"] if "customerID" in df_up.columns
                            else range(1, len(df_up) + 1),
                "Tenure": df_up["tenure"],
                "Contract": df_up["Contract"],
                "Monthly Charges": df_up["MonthlyCharges"],
                "Risk Score (%)": (probs * 100).round(1)})
            out["Risk Level"] = pd.cut(probs, [-0.01, 0.4, threshold, 1.01],
                                       labels=["Low", "Medium", "High"])
            out = out.sort_values("Risk Score (%)", ascending=False)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Customers", len(out))
            k2.metric("High risk", int((out["Risk Level"] == "High").sum()))
            k3.metric("Medium risk", int((out["Risk Level"] == "Medium").sum()))
            k4.metric("Low risk", int((out["Risk Level"] == "Low").sum()))

            counts = out["Risk Level"].value_counts().reset_index()
            counts.columns = ["Risk Level", "Customers"]
            fig = px.pie(counts, names="Risk Level", values="Customers", hole=0.5,
                         color="Risk Level",
                         color_discrete_map={"High": "#DC2626", "Medium": "#D97706",
                                             "Low": "#16A34A"})
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

            st.markdown("##### Customers ranked by risk (highest first)")
            st.dataframe(out, width="stretch", hide_index=True,
                         column_config={"Risk Score (%)": st.column_config.ProgressColumn(
                             "Risk Score (%)", min_value=0, max_value=100, format="%.1f")})
            st.download_button("⬇️ Download results as CSV",
                               out.to_csv(index=False), "churn_risk_results.csv",
                               "text/csv")

# ---------- Tab 3: Model insights ----------
with tab3:
    st.subheader("Model Comparison")
    try:
        st.dataframe(pd.read_csv("models/model_comparison.csv"),
                     width="stretch", hide_index=True)
    except Exception:
        st.info("model_comparison.csv not found in the models folder.")

    st.subheader("Top 10 Factors Driving Churn")
    imp = pd.Series(model.feature_importances_, index=columns)
    top = imp.sort_values().tail(10).reset_index()
    top.columns = ["Feature", "Importance"]
    fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                 color_discrete_sequence=["#1E40AF"])
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ---------- Tab 4: About ----------
with tab4:
    st.subheader("About this project")
    st.write("""
    **Predictive Analytics for Customer Churn Risk in Telecommunications** uses
    machine learning to predict which customers are likely to cancel, so a company
    can contact them before they leave.
    """)
    st.markdown(f"""
    - **Dataset:** Telco Customer Churn (7,043 customers, 21 columns)
    - **Algorithms compared:** Logistic Regression, Random Forest, XGBoost
    - **Final model:** XGBoost
    - **High-risk cut-off:** {threshold:.2f}
    - **Risk levels:** Low (below 40%), Medium (40% to cut-off), High (cut-off and above)
    - **Tools:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly
    """)
    