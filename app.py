import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import clean_data, feature_engineering
from src.explain import get_shap_explanation
from src.llm_report import generate_maintenance_report, answer_followup_question
from src import database as db
import sys
import os

# Consistent color theme used across every chart in the app
COLOR_OK = "#1D9E75"       # teal-green: healthy / low risk
COLOR_WARN = "#EF9F27"     # amber: medium risk
COLOR_DANGER = "#E24B4A"   # red: high risk / failure
FAILURE_COLORS = [COLOR_OK, COLOR_DANGER]  # maps to Machine failure = 0 / 1
CATEGORY_PALETTE = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517"]

# Set page config
st.set_page_config(page_title="Predictive Maintenance System", page_icon="⚙️", layout="wide")

# Ensure src modules can be found if running from a different path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom CSS for modern UI (dark-theme friendly)
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 8px;
    }
    .high-risk {
        color: #E24B4A;
        font-weight: bold;
    }
    .pdm-banner {
        background: linear-gradient(90deg, rgba(55,138,221,0.15), rgba(29,158,117,0.10));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 1.2rem;
    }
    .pdm-banner h1 {
        margin: 0;
        font-size: 1.7rem;
    }
    .pdm-banner p {
        margin: 4px 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Data and Model Loading
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        with open("models/feature_names.json", "r") as f:
            feature_names = json.load(f)
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
        return model, scaler, feature_names, metrics
    except Exception as e:
        st.error(f"Failed to load models. Please ensure you've trained them by running 'python src/train_models.py'. Error: {e}")
        return None, None, None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("data/ai4i2020.csv")
        return df
    except:
        return pd.DataFrame()

model, scaler, feature_names, metrics = load_models()
original_df = load_dataset()

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
st.sidebar.title("⚙️ AI Maintenance")
st.sidebar.info(
    "Predictive Maintenance System powered by Machine Learning and Generative AI."
)
risk_threshold = st.sidebar.slider("Global Risk Alert Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05, help="Risk score above which a machine is considered 'High Risk'")

with st.sidebar.expander("ℹ️ About this project"):
    st.markdown("""
    **Dataset:** AI4I 2020 (UCI Machine Learning Repository) — real
    industrial sensor data with a genuinely imbalanced ~3% failure rate.

    **Pipeline:** Logistic Regression, Random Forest, and XGBoost are
    trained and compared; the model with the best F1-score on the
    minority (failure) class is selected automatically.

    **Explainability:** SHAP values show which sensor readings drove
    each individual prediction.

    **Generative AI layer:** Google Gemini turns the risk score and
    SHAP drivers into a plain-English maintenance report, and answers
    follow-up questions grounded in that machine's data.
    """)

# ------------------------------------------------------------------------------
# Main UI setup
# ------------------------------------------------------------------------------
st.markdown("""
<div class="pdm-banner">
    <h1>🏭 Predictive Maintenance Dashboard</h1>
    <p>Machine learning failure prediction, explained and reported by generative AI.</p>
</div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.stop()

db.init_db()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🔮 Upload & Predict", 
    "📈 Sensor Trends", 
    "🤖 AI Report", 
    "ℹ️ Model Info",
    "🗄️ History"
])

# ------------------------------------------------------------------------------
# Tab 1: Overview
# ------------------------------------------------------------------------------
with tab1:
    st.header("Fleet Overview")
    if original_df.empty:
        st.warning("No original dataset found. Please download it to view overview stats.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        total_machines = len(original_df)
        failure_count = original_df["Machine failure"].sum() if "Machine failure" in original_df.columns else 0
        failure_rate = failure_count / total_machines if total_machines > 0 else 0
        
        # Determine best model F1
        best_f1 = max([m["F1-Score"] for m in metrics]) if metrics else 0
        
        col1.metric("Total Machines Evaluated", f"{total_machines:,}")
        col2.metric("Historical Failures", f"{failure_count:,}")
        col3.metric("Overall Failure Rate", f"{failure_rate:.1%}")
        col4.metric("Model F1-Score (Best)", f"{best_f1:.2f}")
        
        st.subheader("Historical Failure Type Distribution")
        if all(c in original_df.columns for c in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']):
            failure_types = original_df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().reset_index()
            failure_types.columns = ['Failure Type', 'Count']
            fig = px.bar(failure_types, x='Failure Type', y='Count', title="Failures by Type", color='Failure Type', color_discrete_sequence=CATEGORY_PALETTE)
            st.plotly_chart(fig, use_container_width=True)
            
# ------------------------------------------------------------------------------
# Helper for predictions
# ------------------------------------------------------------------------------
def run_prediction_pipeline(df_input):
    df_clean = clean_data(df_input)
    df_engineered = feature_engineering(df_clean)
    
    # Ensure all expected features are present
    for col in feature_names:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
            
    X_input = df_engineered[feature_names]
    X_scaled = scaler.transform(X_input)
    
    # Predict
    if hasattr(model, "predict_proba"):
        risk_scores = model.predict_proba(X_scaled)[:, 1]
    else:
        risk_scores = model.predict(X_scaled)
        
    return X_scaled, risk_scores, df_clean

# ------------------------------------------------------------------------------
# Tab 2: Upload & Predict
# ------------------------------------------------------------------------------
with tab2:
    st.header("Predictive Analysis")
    
    input_method = st.radio("Choose Input Method:", ["Upload CSV", "Manual Entry"])
    
    if input_method == "Upload CSV":
        st.caption("No file handy? Run `python generate_demo_data.py` to create `data/demo_fleet_sample.csv` — a small synthetic fleet (not used for training) for demoing this exact flow.")
        uploaded_file = st.file_uploader("Upload sensor readings CSV (matches training schema)", type="csv")
        
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_data.head())
            
            if st.button("Run Prediction"):
                with st.spinner("Analyzing data..."):
                    X_scaled, risk_scores, clean_df = run_prediction_pipeline(new_data)
                    
                    results_df = clean_df.copy()
                    results_df["Failure Risk"] = risk_scores
                    
                    # Highlight based on threshold
                    def color_risk(val):
                        if val > risk_threshold:
                            color = 'red'
                        elif val > 0.3:
                            color = 'orange'
                        else:
                            color = 'green'
                        return f'color: {color}; font-weight: bold'
                    
                    st.subheader("Prediction Results")
                    cols = list(results_df.columns)
                    cols = ['Failure Risk'] + [c for c in cols if c != 'Failure Risk']
                    results_df = results_df[cols]
                    
                    st.dataframe(results_df.style.map(color_risk, subset=['Failure Risk']))
                    
                    # Save results in session state for other tabs
                    st.session_state['pred_results'] = results_df
                    st.session_state['X_scaled'] = X_scaled

                    # Log to prediction history (non-blocking; failures are ignored)
                    db.log_predictions_batch(
                        source="CSV Upload",
                        rows=results_df.to_dict(orient="records"),
                        threshold=risk_threshold,
                    )
                    
                    st.success("Analysis complete. Head to the 'AI Report' tab to investigate high-risk machines.")
                    
    else:
        st.subheader("Manual Sensor Input")
        col1, col2 = st.columns(2)
        with col1:
            type_val = st.selectbox("Machine Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature (K)", value=298.1)
            process_temp = st.number_input("Process temperature (K)", value=308.6)
        with col2:
            rot_speed = st.number_input("Rotational speed (rpm)", value=1551.0)
            torque = st.number_input("Torque (Nm)", value=42.8)
            tool_wear = st.number_input("Tool wear (min)", value=0.0)
            
        if st.button("Predict Single Machine"):
            # NOTE: keys must match the raw dataset's column names exactly
            # (no unit suffixes) or values silently fail to reach the model.
            input_dict = {
                "Type": [type_val],
                "Air temperature": [air_temp],
                "Process temperature": [process_temp],
                "Rotational speed": [rot_speed],
                "Torque": [torque],
                "Tool wear": [tool_wear]
            }
            manual_df = pd.DataFrame(input_dict)
            X_scaled, risk_scores, _ = run_prediction_pipeline(manual_df)
            risk = risk_scores[0]
            
            st.metric("Predicted Failure Risk", f"{risk:.1%}")
            if risk > risk_threshold:
                st.error("High Risk Detected! Immediate inspection recommended.")
            elif risk > 0.3:
                st.warning("Moderate Risk. Schedule maintenance soon.")
            else:
                st.success("Low Risk. Operating normally.")
                
            # Compute SHAP for the manual entry
            top_feats, _ = get_shap_explanation(model, X_scaled[0], feature_names)
            st.session_state['manual_risk'] = risk
            st.session_state['manual_features'] = top_feats
            st.session_state['manual_raw'] = input_dict

            # Log to prediction history (non-blocking; failures are ignored)
            db.log_prediction(
                source="Manual Entry",
                machine_type=type_val,
                sensor_data={k: v[0] for k, v in input_dict.items()},
                risk=risk,
                threshold=risk_threshold,
            )

# ------------------------------------------------------------------------------
# Tab 3: Sensor Trends
# ------------------------------------------------------------------------------
with tab3:
    st.header("Sensor Trends Analysis")
    if not original_df.empty:
        # Just demonstrating trends on the original dataset for visualization
        # In a real app, this might be time-series data for a specific machine.
        # Since ai4i is just point-in-time cross-sectional data, we'll plot a subset.
        st.info("Visualizing a subset of historical data to show general sensor distributions.")
        
        sensor_choice = st.selectbox("Select Sensor", ["Torque", "Rotational speed", "Tool wear", "Air temperature"])
        
        fig = px.histogram(original_df, x=sensor_choice, color="Machine failure", 
                           barmode="overlay", title=f"Distribution of {sensor_choice} by Failure Status",
                           color_discrete_sequence=FAILURE_COLORS)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot for two features
        st.subheader("Feature Interactions")
        scatter_x = st.selectbox("X-Axis", ["Rotational speed", "Torque", "Tool wear"], index=0)
        scatter_y = st.selectbox("Y-Axis", ["Torque", "Rotational speed", "Tool wear"], index=1)
        
        fig2 = px.scatter(original_df, x=scatter_x, y=scatter_y, color="Machine failure",
                          title=f"Interaction: {scatter_x} vs {scatter_y}",
                          color_discrete_sequence=FAILURE_COLORS, opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Original dataset not found.")

# ------------------------------------------------------------------------------
# Tab 4: AI Maintenance Report
# ------------------------------------------------------------------------------
with tab4:
    st.header("Generative AI Maintenance Report")
    
    # We will use the manual entry for the report if it exists, otherwise prompt user
    if 'manual_risk' in st.session_state:
        risk = st.session_state['manual_risk']
        feats = st.session_state['manual_features']
        raw = st.session_state['manual_raw']
        
        st.subheader("SHAP Feature Importance (Current Machine)")
        
        # Plot SHAP
        shap_df = pd.DataFrame(list(feats.items()), columns=['Feature', 'Impact'])
        fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                          title="Key Drivers of Failure Risk (SHAP)",
                          color='Impact', color_continuous_scale=px.colors.diverging.RdBu_r)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.subheader("AI Generated Report")
        if st.button("Generate Report"):
            with st.spinner("Consulting AI Expert..."):
                report = generate_maintenance_report(raw, risk, feats)
                st.session_state['ai_report'] = report
                st.info(report)
                
        if 'ai_report' in st.session_state:
            st.info(st.session_state['ai_report'])
            
            st.subheader("Ask Follow-up Questions")
            user_q = st.text_input("Ask about this machine (e.g., 'What should I check first?'):")
            if user_q:
                with st.spinner("AI is thinking..."):
                    answer = answer_followup_question(user_q, st.session_state['ai_report'], raw, risk, feats)
                    st.success(answer)
    else:
        st.warning("Please run a manual prediction in the 'Upload & Predict' tab first to generate a report.")

# ------------------------------------------------------------------------------
# Tab 5: Model Info
# ------------------------------------------------------------------------------
with tab5:
    st.header("Model Evaluation & Selection")
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df.drop(columns=["Confusion Matrix"]).style.highlight_max(axis=0, color='lightgreen'))
        
        best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax()]['Model']
        st.success(f"**Selected Model:** {best_model_name} (Highest F1-Score on minority class)")
        
        st.markdown("### Why this model?")
        st.markdown("""
        In predictive maintenance, the dataset is highly imbalanced (failures are rare). 
        Accuracy is a misleading metric here because simply predicting 'No Failure' all the time yields >96% accuracy but catches 0 failures.
        Therefore, we optimize for **F1-Score** (harmonic mean of precision and recall) and evaluate **ROC-AUC** to ensure the model distinguishes well between classes.
        """)
    else:
        st.warning("No metrics found. Run 'src/train_models.py' first.")

# ------------------------------------------------------------------------------
# Tab 6: Prediction History
# ------------------------------------------------------------------------------
with tab6:
    st.header("Prediction History")
    st.caption("Every prediction made in this app (CSV upload or manual entry) is logged locally to a SQLite database.")

    history = db.get_history()

    if not history:
        st.info("No predictions logged yet. Run a prediction in the 'Upload & Predict' tab and it'll show up here.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["failure_risk"] = hist_df["failure_risk"].apply(lambda x: f"{x:.1%}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Logged Predictions", len(history))
        high_risk_count = sum(1 for h in history if h["risk_level"] == "High")
        col2.metric("High Risk Logged", high_risk_count)
        col3.metric("Most Recent", history[0]["timestamp"])

        risk_filter = st.multiselect("Filter by risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        filtered = hist_df[hist_df["risk_level"].isin(risk_filter)]

        st.dataframe(
            filtered[["timestamp", "source", "machine_type", "failure_risk", "risk_level"]],
            use_container_width=True,
        )

        if st.button("🗑️ Clear History"):
            if db.clear_history():
                st.success("History cleared.")
                st.rerun()
            else:
                st.error("Could not clear history.")
