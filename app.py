import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Scorer",
    page_icon="🩺",
    layout="wide"
)

# ── Load model and scaler ─────────────────────────────────────
@st.cache_resource
def load_model():
    model  = pickle.load(open('model/xgb_model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# ── Header ────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Scoring Engine")
st.markdown("Enter patient details in the sidebar to generate a risk score and explanation.")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Patient Details")

pregnancies  = st.sidebar.slider("Pregnancies",               0,    17,   1)
glucose      = st.sidebar.slider("Glucose (mg/dL)",           50,   200,  120)
blood_press  = st.sidebar.slider("Blood Pressure (mm Hg)",    40,   130,  70)
skin_thick   = st.sidebar.slider("Skin Thickness (mm)",       5,    100,  20)
insulin      = st.sidebar.slider("Insulin (mu U/ml)",         10,   900,  80)
bmi          = st.sidebar.slider("BMI",                       10.0, 70.0, 28.0)
dpf          = st.sidebar.slider("Diabetes Pedigree Function",0.0,  2.5,  0.5)
age          = st.sidebar.slider("Age",                       18,   90,   35)

predict_btn  = st.sidebar.button("Calculate Risk", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────
if predict_btn:

    input_df     = pd.DataFrame([[pregnancies, glucose, blood_press, skin_thick,
                                   insulin, bmi, dpf, age]], columns=features)
    input_scaled = scaler.transform(input_df)
    risk_prob    = model.predict_proba(input_scaled)[0][1]
    risk_label   = "High Risk" if risk_prob >= 0.5 else "Low Risk"
    risk_colour  = "#e74c3c" if risk_prob >= 0.5 else "#27ae60"

    # ── Row 1: KPI cards ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Risk Score")
        st.markdown(
            f"<h1 style='color:{risk_colour}; margin:0'>{risk_prob:.1%}</h1>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Classification")
        st.markdown(
            f"<h1 style='color:{risk_colour}; margin:0'>{risk_label}</h1>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("### Glucose Level")
        glucose_class = (
            "Diabetic range"    if glucose > 125 else
            "Pre-diabetic range" if glucose > 99  else
            "Normal range"
        )
        st.markdown(f"<h1 style='margin:0'>{glucose} mg/dL</h1>", unsafe_allow_html=True)
        st.caption(glucose_class)

    st.divider()

    # ── Row 2: SHAP waterfall + patient summary ───────────────
    col4, col5 = st.columns([3, 2])

    with col4:
        st.markdown("### Why this score?")
        st.caption("Each bar shows how much that feature increased or decreased the risk.")

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=features
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col5:
        st.markdown("### Patient summary")
        summary = pd.DataFrame({
            'Feature': features,
            'Value':   [pregnancies, glucose, blood_press, skin_thick,
                        insulin, bmi, dpf, age],
            'SHAP impact': shap_values[0].round(4)
        }).sort_values('SHAP impact', key=abs, ascending=False)

        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── Row 3: Population context ─────────────────────────────
    st.markdown("### How does this patient compare to the dataset?")

    df_all = pd.read_csv('data/diabetes_cleaned.csv')
    
    col6, col7 = st.columns(2)

    with col6:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(df_all['Glucose'], bins=30, color='steelblue',
                 edgecolor='white', alpha=0.7, label='All patients')
        ax2.axvline(glucose, color='#e74c3c', linewidth=2,
                    label=f'This patient ({glucose})')
        ax2.set_title('Glucose distribution')
        ax2.set_xlabel('Glucose (mg/dL)')
        ax2.set_ylabel('Count')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col7:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.hist(df_all['BMI'], bins=30, color='steelblue',
                 edgecolor='white', alpha=0.7, label='All patients')
        ax3.axvline(bmi, color='#e74c3c', linewidth=2,
                    label=f'This patient ({bmi})')
        ax3.set_title('BMI distribution')
        ax3.set_xlabel('BMI')
        ax3.set_ylabel('Count')
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── Clinical note ─────────────────────────────────────────
    st.divider()
    st.info(
        "⚠️ This tool is for educational and portfolio purposes only. "
        "It is not a medical device and should not be used for clinical decisions."
    )

else:
    st.info("👈 Enter patient details in the sidebar and click **Calculate Risk** to begin.")