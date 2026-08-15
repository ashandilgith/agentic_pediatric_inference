import streamlit as st
import numpy as np
import os
import pandas as pd
from modules.synthetic_data import generate_synthetic_patient_records
from core.graph import compile_pediatric_mas_graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pediatric Predictive Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN UI STYLING
# ============================================================

st.markdown("""
<style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f5f7fa;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* ---------- Typography ---------- */

    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: -0.025em;
        color: #14213d;
    }

    h1 {
        font-size: 2.35rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 650 !important;
    }

    p, label, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
    }

    /* ---------- Header ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #14213d 0%,
            #1d3557 55%,
            #244d63 100%
        );
        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(20, 33, 61, 0.14);
        color: white;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 7px;
        color: white;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d9e7ee;
        margin-bottom: 18px;
    }

    .system-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        color: #e7f7f7;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* ---------- Section Labels ---------- */

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #6c7a89;
        margin-bottom: 8px;
    }

    /* ---------- Metric Cards ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e8ee;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(20, 33, 61, 0.05);
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #14213d !important;
        font-weight: 700 !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #d8dee6;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: #457b9d;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #111c2e;
        border-right: 1px solid #25344a;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #e7edf5 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: #1d2b42;
        color: #e7edf5;
        border: 1px solid #34445c;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #263a55;
        border-color: #4f7594;
    }

    /* ---------- Information Cards ---------- */

    .info-card {
        background: white;
        border: 1px solid #e3e8ee;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 14px rgba(20, 33, 61, 0.045);
        margin-bottom: 15px;
    }

    .info-card-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 700;
        color: #718096;
        margin-bottom: 10px;
    }

    /* ---------- Status ---------- */

    .status-ready {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #e7f7f1;
        color: #16705a;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .status-learning {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #fff5df;
        color: #946200;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* ---------- Audit Trail ---------- */

    .audit-header {
        color: #14213d;
        font-weight: 650;
    }

    /* ---------- Divider ---------- */

    hr {
        border: none;
        border-top: 1px solid #e1e6eb;
        margin: 28px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        Pediatric Predictive Intelligence
    </div>
    <div class="hero-subtitle">
        Agentic longitudinal analysis for pediatric chronic illness.
        Monitor patient trajectories, environmental risk, model learning,
        and generated clinical action plans.
    </div>
    <span class="system-pill">MULTI-AGENT CLINICAL INTELLIGENCE SYSTEM</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="section-label">SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Pediatric MAS"
    )

    st.caption(
        "Longitudinal patient analysis and adaptive clinical reasoning."
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Controls</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Reset Patient Data",
        use_container_width=True
    ):
        data_path = "data/PED-8821_longitudinal.csv"

        if os.path.exists(data_path):
            os.remove(data_path)
            st.success("Patient data reset.")
        else:
            st.warning("No patient data found.")

    st.divider()

    run_pipeline = st.button(
        "Run MAS Pipeline",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption("SYSTEM STATUS")
    st.markdown(
        '<span class="status-ready">● ONLINE</span>',
        unsafe_allow_html=True
    )

    st.caption("Synthetic research environment")


# ============================================================
# MAIN APPLICATION LOGIC
# ============================================================

if run_pipeline:

    patient_id = "PED-8821"

    # --------------------------------------------------------
    # DATA GENERATION
    # --------------------------------------------------------

    with st.spinner("Generating synthetic longitudinal data..."):

        data_filepath = generate_synthetic_patient_records(
            patient_id=patient_id,
            days=120
        )

    initial_state = {
        "patient_id": patient_id,
        "age_years": 8,
        "primary_diagnosis": "Pediatric Chronic Bronchial Asthma",
        "secondary_conditions": ["Dust Mite Allergy", "Atopic Eczema"],
        "baseline_fev1_percent": 82.0,
        "baseline_spo2": 97.5,
        "unstructured_medical_notes": "Patient is an 8-year-old male with persistent moderate asthma...",
        "ambient_aqi": 0.0,
        "outdoor_pm25": 0.0,
        "outdoor_pollen_index": 0.0,
        "home_humidity_percent": 0.0,
        "home_indoor_mold_risk": False,
        "home_pet_dander_present": False,
        "parent_reported_cough_frequency": 0,
        "parent_reported_night_waking": False,
        "medication_adherence_percent": 1.0,
        "child_stress_activity_level": "low",
        "extracted_symptom_triggers": [],
        "vital_thresholds": {},
        "raw_history_path": data_filepath,
        "engineered_feature_matrix": [],
        "engineered_target_vector": [],
        "time_series_quality_passed": False,
        "model_weights": [],
        "model_bias": 0.0,
        "training_iteration_count": 0,
        "rolling_accuracy_score": 0.0,
        "is_model_calibrated": False,
        "model_acuteness_score": 0.0,
        "model_binary_prediction": 0,
        "initial_clinical_draft": "",
        "critique_pushback_notes": "",
        "final_clinical_observation": "",
        "parent_distilled_action_plan": "",
        "execution_logs": ["System Boot Initialized."]
    }

    # --------------------------------------------------------
    # MAS EXECUTION
    # --------------------------------------------------------

    with st.spinner("Running Multi-Agent Orchestration..."):

        app = compile_pediatric_mas_graph()

        final_state = app.invoke(initial_state)


    # ========================================================
    # PATIENT OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-label">PATIENT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    overview1, overview2, overview3 = st.columns(3)

    with overview1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Patient</div>
            <strong>PED-8821</strong><br>
            8-year-old male
        </div>
        """, unsafe_allow_html=True)

    with overview2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Primary Diagnosis</div>
            <strong>Pediatric Chronic Bronchial Asthma</strong><br>
            Persistent moderate classification
        </div>
        """, unsafe_allow_html=True)

    with overview3:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Comorbidities</div>
            <strong>Dust Mite Allergy</strong><br>
            Atopic Eczema
        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # LONGITUDINAL DATA
    # ========================================================

    st.markdown(
        '<div class="section-label">LONGITUDINAL MONITORING</div>',
        unsafe_allow_html=True
    )

    st.subheader("Patient Trajectory")

    # Load actual generated CSV data
    df = pd.read_csv(final_state["raw_history_path"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)


    # --------------------------------------------------------
    # DYNAMIC STATS
    # --------------------------------------------------------

    stat1, stat2, stat3, stat4 = st.columns(4)

    stat1.metric(
        "Monitoring Period",
        f"{len(df)} days"
    )

    stat2.metric(
        "Historic Flare-Ups",
        int(df["flare_event"].sum())
    )

    stat3.metric(
        "Lowest SpO₂",
        f"{df['spo2'].min():.1f}%"
    )

    stat4.metric(
        "Peak PM2.5",
        f"{df['pm25'].max():.1f} µg/m³"
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:

        st.markdown(
            '<div class="info-card-title">PHYSIOLOGICAL SIGNAL</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Blood Oxygen · SpO₂")

        st.line_chart(
            df[["spo2"]],
            color="#36A3A3"
        )

        st.caption(
            "Longitudinal peripheral oxygen saturation."
        )


    with col_chart2:

        st.markdown(
            '<div class="info-card-title">ENVIRONMENTAL RISK</div>',
            unsafe_allow_html=True
        )

        st.markdown("### PM2.5 Exposure vs Flare Events")

        df_plot = df[["pm25"]].copy()

        df_plot["Flare Event (Scaled)"] = (
            df["flare_event"] * df["pm25"].max()
        )

        st.line_chart(
            df_plot,
            color=["#457B9D", "#E07A5F"]
        )

        st.caption(
            "Environmental particulate exposure alongside recorded flare events."
        )


    st.divider()


    # ========================================================
    # ADAPTIVE LEARNER
    # ========================================================

    st.markdown(
        '<div class="section-label">ADAPTIVE MODEL</div>',
        unsafe_allow_html=True
    )

    st.subheader("Model Learning State")

    col1, col2, col3, col4 = st.columns(4)

    acc = final_state["rolling_accuracy_score"] * 100
    calibrated = final_state["is_model_calibrated"]

    col1.metric(
        "Rolling Accuracy",
        f"{acc:.1f}%"
    )

    col2.metric(
        "Calibration",
        "READY" if calibrated else "LEARNING",
        delta="≥ 85%" if calibrated else "< 85%",
        delta_color="normal" if calibrated else "inverse"
    )

    col3.metric(
        "Acuteness Index",
        f"{final_state['model_acuteness_score']}/10.0"
    )

    weights_str = ", ".join(
        [f"{w:.2f}" for w in final_state["model_weights"]]
    )

    col4.metric(
        "Model Bias",
        f"{final_state['model_bias']:.3f}"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">Current Learned Weights</div>
            <code>[{weights_str}]</code>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # AGENTIC CLINICAL REASONING
    # ========================================================

    st.markdown(
        '<div class="section-label">AGENTIC REASONING</div>',
        unsafe_allow_html=True
    )

    st.subheader("Clinical Intelligence Output")

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Agent Critique
                </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            final_state["critique_pushback_notes"]
        )

        st.markdown("</div>", unsafe_allow_html=True)


    with col_b:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Parent Action Plan
                </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            final_state["parent_distilled_action_plan"]
        )

        st.markdown("</div>", unsafe_allow_html=True)


    st.divider()


    # ========================================================
    # AUDIT TRAIL
    # ========================================================

    with st.expander("Agent Execution Audit Trail"):

        st.markdown(
            '<div class="audit-header">Multi-Agent System Execution Log</div>',
            unsafe_allow_html=True
        )

        for log in final_state["execution_logs"]:

            st.text(
                f"• {log}"
            )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="info-card" style="text-align:center; padding:55px 30px;">
            <div style="
                font-size:2.2rem;
                margin-bottom:12px;
            ">
                ◉
            </div>

            <div style="
                font-size:1.35rem;
                font-weight:700;
                color:#14213d;
                margin-bottom:8px;
            ">
                Awaiting Patient Analysis
            </div>

            <div style="
                color:#718096;
                max-width:650px;
                margin:auto;
                line-height:1.6;
            ">
                Run the MAS Pipeline to generate a synthetic longitudinal
                patient record and observe the multi-agent clinical
                intelligence workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )