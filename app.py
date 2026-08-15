import streamlit as st
import numpy as np
import os
import pandas as pd
from modules.synthetic_data import generate_synthetic_patient_records
from core.graph import compile_pediatric_mas_graph

# --- Page Configuration ---
st.set_page_config(page_title="Pediatric MAS", page_icon="🩺", layout="wide")

st.title("🩺 Pediatric Predictive MAS Dashboard")
st.markdown("Observe the Multi-Agent System analyze time-series data and formulate clinical plans.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ System Controls")
    
    if st.button("🗑️ Wipe & Reset Data", use_container_width=True):
        data_path = "data/PED-8821_longitudinal.csv"
        if os.path.exists(data_path):
            os.remove(data_path)
            st.success("Data wiped successfully.")
        else:
            st.warning("No data to wipe.")
            
    st.divider()
    run_pipeline = st.button("🚀 Run MAS Pipeline", type="primary", use_container_width=True)

# --- Main Application Logic ---
if run_pipeline:
    patient_id = "PED-8821"
    
    with st.spinner("Generating synthetic longitudinal data..."):
        data_filepath = generate_synthetic_patient_records(patient_id=patient_id, days=120)
        
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

    with st.spinner("Running Multi-Agent Orchestration..."):
        app = compile_pediatric_mas_graph()
        final_state = app.invoke(initial_state)

    # --- NEW: Dynamic Time-Series Visualizations ---
    st.subheader("📈 Patient Longitudinal Time-Series")
    
    # Load the actual generated CSV data
    df = pd.read_csv(final_state["raw_history_path"])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 1. Dynamic Stats Row
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Total Monitored Days", len(df))
    stat2.metric("Historic Flare-Ups", int(df['flare_event'].sum()))
    stat3.metric("Lowest SpO2 Recorded", f"{df['spo2'].min():.1f}%")
    stat4.metric("Peak PM2.5 Exposure", f"{df['pm25'].max():.1f} µg/m³")
    
    # 2. Charts Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Blood Oxygen (SpO2 %) over Time**")
        # Red line for SpO2
        st.line_chart(df[['spo2']], color="#FF4B4B")
        
    with col_chart2:
        st.markdown("**Environmental PM2.5 vs. Acute Flare Events**")
        # Scale the binary flare event (0 or 1) up to the max PM2.5 value so they are visible on the same chart
        df_plot = df[['pm25']].copy()
        df_plot['Flare Event (Scaled)'] = df['flare_event'] * df['pm25'].max() 
        # Blue for pollution, Orange for Flare-ups
        st.line_chart(df_plot, color=["#1f77b4", "#ff7f0e"])

    st.divider()

    # --- Dashboard UI Rendering ---
    
    st.subheader("⚙️ Adaptive Learner Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    acc = final_state["rolling_accuracy_score"] * 100
    calibrated = final_state["is_model_calibrated"]
    
    col1.metric("Rolling Accuracy", f"{acc:.1f}%")
    col2.metric("Calibration Status", "READY" if calibrated else "LEARNING", 
                delta=">= 85%" if calibrated else "< 85%", delta_color="normal" if calibrated else "inverse")
    col3.metric("Acuteness Index", f"{final_state['model_acuteness_score']}/10.0")
    
    weights_str = ", ".join([f"{w:.2f}" for w in final_state['model_weights']])
    col4.metric("Model Bias", f"{final_state['model_bias']:.3f}")
    
    st.caption(f"**Current Learned Weights:** [{weights_str}]")
    st.divider()

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔍 Clinical Critique Pushback")
        st.info(final_state["critique_pushback_notes"])
        
    with col_b:
        st.subheader("👨‍👩‍👧 Final Parent Action Plan")
        st.success(final_state["parent_distilled_action_plan"])

    st.divider()

    with st.expander("📜 View Full Agent Execution Audit Trail"):
        for log in final_state["execution_logs"]:
            st.text(f"• {log}")