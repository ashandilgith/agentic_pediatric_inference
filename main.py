# Import synthetic generator to create local training CSV
from modules.synthetic_data import generate_synthetic_patient_records
# Import graph compiler
from core.graph import compile_pediatric_mas_graph

if __name__ == "__main__":
    print("=================================================================")
    print("PEDIATRIC CHRONIC CARE PREDICTIVE MULTI-AGENT SYSTEM (MAS)")
    print("=================================================================\n")
    
    # Step 1: Generate 120 days of synthetic longitudinal records
    patient_id = "PED-8821"
    data_filepath = generate_synthetic_patient_records(patient_id=patient_id, days=120)
    print(f"✅ Generated synthetic patient longitudinal records at: {data_filepath}\n")
    
    # Step 2: Initialize high-dimensional shared state
    initial_state = {
        "patient_id": patient_id,
        "age_years": 8,
        "primary_diagnosis": "Pediatric Chronic Bronchial Asthma",
        "secondary_conditions": ["Dust Mite Allergy", "Atopic Eczema"],
        "baseline_fev1_percent": 82.0,
        "baseline_spo2": 97.5,
        "unstructured_medical_notes": (
            "Patient is an 8-year-old male with persistent moderate asthma. "
            "History of severe flare-ups when exposed to dust mites, smoke, or intense exercise in cold air. "
            "Prescribed Fluticasone controller daily and Albuterol rescue as needed."
        ),
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
    
    # Step 3: Compile and run graph
    print("🧠 Starting LangGraph Multi-Agent Orchestration...\n")
    app = compile_pediatric_mas_graph()
    final_output_state = app.invoke(initial_state)
    
    # Step 4: Display Agent Reasoning Logs
    print("--- 📜 AGENT EXECUTION AUDIT TRAIL ---")
    for log in final_output_state["execution_logs"]:
        print(f"  • {log}")
    print("--------------------------------------\n")
    
    # Step 5: Display Adaptive Learner Metrics
    print("--- ⚙️ ADAPTIVE LEARNER METRICS ---")
    acc = final_output_state["rolling_accuracy_score"] * 100
    status = "READY (>=85%)" if final_output_state["is_model_calibrated"] else "CALIBRATING (<85%)"
    print(f"Rolling Accuracy: {acc:.2f}% | Status: {status}")
    print(f"Learned Parameters: Weights={np.round(final_output_state['model_weights'], 3)}, Bias={final_output_state['model_bias']:.3f}")
    print(f"Calculated Acuteness Index: {final_output_state['model_acuteness_score']}/10.0\n")
    
    # Step 6: Display Critique Pushback
    print("--- 🔍 CLINICAL CRITIQUE PUSHBACK ---")
    print(final_output_state["critique_pushback_notes"])
    print("------------------------------------\n")
    
    # Step 7: Display Final Parent Plan
    print("--- 👨‍👩‍👧 PARENT-FACING ACTION PLAN ---")
    print(final_output_state["parent_distilled_action_plan"])
    print("=================================================================\n")