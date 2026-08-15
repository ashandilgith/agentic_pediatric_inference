# Import shared state
from core.state import PediatricHealthState

def draft_initial_clinical_prediction(state: PediatricHealthState) -> PediatricHealthState:
    """Drafts an initial technical prognosis integrating quantitative and qualitative factors."""
    # Log action
    state["execution_logs"].append("Initial Orchestrator: Synthesizing metrics into preliminary assessment.")
    
    calibrated = state["is_model_calibrated"]
    acc = state["rolling_accuracy_score"] * 100
    acuteness = state["model_acuteness_score"]
    cough = state["parent_reported_cough_frequency"]
    pm25 = state["outdoor_pm25"]
    spo2 = state["baseline_spo2"]
    
    # If the model has achieved >= 85% accuracy, trust its prediction index
    if calibrated:
        confidence_text = f"Validated Adaptive Model ({acc:.1f}% accuracy)"
        if state["model_binary_prediction"] == 1:
            severity = "HIGH RISK of acute flare-up in next 24 hours"
        else:
            severity = "MODERATE/STABLE risk profile"
    else:
        # Fallback to rule-based logic if model is still in calibration phase
        confidence_text = f"Rule-Based Baseline (Model accuracy {acc:.1f}% below 85% threshold)"
        if cough >= 4 or pm25 > 40.0:
            severity = "ELEVATED RISK due to environmental triggers and cough frequency"
        else:
            severity = "LOW/STABLE baseline"
            
    # Formulate clinical draft
    draft = (
        f"[Preliminary Clinical Assessment]\n"
        f"Assessment Basis: {confidence_text}\n"
        f"Predicted Acuteness Index: {acuteness}/10.0\n"
        f"Status: {severity}\n"
        f"Key Contributing Factors: Outdoor PM2.5 at {pm25} ug/m3, Cough Severity: {cough}/5, Resting SpO2: {spo2}%."
    )
    
    # Store in state
    state["initial_clinical_draft"] = draft
    return state