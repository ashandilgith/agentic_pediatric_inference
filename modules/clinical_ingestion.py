# Import the shared state schema
from core.state import PediatricHealthState

def ingest_medical_records(state: PediatricHealthState) -> PediatricHealthState:
    """Parses raw patient notes and establishes clinical tracking guardrails."""
    # Append execution log for transparency
    state["execution_logs"].append("Clinical Ingestion: Ingesting unstructured EHR and establishing baselines.")
    
    # Unstructured medical chart text to simulate long-context EHR intake
    raw_notes = state.get("unstructured_medical_notes", "")
    
    # Parse critical diagnostic indicators from text (Deterministic NLP logic)
    triggers = []
    if "dust" in raw_notes.lower():
        triggers.append("Dust Mite Allergen Sensitivity")
    if "exercise" in raw_notes.lower():
        triggers.append("Cold Air & Exercise-Induced Hyperreactivity")
    if "smoke" in raw_notes.lower():
        triggers.append("Environmental Particulate/Smoke Vulnerability")
        
    # Set patient specific vital cutoffs
    thresholds = {
        "critical_min_spo2": 91.0,         # Alert if SpO2 drops below 91%
        "warning_max_cough_score": 3.5,    # Alert if cough persists above 3.5
        "extreme_pm25_exposure": 45.0      # Dangerous PM2.5 threshold for this child
    }
    
    # Populate the state with extracted knowledge
    state["extracted_symptom_triggers"] = triggers
    state["vital_thresholds"] = thresholds
    
    # Log successful entity extraction
    state["execution_logs"].append(
        f"Clinical Ingestion: Extracted {len(triggers)} risk triggers and configured safety bounds."
    )
    # Return the updated state
    return state