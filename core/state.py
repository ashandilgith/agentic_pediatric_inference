# Import TypedDict to define explicit type annotations for every key in the state
from typing import TypedDict, Optional, List, Dict, Any

class PediatricHealthState(TypedDict, total=False):
    # --- 1. Patient Identifiers & Baseline Medical Records ---
    patient_id: str                      # Unique patient tracking ID
    age_years: int                       # Age of the child
    primary_diagnosis: str               # e.g., 'Pediatric Asthma with Atopic Dermatitis'
    secondary_conditions: List[str]      # e.g., ['Dust Mite Allergy', 'Exercise-Induced Bronchospasm']
    baseline_fev1_percent: float         # Baseline lung function (Forced Expiratory Volume %)
    baseline_spo2: float                 # Baseline resting blood oxygen saturation (%)
    unstructured_medical_notes: str      # Raw clinical/discharge summary notes
    
    # --- 2. Real-Time Environmental & Home Factors ---
    ambient_aqi: float                   # Outdoor Air Quality Index (0-500)
    outdoor_pm25: float                  # Outdoor PM2.5 particulate matter (ug/m3)
    outdoor_pollen_index: float          # Local pollen count (0-10 scale)
    home_humidity_percent: float         # Indoor humidity level
    home_indoor_mold_risk: bool          # Presence of indoor dampness/mold
    home_pet_dander_present: bool        # Exposure to indoor pets
    
    # --- 3. Qualitative Parent Interview Signals ---
    parent_reported_cough_frequency: int # Daily cough score (1-5 scale)
    parent_reported_night_waking: bool   # Did the child wake up gasping or coughing?
    medication_adherence_percent: float  # Percentage of prescribed controller doses taken
    child_stress_activity_level: str     # 'low', 'moderate', 'intense_exercise'
    daily_checkin: Dict[str, Any]        # Optional real inputs for today; overrides the demo values
    
    # --- 4. Clinical Ingestion Outputs ---
    extracted_symptom_triggers: List[str] # Triggers identified by ingestion agent
    vital_thresholds: Dict[str, float]    # Critical cutoffs (e.g., {'min_spo2': 92.0})
    
    # --- 5. Engineered Time-Series Data ---
    raw_history_path: str                # Filepath to raw synthetic multi-day CSV
    engineered_feature_matrix: List[List[float]] # Windowed numerical vectors for model training
    engineered_target_vector: List[int]  # Binary next-day flare indicators (0 = Stable, 1 = Acute Flare)
    time_series_quality_passed: bool     # Quality gate boolean
    
    # --- 6. Adaptive Online Learning Engine State ---
    model_weights: List[float]           # Current parameter weights vector
    model_bias: float                    # Current intercept/bias term
    training_iteration_count: int        # Number of learning epochs/trials executed
    rolling_accuracy_score: float        # Recent sliding-window accuracy (0.0 to 1.0)
    is_model_calibrated: bool            # True ONLY if rolling accuracy >= 0.85 (85%)
    model_acuteness_score: float         # Raw predicted severity index (0.0 to 10.0)
    model_binary_prediction: int         # 0 (Mild) or 1 (Severe Flare Imminent)
    today_flare_probability: float       # Predicted probability of a flare TOMORROW (0.0 to 1.0)
    model_recall: float                  # Share of past flares the model caught (prequential)
    model_precision: float               # Share of model alerts that were real flares (prequential)
    model_balanced_accuracy: float       # Mean of recall and specificity (robust to class imbalance)
    baseline_accuracy: float             # Accuracy of always predicting "no flare" (the bar to beat)
    flare_events_observed: int           # Number of flare days in the training history
    risk_tier: str                       # 'low', 'elevated' or 'high' after critique reconciliation
    
    # --- 7. Multi-Agent Orchestration & Critique Dialogue ---
    initial_clinical_draft: str          # Preliminary technical diagnosis by first orchestrator
    critique_pushback_notes: str         # Challenges, missed factors, or edge cases flagged by critique
    final_clinical_observation: str      # Technical synthesized consensus
    parent_distilled_action_plan: str    # Simple, empathetic, jargon-free parent summary
    
    # --- 8. System Logs ---
    execution_logs: List[str]            # Audit trail of agent reasoning steps