# Import pandas for data manipulation
import pandas as pd
# Import numpy for calculations
import numpy as np
# Import shared state
from core.state import PediatricHealthState

def engineer_time_series_features(state: PediatricHealthState) -> PediatricHealthState:
    """Cleans data and builds rolling-window feature matrices for forecasting."""
    # Append log
    state["execution_logs"].append("Time-Series Engineer: Resampling and constructing rolling window features.")
    
    # Read the synthetic longitudinal CSV from disk
    df = pd.read_csv(state["raw_history_path"])
    
    # Ensure chronological sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # --- Feature Engineering: Rolling 3-Day Moving Averages ---
    # Rolling mean for PM2.5 particulate pollution
    df["pm25_3d_avg"] = df["pm25"].rolling(window=3, min_periods=1).mean()
    # Rolling mean for blood oxygen saturation
    df["spo2_3d_avg"] = df["spo2"].rolling(window=3, min_periods=1).mean()
    # Rolling mean for parent cough score
    df["cough_3d_avg"] = df["cough_score"].rolling(window=3, min_periods=1).mean()
    
    # Feature Interaction: Risk multiplier between low adherence and high pollution
    df["exposure_risk_index"] = df["pm25_3d_avg"] * (1.1 - df["med_adherence"])
    
    # Target: Predict next-day flare event
    df["target"] = df["flare_event"]
    
    # Quality Control Check: Detect missing values or zero-variance sensor failure
    has_nulls = df.isnull().values.any()
    zero_variance = (df["spo2"].std() < 0.1)
    
    # Evaluate quality
    if has_nulls or zero_variance:
        state["time_series_quality_passed"] = False
        state["execution_logs"].append("Time-Series Engineer: QC FAILED (Missing values or flatline sensor detected).")
    else:
        state["time_series_quality_passed"] = True
        state["execution_logs"].append("Time-Series Engineer: QC PASSED. Normalized feature vectors generated.")
    
    # Select our clean input features
    feature_cols = ["pm25_3d_avg", "spo2_3d_avg", "cough_3d_avg", "exposure_risk_index"]
    
    # Store matrix and target in state as pure Python lists for serialization
    state["engineered_feature_matrix"] = df[feature_cols].values.tolist()
    state["engineered_target_vector"] = df["target"].values.tolist()
    
    return state