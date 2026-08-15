# Import os to manage file directories
import os
# Import numpy for statistical distributions and matrix operations
import numpy as np
# Import pandas to create and save tabular clinical records
import pandas as pd

def generate_synthetic_patient_records(patient_id: str = "PED-8821", days: int = 120) -> str:
    """Generates 120 days of correlated clinical, environmental, and behavioral data."""
    # Set seed for reproducible experiments
    np.random.seed(42)
    
    # Generate daily chronological timestamps ending today
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    # Simulate continuous outdoor PM2.5 air pollution with occasional spikes
    pm25 = np.clip(np.random.normal(loc=25.0, scale=12.0, size=days), 5.0, 150.0)
    
    # Simulate resting blood oxygen saturation (SpO2) normally around 98%
    spo2 = np.clip(np.random.normal(loc=98.0, scale=1.5, size=days), 88.0, 100.0)
    
    # Simulate daily cough severity reported by parents (integer 1 to 5)
    cough_scores = np.random.choice([1, 2, 3, 4, 5], size=days, p=[0.50, 0.25, 0.15, 0.07, 0.03])
    
    # Simulate medication adherence percentage (0.0 to 1.0)
    adherence = np.random.choice([1.0, 0.75, 0.50, 0.25], size=days, p=[0.70, 0.15, 0.10, 0.05])
    
    # Array to hold ground-truth next-day flare-ups (Target variable)
    flare_ups = np.zeros(days, dtype=int)
    
    # Introduce deterministic clinical logic: High pollution + dropping SpO2 + low adherence = Flare
    for i in range(days - 1):
        risk_index = (pm25[i] / 50.0) + (cough_scores[i] * 0.5) + ((100 - spo2[i]) * 0.4) - (adherence[i] * 1.5)
        # If risk exceeds clinical threshold, a flare occurs next day
        if risk_index > 2.8:
            flare_ups[i + 1] = 1
            # Next day SpO2 drops further during the flare
            spo2[i + 1] = np.clip(spo2[i + 1] - 4.0, 85.0, 95.0)
    
    # Construct complete tabular dataframe
    df = pd.DataFrame({
        "date": date_range,
        "pm25": pm25,
        "spo2": spo2,
        "cough_score": cough_scores,
        "med_adherence": adherence,
        "flare_event": flare_ups
    })
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    # Define file path
    filepath = f"data/{patient_id}_longitudinal.csv"
    # Save to disk
    df.to_csv(filepath, index=False)
    # Return path to stored data
    return filepath