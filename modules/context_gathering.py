# Import the shared state schema
from core.state import PediatricHealthState

def gather_contextual_factors(state: PediatricHealthState) -> PediatricHealthState:
    """Aggregates household, environmental, and parent interview signals."""
    # Log the step
    state["execution_logs"].append("Context Gathering: Aggregating environmental APIs and parent check-in.")
    
    # Set simulated ambient parameters for the current day
    state["ambient_aqi"] = 115.0               # Unhealthy for sensitive groups
    state["outdoor_pm25"] = 48.5              # Elevated PM2.5 particulate count
    state["outdoor_pollen_index"] = 6.2        # Moderate-High pollen
    
    # Set simulated household indicators
    state["home_humidity_percent"] = 68.0      # High humidity can promote dust mites
    state["home_indoor_mold_risk"] = False     # No dampness/mold detected
    state["home_pet_dander_present"] = True    # Household cat present
    
    # Set simulated parent daily check-in responses
    state["parent_reported_cough_frequency"] = 4   # 4/5 severe persistent coughing
    state["parent_reported_night_waking"] = True   # Child woke up coughing at 2:00 AM
    state["medication_adherence_percent"] = 0.50   # Skipped morning controller dose
    state["child_stress_activity_level"] = "moderate"
    
    # Log summary
    state["execution_logs"].append(
        "Context Gathering: Captured elevated outdoor PM2.5 (48.5 ug/m3) and parent-reported night waking."
    )
    # Return updated state
    return state