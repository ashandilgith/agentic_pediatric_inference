# Import the shared state schema
from core.state import PediatricHealthState

# Demo values used only when no real check-in is supplied. Swap this for real
# environmental APIs (e.g. AirNow / OpenAQ) and the parent check-in form.
DEMO_CHECKIN = {
    "ambient_aqi": 115.0,                 # Unhealthy for sensitive groups
    "outdoor_pm25": 48.5,                 # Elevated PM2.5 particulate count
    "outdoor_pollen_index": 6.2,          # Moderate-High pollen
    "home_humidity_percent": 68.0,        # High humidity can promote dust mites
    "home_indoor_mold_risk": False,
    "home_pet_dander_present": True,      # Household cat present
    "parent_reported_cough_frequency": 4, # 4/5 persistent coughing
    "parent_reported_night_waking": True, # Child woke up coughing at 2:00 AM
    "medication_adherence_percent": 0.50, # Skipped morning controller dose
    "child_stress_activity_level": "moderate",
}


def gather_contextual_factors(state: PediatricHealthState) -> PediatricHealthState:
    """Aggregates household, environmental, and parent interview signals."""
    state["execution_logs"].append("Context Gathering: Aggregating environmental APIs and parent check-in.")

    checkin = state.get("daily_checkin") or {}
    source = "parent check-in" if checkin else "DEMO values (no check-in supplied)"
    for key, demo_value in DEMO_CHECKIN.items():
        state[key] = checkin.get(key, demo_value)

    state["execution_logs"].append(
        f"Context Gathering: Using {source}. PM2.5 {state['outdoor_pm25']} ug/m3, "
        f"cough {state['parent_reported_cough_frequency']}/5, night waking: {state['parent_reported_night_waking']}."
    )
    return state
