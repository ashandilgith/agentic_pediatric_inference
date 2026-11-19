# Import shared state
from core.state import PediatricHealthState

TIERS = ["low", "elevated", "high"]


def draft_initial_clinical_prediction(state: PediatricHealthState) -> PediatricHealthState:
    """Drafts an initial technical assessment and sets a preliminary risk tier."""
    state["execution_logs"].append("Initial Orchestrator: Synthesizing metrics into preliminary assessment.")

    calibrated = state["is_model_calibrated"]
    prob = state["today_flare_probability"]
    cough = state["parent_reported_cough_frequency"]
    pm25 = state["outdoor_pm25"]
    thresholds = state.get("vital_thresholds", {})
    pm25_limit = thresholds.get("extreme_pm25_exposure", 45.0)
    cough_limit = thresholds.get("warning_max_cough_score", 3.5)

    if calibrated:
        basis = (f"Validated model (recall {state['model_recall']*100:.0f}%, "
                 f"precision {state['model_precision']*100:.0f}%)")
        tier = "high" if prob >= 0.5 else "elevated" if prob >= 0.25 else "low"
    else:
        basis = (f"Rule-based fallback (model not validated: balanced accuracy "
                 f"{state['model_balanced_accuracy']*100:.0f}%, {state['flare_events_observed']} flares observed)")
        flags = int(cough > cough_limit) + int(pm25 > pm25_limit)
        tier = TIERS[min(flags, 2)]

    state["risk_tier"] = tier
    state["initial_clinical_draft"] = (
        f"[Preliminary Clinical Assessment]\n"
        f"Assessment Basis: {basis}\n"
        f"Model flare probability (tomorrow): {prob*100:.0f}% | Acuteness Index: {state['model_acuteness_score']}/10.0\n"
        f"Preliminary risk tier: {tier.upper()}\n"
        f"Key Contributing Factors: Outdoor PM2.5 at {pm25} ug/m3, Cough Severity: {cough}/5, "
        f"Baseline SpO2: {state['baseline_spo2']}%."
    )
    return state
