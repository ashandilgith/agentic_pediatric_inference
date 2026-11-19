# Import shared state
from core.state import PediatricHealthState

STATUS_LINE = {
    "low": "✅ Status Today: Things look stable",
    "elevated": "⚠️ Status Today: Somewhat higher risk of a flare-up",
    "high": "🔶 Status Today: Higher risk of a flare-up",
}

EMERGENCY_NOTE = (
    "🚨 Get emergency care now (call your local emergency number) if your child is struggling to breathe, "
    "can't speak in full sentences, has blue or grey lips, or the rescue inhaler isn't helping.\n"
    "This update is an early-warning aid, not medical advice. Always follow the asthma action plan "
    "your child's doctor gave you."
)


def _reasons(state: PediatricHealthState) -> list:
    reasons = []
    if state["outdoor_pm25"] > state.get("vital_thresholds", {}).get("extreme_pm25_exposure", 45.0):
        reasons.append("Outdoor air pollution is high today.")
    if state["outdoor_pollen_index"] >= 6:
        reasons.append("Pollen levels are high.")
    if state["parent_reported_night_waking"]:
        reasons.append("You reported night-time coughing, which can mean the airways are irritated.")
    if state["parent_reported_cough_frequency"] >= 4:
        reasons.append("Coughing has been frequent today.")
    if state["medication_adherence_percent"] < 1.0:
        reasons.append("Some controller doses were missed recently.")
    if state["home_pet_dander_present"]:
        reasons.append("Pet allergens at home may add to symptoms.")
    return reasons


def synthesize_and_translate_for_parents(state: PediatricHealthState) -> PediatricHealthState:
    """Reconciles the critique and writes guidance whose content depends on the risk tier.

    Guidance deliberately does not tell parents to change medication doses: that is the
    job of the child's written asthma action plan and their clinician.
    """
    state["execution_logs"].append("Final Orchestrator: Reconciling critique and drafting parent action plan.")
    tier = state.get("risk_tier", "low")
    reasons = _reasons(state)

    state["final_clinical_observation"] = (
        f"FINAL CONSENSUS: risk tier {tier.upper()}.\n"
        f"Basis: {'validated model' if state['is_model_calibrated'] else 'rule-based fallback'}; "
        f"model P(flare tomorrow) = {state['today_flare_probability']*100:.0f}%.\n"
        f"Critique:\n{state['critique_pushback_notes']}\n"
        f"Suggested clinician review: {'yes' if tier == 'high' else 'routine'}."
    )

    steps = []
    if tier in ("elevated", "high"):
        steps.append("Follow the 'yellow zone' steps on your child's asthma action plan if they have symptoms.")
        if state["medication_adherence_percent"] < 1.0:
            steps.append("Give the next controller dose on schedule as prescribed. Don't double up unless your doctor told you to.")
        steps.append("Keep the rescue inhaler with your child today.")
        if state["outdoor_pm25"] > 35:
            steps.append("Keep windows closed and limit hard outdoor exercise while air quality is poor.")
    if tier == "high":
        steps.append("Consider calling your child's doctor or nurse line today to talk about these warning signs.")
    if tier == "low":
        steps.append("Keep up the usual routine and controller medicine as prescribed.")

    lines = ["💙 DAILY HEALTH UPDATE", "-" * 50, STATUS_LINE[tier], ""]
    if reasons:
        lines += ["Why:"] + [f"• {r}" for r in reasons] + [""]
    lines += ["📋 Suggested steps:"] + [f"{i}. {s}" for i, s in enumerate(steps, 1)] + ["", EMERGENCY_NOTE]

    state["parent_distilled_action_plan"] = "\n".join(lines)
    state["execution_logs"].append(f"Final Orchestrator: Parent guidance generated for tier {tier.upper()}.")
    return state
