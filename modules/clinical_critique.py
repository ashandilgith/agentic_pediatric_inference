# Import shared state
from core.state import PediatricHealthState

TIERS = ["low", "elevated", "high"]


def critique_clinical_draft(state: PediatricHealthState) -> PediatricHealthState:
    """Adversarially cross-checks the draft and can escalate (never lower) the risk tier."""
    state["execution_logs"].append("Critique Agent: Cross-examining draft against patient history and missed risks.")

    adherence = state["medication_adherence_percent"]
    humidity = state["home_humidity_percent"]
    diagnoses = state["secondary_conditions"]
    points = []

    if adherence <= 0.50:
        points.append(
            f"CRITIQUE - ADHERENCE DEFICIT: Only {adherence*100:.0f}% of controller doses taken. "
            "The draft may underestimate airway inflammation compounding with environmental triggers."
        )
    if state["parent_reported_night_waking"]:
        points.append(
            "CRITIQUE - NOCTURNAL SYMPTOMS: Night waking from cough is a recognised sign of poor asthma control."
        )
    has_allergy = any("dust" in d.lower() or "allergy" in d.lower() for d in diagnoses)
    if state["home_pet_dander_present"] and has_allergy:
        detail = f" Indoor humidity is {humidity:.0f}%, which also favours dust mites." if humidity >= 60 else ""
        points.append("CRITIQUE - ALLERGEN EXPOSURE: Known allergies with a pet in the home." + detail)

    # Each counterpoint escalates the tier by one step; critique never downgrades risk.
    current = TIERS.index(state.get("risk_tier", "low"))
    new_tier = TIERS[min(current + (1 if len(points) >= 1 else 0) + (1 if len(points) >= 3 else 0), 2)]
    if new_tier != state.get("risk_tier"):
        points.append(f"CRITIQUE - ESCALATION: Risk tier raised from {state.get('risk_tier', 'low').upper()} to {new_tier.upper()}.")
    state["risk_tier"] = new_tier

    state["critique_pushback_notes"] = "\n".join(points) if points else \
        "CRITIQUE: No additional risk factors found beyond the initial draft."
    state["execution_logs"].append(f"Critique Agent: Raised {len(points)} counterpoints. Final tier: {new_tier.upper()}.")
    return state
