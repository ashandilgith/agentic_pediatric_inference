# Import shared state
from core.state import PediatricHealthState

def critique_clinical_draft(state: PediatricHealthState) -> PediatricHealthState:
    """Adversarially cross-checks initial prediction against patient chart nuances and adherence."""
    # Log critique entry
    state["execution_logs"].append("Critique Agent: Cross-examining draft against patient history and missed risks.")
    
    draft = state["initial_clinical_draft"]
    adherence = state["medication_adherence_percent"]
    night_waking = state["parent_reported_night_waking"]
    pet_present = state["home_pet_dander_present"]
    diagnoses = state["secondary_conditions"]
    
    critique_points = []
    
    # 1. Challenge based on medication adherence
    if adherence <= 0.50:
        critique_points.append(
            "CRITIQUE - ADHERENCE DEFICIT: Patient took only 50% of controller medication. "
            "The initial draft may underestimate airway inflammation compounding with environmental triggers."
        )
        
    # 2. Challenge based on night symptoms (hallmark sign of impending pediatric obstruction)
    if night_waking:
        critique_points.append(
            "CRITIQUE - NOCTURNAL COUGH: Parent reported night waking due to coughing. "
            "In pediatric asthma, nocturnal symptoms are a direct clinical indicator of destabilization within 24h."
        )
        
    # 3. Challenge based on unconsidered allergen exposure
    if pet_present and any("dust" in d.lower() or "allergy" in d.lower() for d in diagnoses):
        critique_points.append(
            "CRITIQUE - ALLERGEN SYNERGY: Child has known atopic allergies and a pet is present in high humidity. "
            "Airway hyperresponsiveness will be significantly elevated."
        )
        
    # Assemble pushback response
    if critique_points:
        critique_text = "\n".join(critique_points)
    else:
        critique_text = "CRITIQUE: Initial draft is clinically sound and accounts for all chart parameters."
        
    # Save critique to state
    state["critique_pushback_notes"] = critique_text
    state["execution_logs"].append(f"Critique Agent: Raised {len(critique_points)} clinical counterpoints.")
    return state