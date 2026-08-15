# Import shared state
from core.state import PediatricHealthState

def synthesize_and_translate_for_parents(state: PediatricHealthState) -> PediatricHealthState:
    """Reconciles clinical critique and translates medical analysis into clear parent guidance."""
    # Log reconciliation step
    state["execution_logs"].append("Final Orchestrator: Reconciling critique and drafting parent action plan.")
    
    # 1. Synthesize Technical Clinical Observation
    technical_summary = (
        f"FINAL CLINICAL CONSENSUS:\n"
        f"Child is entering a PRE-ACUTE FLARE STAGE.\n"
        f"Initial assessment risk confirmed, elevated further by nocturnal cough and 50% controller missed dose.\n"
        f"Recommended Action: Step-up inhaled corticosteroid (ICS) protocol per Asthma Action Plan, "
        f"isolate child from pet dander, and initiate air filtration."
    )
    state["final_clinical_observation"] = technical_summary
    
    # 2. Distill into Empathetic, Non-Jargon Parent Language
    parent_guide = (
        f"💙 DAILY HEALTH UPDATE FOR PARENTS\n"
        f"--------------------------------------------------\n"
        f"Hello! Here is how your child is doing today and what to keep an eye on:\n\n"
        f"⚠️ Status Today: Elevated Risk of an Asthma Flare-Up\n\n"
        f"Why we are flagging this:\n"
        f"• Outdoor air quality is poor today (dust and pollution are elevated).\n"
        f"• The night-time coughing reported during check-in means their lungs are feeling extra sensitive.\n"
        f"• We noticed the morning controller inhaler was missed.\n\n"
        f"📋 Action Steps for Today:\n"
        f"1. Make sure your child takes their prescribed controller inhaler right away.\n"
        f"2. Keep windows closed and run the indoor air purifier if available.\n"
        f"3. Keep rescue inhaler (Albuterol) accessible in their backpack or room.\n"
        f"4. Limit heavy outdoor running until the air clears up.\n\n"
        f"📞 When to Call the Clinic:\n"
        f"If coughing continues despite using the rescue inhaler, or if you notice their chest pulling in when breathing, "
        f"call your pediatrician or visit urgent care immediately."
    )
    
    # Save parent translation to state
    state["parent_distilled_action_plan"] = parent_guide
    state["execution_logs"].append("Final Orchestrator: Parent guidance generated.")
    return state