# Import LangGraph StateGraph engine and terminal END state
from langgraph.graph import StateGraph, END
# Import the shared state
from core.state import PediatricHealthState

# Import all agent nodes
from modules.clinical_ingestion import ingest_medical_records
from modules.context_gathering import gather_contextual_factors
from modules.time_series_engineer import engineer_time_series_features
from modules.adaptive_learner import train_and_predict_acuteness
from modules.initial_orchestrator import draft_initial_clinical_prediction
from modules.clinical_critique import critique_clinical_draft
from modules.final_orchestrator import synthesize_and_translate_for_parents

def check_time_series_quality_edge(state: PediatricHealthState) -> str:
    """Conditional routing function checking if time-series data passed QC."""
    # Check boolean flag set by time_series_engineer
    if state["time_series_quality_passed"]:
        return "adaptive_learner"
    else:
        return "flag_bad_data"

def handle_data_quality_error(state: PediatricHealthState) -> PediatricHealthState:
    """Fallback node triggered when sensor or timeseries data fails quality verification."""
    state["execution_logs"].append("CRITICAL: Pipeline halted due to invalid time series or sensor flatline.")
    state["parent_distilled_action_plan"] = (
        "⚠️ System Alert: We were unable to read sensor data accurately today. "
        "Please verify your child's monitor connection."
    )
    return state

def compile_pediatric_mas_graph():
    """Assembles and compiles the full LangGraph workflow."""
    # 1. Initialize Graph with shared state
    workflow = StateGraph(PediatricHealthState)
    
    # 2. Add all Agent Nodes
    workflow.add_node("clinical_ingestion", ingest_medical_records)
    workflow.add_node("context_gathering", gather_contextual_factors)
    workflow.add_node("time_series_engineer", engineer_time_series_features)
    workflow.add_node("adaptive_learner", train_and_predict_acuteness)
    workflow.add_node("initial_orchestrator", draft_initial_clinical_prediction)
    workflow.add_node("clinical_critique", critique_clinical_draft)
    workflow.add_node("final_orchestrator", synthesize_and_translate_for_parents)
    workflow.add_node("flag_bad_data", handle_data_quality_error)
    
    # 3. Define Entry Point and Edges
    workflow.set_entry_point("clinical_ingestion")
    workflow.add_edge("clinical_ingestion", "context_gathering")
    workflow.add_edge("context_gathering", "time_series_engineer")
    
    # 4. Add Conditional Routing based on Time-Series QC
    workflow.add_conditional_edges(
        "time_series_engineer",
        check_time_series_quality_edge,
        {
            "adaptive_learner": "adaptive_learner",
            "flag_bad_data": "flag_bad_data"
        }
    )
    
    # 5. Connect downstream pipeline
    workflow.add_edge("adaptive_learner", "initial_orchestrator")
    workflow.add_edge("initial_orchestrator", "clinical_critique")
    workflow.add_edge("clinical_critique", "final_orchestrator")
    workflow.add_edge("final_orchestrator", END)
    workflow.add_edge("flag_bad_data", END)
    
    # 6. Compile executable LangGraph application
    return workflow.compile()