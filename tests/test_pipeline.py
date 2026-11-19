import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import compile_pediatric_mas_graph  # noqa: E402
from modules.adaptive_learner import train_and_predict_acuteness  # noqa: E402
from modules.synthetic_data import generate_synthetic_patient_records  # noqa: E402
from modules.time_series_engineer import engineer_time_series_features  # noqa: E402

LOW_RISK_CHECKIN = {
    "ambient_aqi": 30.0, "outdoor_pm25": 8.0, "outdoor_pollen_index": 1.0,
    "home_humidity_percent": 40.0, "home_indoor_mold_risk": False,
    "home_pet_dander_present": False, "parent_reported_cough_frequency": 1,
    "parent_reported_night_waking": False, "medication_adherence_percent": 1.0,
    "child_stress_activity_level": "low",
}


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("run"))
    path = os.path.abspath(generate_synthetic_patient_records(days=120))
    os.chdir(cwd)
    return path


def base_state(path, checkin=None):
    state = {
        "patient_id": "TEST", "age_years": 8,
        "primary_diagnosis": "Pediatric Asthma",
        "secondary_conditions": ["Dust Mite Allergy"],
        "baseline_fev1_percent": 82.0, "baseline_spo2": 97.5,
        "unstructured_medical_notes": "dust smoke exercise",
        "raw_history_path": path, "model_weights": [], "model_bias": 0.0,
        "execution_logs": [],
    }
    if checkin is not None:
        state["daily_checkin"] = checkin
    return state


def test_target_is_next_day_flare(data_path):
    s = engineer_time_series_features(base_state(data_path))
    raw = pd.read_csv(data_path)["flare_event"].tolist()
    assert s["engineered_target_vector"] == raw[1:]
    assert len(s["engineered_feature_matrix"]) == len(s["engineered_target_vector"]) + 1


def test_always_negative_model_is_not_validated(data_path):
    """A model that never predicts a flare must never pass the validation gate."""
    s = engineer_time_series_features(base_state(data_path))
    s["model_weights"] = [0.0] * 4
    s["model_bias"] = -50.0  # pushes every prediction to ~0
    import modules.adaptive_learner as al
    old_lr, al.LEARNING_RATE = al.LEARNING_RATE, 0.0
    try:
        s = train_and_predict_acuteness(s)
    finally:
        al.LEARNING_RATE = old_lr
    assert s["model_recall"] == 0.0
    assert s["is_model_calibrated"] is False
    assert s["rolling_accuracy_score"] >= 0.85  # high accuracy, yet correctly rejected


def test_parent_message_depends_on_risk(data_path):
    app = compile_pediatric_mas_graph()
    high = app.invoke(base_state(data_path))  # demo check-in is high risk
    low = app.invoke(base_state(data_path, LOW_RISK_CHECKIN))
    assert high["risk_tier"] == "high"
    assert low["risk_tier"] == "low"
    assert high["parent_distilled_action_plan"] != low["parent_distilled_action_plan"]
    for out in (high, low):
        plan = out["parent_distilled_action_plan"].lower()
        assert "emergency" in plan
        assert "step-up" not in plan  # no unsupervised dose-change advice


def test_flatline_sensor_halts_pipeline(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30),
        "pm25": np.linspace(10, 40, 30), "spo2": [98.0] * 30,
        "cough_score": [1] * 30, "med_adherence": [1.0] * 30, "flare_event": [0] * 30,
    })
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    out = compile_pediatric_mas_graph().invoke(base_state(str(path)))
    assert "unable to read sensor data" in out["parent_distilled_action_plan"]
