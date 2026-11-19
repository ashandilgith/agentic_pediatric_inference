# Import numpy for vector math and sigmoid activations
import numpy as np
# Import shared state
from core.state import PediatricHealthState

# --- Validation gate --------------------------------------------------------
# Flares are rare (roughly 5-10% of days), so plain accuracy is misleading: a model
# that always says "no flare" scores ~93% while catching nothing. The gate therefore
# uses balanced accuracy and recall, and needs enough flare examples to judge them.
MIN_FLARE_EVENTS = 5
MIN_BALANCED_ACCURACY = 0.75
MIN_RECALL = 0.60
ALERT_THRESHOLD = 0.5
LEARNING_RATE = 0.08


def sigmoid(z: float) -> float:
    """Applies standard logistic sigmoid function to bound predictions between 0 and 1."""
    z_clipped = np.clip(z, -20.0, 20.0)
    return float(1.0 / (1.0 + np.exp(-z_clipped)))


def train_and_predict_acuteness(state: PediatricHealthState) -> PediatricHealthState:
    """Online logistic regression with prequential (predict-then-learn) evaluation.

    Each historical day is first predicted with the weights learned from earlier days
    only, then used to update the weights. Metrics are computed from those
    out-of-sample predictions, so they estimate how the model would have performed live.
    """
    state["execution_logs"].append("Adaptive Learner: Starting online learning cycle over historical patient samples.")

    X_all = np.array(state["engineered_feature_matrix"], dtype=float)  # (N+1, F); last row = today
    y = np.array(state["engineered_target_vector"], dtype=int)          # (N,) next-day flare labels
    X_hist, x_today = X_all[: len(y)], X_all[-1]

    # Normalise with historical statistics only (today's row is not used to fit them)
    mean = X_hist.mean(axis=0)
    std = X_hist.std(axis=0) + 1e-7
    X_norm = (X_hist - mean) / std
    today_norm = (x_today - mean) / std

    num_features = X_hist.shape[1]
    weights = np.array(state.get("model_weights") or [0.0] * num_features, dtype=float)
    bias = float(state.get("model_bias") or 0.0)

    # Up-weight the rare positive class so the model is not rewarded for ignoring flares
    n_pos = int(y.sum())
    pos_weight = (len(y) - n_pos) / max(n_pos, 1)

    preds = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        prob = sigmoid(np.dot(weights, X_norm[i]) + bias)
        preds[i] = int(prob >= ALERT_THRESHOLD)
        # Standard SGD step on every sample (not just on errors), weighted by class
        sample_weight = pos_weight if y[i] == 1 else 1.0
        grad = sample_weight * (prob - y[i])
        weights -= LEARNING_RATE * grad * X_norm[i]
        bias -= LEARNING_RATE * grad

    # Skip a short warm-up period when scoring, since the model starts untrained
    warmup = min(14, len(y) // 4)
    p, t = preds[warmup:], y[warmup:]
    tp = int(((p == 1) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    accuracy = (tp + tn) / max(len(t), 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    balanced_acc = (recall + specificity) / 2
    baseline_acc = 1.0 - (t.mean() if len(t) else 0.0)

    is_calibrated = bool(
        (tp + fn) >= MIN_FLARE_EVENTS
        and balanced_acc >= MIN_BALANCED_ACCURACY
        and recall >= MIN_RECALL
    )

    # Forecast tomorrow from today's features
    today_prob = sigmoid(np.dot(weights, today_norm) + bias)
    acuteness_score = round(1.0 + today_prob * 9.0, 2)

    state["model_weights"] = weights.tolist()
    state["model_bias"] = bias
    state["training_iteration_count"] = len(y)
    state["rolling_accuracy_score"] = round(accuracy, 4)
    state["model_recall"] = round(recall, 4)
    state["model_precision"] = round(precision, 4)
    state["model_balanced_accuracy"] = round(balanced_acc, 4)
    state["baseline_accuracy"] = round(float(baseline_acc), 4)
    state["flare_events_observed"] = int(tp + fn)
    state["is_model_calibrated"] = is_calibrated
    state["today_flare_probability"] = round(today_prob, 4)
    state["model_acuteness_score"] = acuteness_score
    state["model_binary_prediction"] = int(today_prob >= ALERT_THRESHOLD)

    status = "VALIDATED" if is_calibrated else "NOT VALIDATED (rule-based fallback will be used)"
    state["execution_logs"].append(
        f"Adaptive Learner: Out-of-sample accuracy {accuracy*100:.1f}% (always-'no flare' baseline "
        f"{baseline_acc*100:.1f}%), flare recall {recall*100:.0f}% ({tp}/{tp+fn}), precision "
        f"{precision*100:.0f}%, balanced accuracy {balanced_acc*100:.1f}%. Status: {status}."
    )
    return state
