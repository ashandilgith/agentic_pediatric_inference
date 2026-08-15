# Import numpy for vector math and sigmoid activations
import numpy as np
# Import shared state
from core.state import PediatricHealthState

def sigmoid(z: float) -> float:
    """Applies standard logistic sigmoid function to bound predictions between 0 and 1."""
    # Clip z to prevent numerical overflow in exponential function
    z_clipped = np.clip(z, -20.0, 20.0)
    # Return standard sigmoid formula
    return float(1.0 / (1.0 + np.exp(-z_clipped)))

def train_and_predict_acuteness(state: PediatricHealthState) -> PediatricHealthState:
    """Iteratively trains adaptive weights, adjusts on errors, and validates 85% accuracy threshold."""
    # Log training entry
    state["execution_logs"].append("Adaptive Learner: Starting online learning cycle over historical patient samples.")
    
    # Extract matrices from state
    X = np.array(state["engineered_feature_matrix"]) # Features: shape (N, 4)
    y = np.array(state["engineered_target_vector"])   # Binary targets: shape (N,)
    
    # Normalize features using z-score normalization for training stability
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7 # Prevent division by zero
    X_norm = (X - mean) / std
    
    # Initialize or load existing weights (4 features + 1 bias)
    num_features = X.shape[1]
    weights = np.array(state.get("model_weights") or [0.1] * num_features, dtype=float)
    bias = float(state.get("model_bias") or 0.0)
    learning_rate = 0.08
    
    # Perform online sample-by-sample training (Sequential trial and error)
    recent_predictions = []
    correct_count = 0
    total_samples = len(y)
    
    for i in range(total_samples):
        xi = X_norm[i]
        yi = y[i]
        
        # Calculate linear combination: z = w·x + b
        z = np.dot(weights, xi) + bias
        # Calculate continuous probability output via sigmoid
        prob = sigmoid(z)
        # Convert continuous probability to binary prediction (0 or 1)
        pred_binary = 1 if prob >= 0.5 else 0
        
        # Check if the prediction matched actual outcome
        is_correct = (pred_binary == yi)
        if is_correct:
            correct_count += 1
        else:
            # ERROR OCCURRED: Apply gradient update to correct error
            error = prob - yi
            # Update weights in opposite direction of error gradient
            weights -= learning_rate * error * xi
            # Update bias term
            bias -= learning_rate * error
            
        recent_predictions.append(1 if is_correct else 0)
    
    # Compute accuracy across the recent 40 samples (Rolling window accuracy)
    window_size = min(40, total_samples)
    rolling_acc = sum(recent_predictions[-window_size:]) / float(window_size)
    
    # --- Check 85% Accuracy Threshold ---
    is_calibrated = bool(rolling_acc >= 0.85)
    
    # Perform inference on today's latest data (the final row)
    today_vector = X_norm[-1]
    today_z = np.dot(weights, today_vector) + bias
    today_prob = sigmoid(today_z)
    
    # Scale probability into an intuitive acuteness score from 1.0 (calm) to 10.0 (critical emergency)
    acuteness_score = round(1.0 + (today_prob * 9.0), 2)
    binary_pred = 1 if today_prob >= 0.50 else 0
    
    # Save all learned parameters and calibration status back to state
    state["model_weights"] = weights.tolist()
    state["model_bias"] = bias
    state["training_iteration_count"] = total_samples
    state["rolling_accuracy_score"] = round(rolling_acc, 4)
    state["is_model_calibrated"] = is_calibrated
    state["model_acuteness_score"] = acuteness_score
    state["model_binary_prediction"] = binary_pred
    
    # Log results
    calibration_status = "READY (>=85%)" if is_calibrated else "NOT CALIBRATED (<85%)"
    state["execution_logs"].append(
        f"Adaptive Learner: Training completed. Accuracy: {rolling_acc*100:.1f}%. Status: {calibration_status}."
    )
    return state