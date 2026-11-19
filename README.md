# Pediatric Predictive Multi-Agent System (MAS)

> **Status: research prototype.** Runs on synthetic data only. It has not been clinically validated and must not be used to make care decisions for a real child.

An agentic machine-learning architecture designed to predict, intervene, and reduce pediatric hospitalization rates for chronic illnesses. 

This system orchestrates a network of specialized, interacting AI agents to ingest longitudinal medical records, track environmental data, and apply adaptive time-series forecasting. The result is a highly vetted, daily risk assessment distilled into actionable guidance for parents.

---

##  For Developers & Engineers: Technical Overview

This repository demonstrates the orchestration of a stateful, pure-Python Multi-Agent System without relying solely on black-box LLMs. 

### Core Architecture
*   **Orchestration Framework:** Built on **LangGraph**. The system leverages a high-dimensional `TypedDict` state, allowing deterministic routing and strict quality-control gates between agents.
*   **Data Engineering Agent:** Utilizes **Pandas** to transform raw, asynchronous sensor and environmental data into structured, rolling-window feature matrices suitable for predictive modeling.
*   **Adaptive Learner Agent:** Implements an online, continuous-learning loop using **NumPy**. The agent performs sample-by-sample gradient updates and tracks a rolling accuracy window. Every historical day is predicted *before* the model learns from it (prequential evaluation), so the reported metrics are out-of-sample. A safety gate only trusts the model when it has seen at least 5 flares, catches ≥60% of them (recall) and reaches ≥75% balanced accuracy. Otherwise the system falls back to rule-based thresholds. (Plain accuracy is not used for the gate. Flares occur on only ~7% of days, so a model that never predicts one would score ~93%.)
*   **Adversarial Critique Agent:** An expert rule-based node designed to mitigate confirmation bias. It cross-examines the initial technical draft against secondary conditions and medication adherence gaps, and can only *raise* the risk tier, never lower it.
*   **Frontend UI:** Deployed via **Streamlit**, providing a responsive dashboard to visualize time-series anomaly detection, model weight updates, and execution logs.

### Why this approach?
Rather than using a single, monolithic LLM prompt to "guess" health outcomes, this architecture separates concerns. Quantitative tasks (rolling averages, statistical variance checks, gradient descent) are handled by mathematical modules, while qualitative tasks (context gathering, parent distillation) are routed to semantic reasoners. This makes the system auditable and deterministic. Note: the current version contains no LLM; the "semantic" nodes are rule-based placeholders.

---

## Disease-Agnostic Architecture (Extensibility)

While this repository uses Pediatric Asthma as its primary demonstration, **the underlying Multi-Agent System is entirely disease-agnostic.** It can be rapidly reconfigured to predict hospitalizations for almost any chronic pediatric condition, including Type 1 Diabetes, Epilepsy, Cystic Fibrosis, or Sickle Cell Anemia.

### Why it works for any illness:
The core LangGraph orchestrator and mathematical agents (Time-Series Engineer, Adaptive Learner) do not rely on hard-coded medical logic. They process abstracted numerical matrices. A sudden drop in blood oxygen (Asthma) is mathematically identical to a sudden drop in blood glucose (Diabetes) or an isolated spike in body temperature (Infection risk in Sickle Cell)—they are all time-series anomalies that trigger the gradient update loop.

### How to adapt the system for a new condition:
Because the architecture is strictly modular, transitioning to a new illness requires no changes to the core orchestration (`graph.py`). You need to swap the context in the outer nodes and the feature columns:
1. **The Ingestion Agent (The Schema):** Update the expected variables. *(e.g., For Diabetes, swap SpO2 and PM2.5 for Continuous Glucose Monitor readings and insulin doses).*
2. **The Context Gathering Agent:** Update the qualitative inputs. *(e.g., For Epilepsy, swap "cough frequency" for "hours of uninterrupted sleep").*
3. **The Time-Series Engineer:** Update `feature_cols` and the raw column names it reads.
4. **The Final Orchestrator:** Update the text template to provide disease-specific instructions to parents based on the new risk profile.

---

## For Hospitals & Healthcare Insurers: System Value

Chronic pediatric hospitalizations are a massive driver of avoidable healthcare expenditure and bed utilization. This system is designed as a scalable preventative intervention layer.

### The Value Proposition
*   **Proactive vs. Reactive Care:** The system identifies the *pre-acute* stage of a flare-up. The goal is to forecast exacerbations a day in advance (the current model predicts next-day flares), so that clinicians can recommend minor, low-cost interventions rather than managing a high-cost Emergency Department admission.
*   **Holistic Data Integration:** Clinical records are static; chronic illnesses are dynamic. This system correlates static EHR baselines with dynamic environmental APIs (local PM2.5, pollen counts) and daily behavioral markers (medication adherence, sleep quality).
*   **Automated Quality Control:** The system includes automated variance checks to detect sensor failures or flatlines, ensuring alerts sent to clinical triage teams are high-confidence, minimizing alarm fatigue.
*   **Scalable Patient Monitoring:** A single orchestrator can run daily inference over thousands of patient time-series streams simultaneously, flagging only the highest-risk cohorts for human clinical review.

---

## For Parents: How It Works & What to Expect

Managing a child's chronic illness can feel overwhelming. This system is built to act as a supportive, vigilant assistant that helps you stay one step ahead of flare-ups.

### How It Helps
The dashboard looks at everything affecting your child's health today—how they slept, if they missed a medication dose, and even the air pollution or pollen levels outside your home. It uses past data to learn your child’s unique triggers and warns you *before* a flare-up happens.

### How to Use It
1.  **Daily Check-In:** You will provide quick updates on your child’s sleep, cough frequency, or activity levels.
2.  **Read the Action Plan:** The system will provide a simple, daily summary. It might say: *"Air quality is poor today and a controller dose was missed. Keep the rescue inhaler with your child and follow the yellow-zone steps of their asthma action plan."* The system never tells you to change a medication dose; that comes from your child's doctor.
3.  **Share with your Doctor:** You can show the dashboard's charts to your pediatrician so they can see exactly what triggers your child’s symptoms over time.

###  Important Limitations (Please Read)
*   **Not a Replacement for Doctors:** This tool is an "early warning system," not a medical professional. It **cannot** diagnose conditions or prescribe medications.
*   **Always Follow the Action Plan:** If your child is struggling to breathe, showing distress, or experiencing severe symptoms, **do not wait for the dashboard to update.** Follow the emergency action plan provided by your doctor and seek immediate medical care.
*   **Data Dependent:** The system is only as good as the information it receives. If a wearable sensor is uncharged or a daily check-in is skipped, the system's ability to predict a flare-up is reduced.

---

## Getting Started

```bash
pip install -r requirements.txt
python main.py              # run the pipeline in the terminal
streamlit run app_v2.py     # interactive dashboard (app.py is the simpler original)
python -m pytest -q tests   # run the test suite
```

To supply a real daily check-in instead of the demo values, add a `daily_checkin` dict to the initial state (see `modules/context_gathering.py` for the keys).

## Model evaluation (synthetic PED-8821, 120 days)

| Metric (out-of-sample) | Value |
|---|---|
| Always-"no flare" baseline accuracy | 93.3% |
| Model accuracy | 76.2% |
| Flare recall | 71.4% |
| Alert precision | 17.9% |
| Gate status | Not validated → rule-based fallback |

The model catches most flares but raises about 4 false alarms for each real one, so the gate correctly declines to trust it yet. More history, better features and threshold tuning are the next steps.
