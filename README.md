🖥️ USER'S BROWSER (Streamlit Frontend - Port 8501)
       │
       ▼ (User enters patient data, clicks "Predict")
       │  HTTP POST Request (JSON payload)
       ▼
======================================================
⚙️ FASTAPI BACKEND (main.py - Port 8000)
======================================================
  1. 🛡️ Pydantic Input Validation (Rejects invalid numbers/types)
  2. 🔄 DataFrame Assembly (Orders the 17 features to match training data)
  3. 🧠 XGBoost Model Prediction (Runs optimized Optuna model from RAM)
  4. 🏷️ Response Casting (Converts numpy types to clean strings/floats)
       │
       ▼  HTTP 200 OK Response (JSON: {"risk_level": "High", "probabilities": {...}})
       │
🖥️ STREAMLIT UI (Renders the risk level and probability table on screen)
