import streamlit as st
import pandas as pd
import requests

import os
import streamlit as st
import pandas as pd
import requests

# Fetch the URL from Docker environment variables. 
# If not found (running locally), it defaults to your localhost.
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000/v1/predict")

st.set_page_config(page_title="Cancer Risk Predictor", layout="centered")

# The URL where your FastAPI server is listening
FASTAPI_URL = "http://127.0.0.1:8000/v1/predict"

# We define the features here exactly as they appear in the FastAPI Pydantic model
FEATURE_NAMES = [
    "Age", "Gender", "BMI", "Smoking", "Alcohol_Use", "Obesity",
    "Diet_Red_Meat", "Diet_Salted_Processed", "Fruit_Veg_Intake",
    "Physical_Activity", "Physical_Activity_Level", "Air_Pollution",
    "Occupational_Hazards", "Calcium_Intake", "Family_History",
    "BRCA_Mutation", "H_Pylori_Infection"
]

st.title("Cancer Risk Level Predictor")
st.markdown("Predict `Risk_Level` (Low / Medium / High) via FastAPI backend.")

option = st.radio("Prediction mode", ("Manual input (single)", "Upload CSV (batch)"))

if option == "Manual input (single)":
    st.sidebar.header("Patient features")
    input_data = {}
    
    # Generate UI inputs for each feature
    for feat in FEATURE_NAMES:
        if feat in ["BMI"]:
            input_data[feat] = st.sidebar.number_input(feat, value=25.0)
        else:
            input_data[feat] = st.sidebar.number_input(feat, value=0, step=1)

    if st.sidebar.button("Predict"):
        # 1. Send the HTTP POST request to FastAPI
        with st.spinner("Connecting to FastAPI Brain..."):
            try:
                response = requests.post(FASTAPI_URL, json=input_data)
                
                # Check if FastAPI rejected the data (422) or had an internal error (500)
                if response.status_code != 200:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    st.stop() # Stops the rest of the code from running
                
                # 2. Parse the JSON response
                result = response.json()
                
                # 3. Display results
                st.write("### Prediction")
                st.write(f"**Predicted Risk_Level:** {result['risk_level']}")
                st.write("**Class probabilities:**")
                
                prob_df = pd.DataFrame(
                    list(result['probabilities'].items()), 
                    columns=['class', 'probability']
                ).sort_values('probability', ascending=False).reset_index(drop=True)
                
                st.table(prob_df)
                
            except requests.exceptions.ConnectionError:
                st.error("Error: Could not connect to the backend. Is FastAPI (main.py) running on port 8000?")
            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    # Upload CSV (batch) Mode
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type=['csv'])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        # Verify required columns exist
        missing = [c for c in FEATURE_NAMES if c not in input_df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            if st.button("Run Batch Prediction"):
                results_list = []
                progress_bar = st.progress(0)
                
                # Loop through each row and send to our single-prediction endpoint
                # We use enumerate() to create a safe 'step' counter (0, 1, 2...)
            for step, (index, row) in enumerate(input_df.iterrows()):
                payload = {feat: row[feat] for feat in FEATURE_NAMES}
                
                try:
                    resp = requests.post(FASTAPI_URL, json=payload)
                    if resp.status_code == 200:
                        api_data = resp.json()
                        row_result = payload.copy()
                        row_result['Predicted_Risk_Level'] = api_data['risk_level']
                        
                        # Add probabilities to the row
                        for cls, prob in api_data['probabilities'].items():
                            row_result[f'prob_{cls}'] = prob
                            
                        results_list.append(row_result)
                    else:
                        st.warning(f"Row {index} failed with status {resp.status_code}")
                except Exception:
                    st.warning(f"Failed to connect on row {index}")
                    
                # THIS IS FIXED: Aligned with try/except, and uses 'step' instead of 'index'
                progress_bar.progress((step + 1) / len(input_df))
                
                # Show final table
                if results_list:
                    st.success("Batch Predictions complete!")
                    final_df = pd.DataFrame(results_list)
                    st.dataframe(final_df)
                    st.download_button("Download results (CSV)", final_df.to_csv(index=False), file_name="batch_predictions.csv", mime="text/csv")