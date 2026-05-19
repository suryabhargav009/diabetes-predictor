streamlit_code = '''
# app.py — Diabetes Prediction Streamlit App
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pickle

# Load model and scaler
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# App layout
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient details to predict the likelihood of diabetes.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies  = st.number_input("Pregnancies",         min_value=0,   max_value=20,   value=1)
    glucose      = st.number_input("Glucose (mg/dL)",     min_value=0,   max_value=300,  value=120)
    blood_press  = st.number_input("Blood Pressure (mmHg)",min_value=0,  max_value=200,  value=70)
    skin_thick   = st.number_input("Skin Thickness (mm)", min_value=0,   max_value=100,  value=20)

with col2:
    insulin      = st.number_input("Insulin (mu U/mL)",   min_value=0,   max_value=900,  value=80)
    bmi          = st.number_input("BMI",                 min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf          = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age          = st.number_input("Age",                 min_value=1,   max_value=120,  value=30)

st.divider()
if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_press, skin_thick,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes — Probability: {probability*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes — Probability: {probability*100:.1f}%")

    st.progress(float(probability))
    st.caption(f"Confidence: {probability*100:.1f}% probability of diabetes")
'''

# Save to file
with open('app.py', 'w') as f:
    f.write(streamlit_code)

print(streamlit_code)
print('\n✅ Streamlit app saved as app.py')
print('Run with: streamlit run app.py')