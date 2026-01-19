import streamlit as st
import pandas as pd
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    layout="centered"
)

st.title("🍔 Food Delivery Time Prediction")

# ----------------------------------
# Load model & encoders
# ----------------------------------
@st.cache_resource
def load_model():
    with open("optimized_rf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
label_encoders = load_label_encoders()

# ----------------------------------
# ✅ USE TRAINING FEATURES ONLY
# ----------------------------------
FEATURES = list(model.feature_names_in_)   # <-- MOST IMPORTANT LINE

st.subheader("📋 Enter Order Details")

input_data = {}

for col in FEATURES:
    if col in label_encoders:  # categorical
        input_data[col] = st.selectbox(
            col,
            label_encoders[col].classes_
        )
    else:  # numerical
        input_data[col] = st.number_input(
            col,
            value=0.0
        )

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🚀 Predict Delivery Time"):
    input_df = pd.DataFrame([input_data])

    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # ✅ FORCE exact order & columns
    input_df = input_df[FEATURES]

    prediction = model.predict(input_df)[0]

    st.success(f"⏱️ Estimated Delivery Time: **{round(prediction, 2)} minutes**")