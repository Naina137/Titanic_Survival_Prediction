import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page title
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "Q", "C"]
)

# Convert categorical values to the same format used during training
sex_male = 1 if sex == "Male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Prediction button
if st.button("Predict Survival"):

    input_data = pd.DataFrame([[
        pclass,
        age,
        sibsp,
        parch,
        fare,
        sex_male,
        embarked_q,
        embarked_s
    ]], columns=[
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex_male",
        "Embarked_Q",
        "Embarked_S"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(" Passenger is predicted to SURVIVE.")
    else:
        st.error("Passenger is predicted NOT to survive.")