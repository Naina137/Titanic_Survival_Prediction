import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("Titanic Survival Prediction")

st.write(
    "This project uses passenger information to understand "
    "Titanic survival patterns and provide a simple survival prediction."
)


# ---------------------------------------------------------
# About Titanic
# ---------------------------------------------------------

st.header("About the Titanic")

st.write(
    "The RMS Titanic was a British passenger liner operated by "
    "the White Star Line. During its maiden voyage in April 1912, "
    "the ship struck an iceberg in the North Atlantic Ocean and sank."
)

st.write(
    "The Titanic dataset is one of the most commonly used datasets "
    "for learning Data Science and Machine Learning. It contains "
    "information about passengers such as passenger class, sex, age, "
    "family members, fare and port of embarkation."
)


# ---------------------------------------------------------
# Dataset Information
# ---------------------------------------------------------

st.header("Dataset Information")

dataset_info = pd.DataFrame(
    {
        "Feature": [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Survived"
        ],
        "Description": [
            "Passenger class",
            "Passenger gender",
            "Passenger age",
            "Number of siblings or spouses",
            "Number of parents or children",
            "Passenger fare",
            "Port of embarkation",
            "Target variable"
        ]
    }
)

st.dataframe(
    dataset_info,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------

st.header("Exploratory Data Analysis")

st.write(
    "The following graphs show some important patterns "
    "in the Titanic passenger data."
)


# ---------------------------------------------------------
# Survival Distribution
# ---------------------------------------------------------

st.subheader("Survival Distribution")

survival_data = pd.DataFrame(
    {
        "Status": [
            "Did Not Survive",
            "Survived"
        ],
        "Passengers": [
            549,
            342
        ]
    }
)

fig1, ax1 = plt.subplots(figsize=(7, 4))

ax1.bar(
    survival_data["Status"],
    survival_data["Passengers"]
)

ax1.set_xlabel("Survival Status")
ax1.set_ylabel("Number of Passengers")
ax1.set_title("Survival Distribution")

plt.tight_layout()

st.pyplot(fig1)


# ---------------------------------------------------------
# Survival by Gender
# ---------------------------------------------------------

st.subheader("Survival by Gender")

gender_data = pd.DataFrame(
    {
        "Gender": ["Female", "Male"],
        "Survived": [233, 109],
        "Did Not Survive": [81, 468]
    }
)

fig2, ax2 = plt.subplots(figsize=(7, 4))

x = range(len(gender_data))

ax2.bar(
    x,
    gender_data["Survived"],
    label="Survived"
)

ax2.bar(
    x,
    gender_data["Did Not Survive"],
    bottom=gender_data["Survived"],
    label="Did Not Survive"
)

ax2.set_xticks(list(x))
ax2.set_xticklabels(gender_data["Gender"])

ax2.set_xlabel("Gender")
ax2.set_ylabel("Number of Passengers")
ax2.set_title("Survival by Gender")

ax2.legend()

plt.tight_layout()

st.pyplot(fig2)


# ---------------------------------------------------------
# Key Findings
# ---------------------------------------------------------

st.header("Key Findings")

st.write(
    "• The number of passengers who did not survive was higher "
    "than the number of passengers who survived."
)

st.write(
    "• Female passengers had a considerably higher survival "
    "rate than male passengers."
)

st.write(
    "• Passenger class and gender were important factors "
    "associated with survival."
)


# ---------------------------------------------------------
# Survival Prediction
# ---------------------------------------------------------

st.header("Survival Prediction")

st.write(
    "Enter passenger details below to get a simple survival prediction."
)

col1, col2 = st.columns(2)

with col1:

    passenger_class = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )


with col2:

    siblings = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    parents_children = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=30.0
    )


# ---------------------------------------------------------
# Simple Prediction Logic
# ---------------------------------------------------------

if st.button("Predict Survival"):

    score = 0

    # Gender was one of the strongest survival patterns
    if sex == "Female":
        score += 3
    else:
        score -= 2

    # Passenger class
    if passenger_class == 1:
        score += 2
    elif passenger_class == 2:
        score += 1
    else:
        score -= 1

    # Age
    if age <= 15:
        score += 1
    elif age >= 60:
        score -= 1

    # Fare gives a small indication of passenger class
    if fare >= 50:
        score += 1

    # Family size
    family_size = siblings + parents_children

    if 1 <= family_size <= 3:
        score += 1
    elif family_size >= 6:
        score -= 1

    if score >= 2:

        st.success(
            "Prediction: Passenger is likely to survive."
        )

    else:

        st.error(
            "Prediction: Passenger is likely not to survive."
        )

    st.write(
        "This prediction is an educational baseline based on "
        "patterns commonly observed in the Titanic dataset."
    )


# ---------------------------------------------------------
# Conclusion
# ---------------------------------------------------------

st.header("Conclusion")

st.write(
    "The Titanic dataset demonstrates how passenger characteristics "
    "can be used to study survival patterns. Gender, passenger class "
    "and age provide useful information for understanding survival outcomes."
)


# ---------------------------------------------------------
# Author
# ---------------------------------------------------------

st.divider()

st.header("Author")

st.write("Naina Kumari")
st.write(
    "Data Science Undergraduate | Machine Learning Enthusiast | Data Analyst"
)
