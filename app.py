import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Titanic Survival Prediction")

st.write(
    "This project uses Machine Learning to predict whether "
    "a Titanic passenger survived based on passenger information."
)

st.divider()


# ============================================================
# ABOUT TITANIC
# ============================================================

st.subheader("About the Titanic")

st.write(
    "The RMS Titanic was a British passenger liner operated by "
    "the White Star Line. During its maiden voyage from Southampton "
    "to New York in April 1912, the ship struck an iceberg in the "
    "North Atlantic Ocean and sank."
)

st.write(
    "This project uses passenger information such as passenger "
    "class, sex, age, family members, fare and port of embarkation "
    "to train classification models."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    possible_paths = [
        "data/train.csv",
        "./data/train.csv",
        "train.csv",
        "./train.csv"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            try:
                data = pd.read_csv(path)

                if not data.empty:
                    return data

            except Exception:
                pass

    return None


df = load_data()


# ============================================================
# DATASET ERROR HANDLING
# ============================================================

if df is None:

    st.error(
        "Titanic dataset could not be found. "
        "Please make sure train.csv is inside the data folder."
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Total Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


with st.expander("View Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# FEATURES
# ============================================================

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    column
    for column in features + [target]
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Required columns are missing: {missing_columns}"
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

X = df[features].copy()

y = df[target].copy()


# Remove rows where target is missing

valid_rows = y.notna()

X = X.loc[valid_rows]

y = y.loc[valid_rows]


# ============================================================
# NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

trained_models = {}

model_scores = {}


for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    trained_models[model_name] = pipeline

    model_scores[model_name] = accuracy


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = max(
    model_scores,
    key=model_scores.get
)

best_model = trained_models[
    best_model_name
]

best_accuracy = model_scores[
    best_model_name
]


# ============================================================
# PASSENGER PREDICTION
# ============================================================

st.divider()

st.header("Predict Passenger Survival")

st.write(
    "Enter passenger information below. "
    "The best-performing model will generate the prediction."
)


with st.form("prediction_form"):

    passenger_class = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    siblings = st.number_input(
        "Number of Siblings/Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    parents_children = st.number_input(
        "Number of Parents/Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=1.0
    )

    sex = st.selectbox(
        "Sex",
        [
            "female",
            "male"
        ]
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        [
            "S",
            "C",
            "Q"
        ]
    )

    predict_button = st.form_submit_button(
        "Predict Survival",
        use_container_width=True
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

if predict_button:

    input_data = pd.DataFrame(
        {
            "pclass": [passenger_class],
            "sex": [sex],
            "age": [age],
            "sibsp": [siblings],
            "parch": [parents_children],
            "fare": [fare],
            "embarked": [embarked]
        }
    )

    prediction = best_model.predict(
        input_data
    )[0]

    probabilities = best_model.predict_proba(
        input_data
    )[0]

    survival_probability = probabilities[1] * 100

    if prediction == 1:

        st.success(
            f"Prediction: SURVIVED | "
            f"Survival Probability: "
            f"{survival_probability:.2f}%"
        )

    else:

        st.error(
            f"Prediction: DID NOT SURVIVE | "
            f"Survival Probability: "
            f"{survival_probability:.2f}%"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("Model Performance")


results = pd.DataFrame(
    {
        "Model": list(
            model_scores.keys()
        ),
        "Accuracy (%)": [
            round(
                score * 100,
                2
            )
            for score in model_scores.values()
        ]
    }
)


st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)


st.success(
    f"Best Model: {best_model_name} | "
    f"Accuracy: {best_accuracy * 100:.2f}%"
)


# ============================================================
# ACCURACY CHART
# ============================================================

st.subheader("Model Accuracy Comparison")


fig, ax = plt.subplots(
    figsize=(9, 4)
)

ax.bar(
    results["Model"],
    results["Accuracy (%)"]
)

ax.set_xlabel(
    "Machine Learning Model"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Titanic Survival Model Accuracy"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("Confusion Matrix")


best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)


fig2, ax2 = plt.subplots(
    figsize=(6, 5)
)

ax2.imshow(cm)

ax2.set_title(
    f"Confusion Matrix - {best_model_name}"
)

ax2.set_xlabel(
    "Predicted"
)

ax2.set_ylabel(
    "Actual"
)

ax2.set_xticks(
    [0, 1]
)

ax2.set_yticks(
    [0, 1]
)

ax2.set_xticklabels(
    [
        "Not Survived",
        "Survived"
    ]
)

ax2.set_yticklabels(
    [
        "Not Survived",
        "Survived"])
