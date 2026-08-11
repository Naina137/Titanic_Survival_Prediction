import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix


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
    "A Machine Learning application that predicts whether "
    "a Titanic passenger survived based on passenger information."
)

st.divider()


# ============================================================
# ABOUT TITANIC
# ============================================================

st.header("About the Titanic")

st.write(
    "The RMS Titanic was a British passenger liner operated by "
    "the White Star Line. During its maiden voyage in April 1912, "
    "the ship struck an iceberg in the North Atlantic and sank."
)

st.write(
    "This project uses passenger information such as passenger "
    "class, sex, age, family members, fare and port of embarkation "
    "to predict passenger survival."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")


try:
    df = load_data()

except Exception:
    st.error(
        "Dataset could not be loaded. "
        "Please check that data/train.csv exists."
    )
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# COLUMN MAPPING
# ============================================================

column_mapping = {
    "Pclass": "pclass",
    "Sex": "sex",
    "Age": "age",
    "SibSp": "sibsp",
    "Parch": "parch",
    "Fare": "fare",
    "Embarked": "embarked",
    "Survived": "survived"
}

df = df.rename(columns=column_mapping)


# ============================================================
# FEATURES AND TARGET
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


X = df[features].copy()
y = df[target].copy()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Passengers",
        df.shape[0]
    )

with col2:
    st.metric(
        "Total Features",
        len(features)
    )

with col3:
    st.metric(
        "Missing Values",
        int(df[features].isnull().sum().sum())
    )


with st.expander("View Dataset"):
    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# DATA PREPROCESSING
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
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("Model Performance")

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


st.subheader(
    "Model Accuracy Comparison"
)

st.bar_chart(
    results.set_index("Model")["Accuracy (%)"]
)


# ============================================================
# PASSENGER PREDICTION
# ============================================================

st.divider()

st.header("Passenger Survival Prediction")

st.write(
    "Enter passenger details below to generate "
    "a survival prediction."
)


with st.form(
    "prediction_form"
):

    passenger_class = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    siblings = st.number_input(
        "Number of Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    parents_children = st.number_input(
        "Number of Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Passenger Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
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
            "pclass": [
                passenger_class
            ],
            "sex": [
                sex
            ],
            "age": [
                age
            ],
            "sibsp": [
                siblings
            ],
            "parch": [
                parents_children
            ],
            "fare": [
                fare
            ],
            "embarked": [
                embarked
            ]
        }
    )


    prediction = best_model.predict(
        input_data
    )[0]


    probability = (
        best_model.predict_proba(
            input_data
        )[0][1] * 100
    )


    if prediction == 1:

        st.success(
            f"Prediction: SURVIVED\n\n"
            f"Survival Probability: "
            f"{probability:.2f}%"
        )

    else:

        st.error(
            f"Prediction: DID NOT SURVIVE\n\n"
            f"Survival Probability: "
            f"{probability:.2f}%"
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.header("Confusion Matrix")

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual: Not Survived",
        "Actual: Survived"
    ],
    columns=[
        "Predicted: Not Survived",
        "Predicted: Survived"
    ]
)


st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# PROJECT DETAILS
# ============================================================

st.divider()

st.header("Project Details")

st.markdown(
    """
**Dataset:** Titanic Passenger Dataset

**Target Variable:**
- 0 = Did not survive
- 1 = Survived

**Features Used:**
- Passenger Class
- Sex
- Age
- Siblings / Spouses
- Parents / Children
- Fare
- Port of Embarkation

**Machine Learning Models:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

**Data Preprocessing:**
- Missing value handling
- Numerical feature scaling
- Categorical feature encoding
- Train/Test split

**Deployment:**
- Streamlit
- Streamlit Cloud
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Titanic Survival Prediction | Machine Learning + Streamlit"
)
