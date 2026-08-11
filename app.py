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

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("Titanic Survival Prediction")

st.write(
    "This project uses Machine Learning to predict whether a "
    "Titanic passenger survived based on passenger information."
)

st.subheader("About the Titanic")

st.write(
    "The RMS Titanic was a British passenger liner operated by "
    "the White Star Line. It began its maiden voyage from "
    "Southampton to New York in April 1912. On April 14, 1912, "
    "the ship struck an iceberg in the North Atlantic and sank "
    "in the early hours of April 15, 1912."
)

st.write(
    "The Titanic dataset contains information such as passenger "
    "class, sex, age, family members, fare and port of embarkation. "
    "These features are used to train classification models."
)

st.divider()


@st.cache_data
def load_data():
    import seaborn as sns
    return sns.load_dataset("titanic")


try:
    df = load_data()

except Exception:
    st.error("The Titanic dataset could not be loaded.")

    uploaded_file = st.file_uploader(
        "Upload Titanic CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()


st.subheader("Predict Passenger Survival")

st.write(
    "Enter the passenger details below to get a survival prediction."
)

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

X = df[features].copy()
y = df[target].copy()


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


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


trained_models = {}
model_scores = {}


for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    model_scores[name] = accuracy_score(
        y_test,
        predictions
    )

    trained_models[name] = pipeline


best_model_name = max(
    model_scores,
    key=model_scores.get
)

best_model = trained_models[best_model_name]


with st.form("prediction_form"):

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
        value=25.0
    )

    siblings = st.number_input(
        "Siblings / Spouse",
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
        max_value=600.0,
        value=30.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )

    predict = st.form_submit_button(
        "Predict Survival",
        use_container_width=True
    )


if predict:

    input_data = pd.DataFrame({
        "pclass": [passenger_class],
        "sex": [sex],
        "age": [age],
        "sibsp": [siblings],
        "parch": [parents_children],
        "fare": [fare],
        "embarked": [embarked]
    })

    prediction = best_model.predict(input_data)[0]

    probability = best_model.predict_proba(input_data)[0][1]

    if prediction == 1:

        st.success(
            f"Prediction: SURVIVED | "
            f"Survival Probability: {probability * 100:.2f}%"
        )

    else:

        st.error(
            f"Prediction: DID NOT SURVIVE | "
            f"Survival Probability: {probability * 100:.2f}%"
        )


st.divider()

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

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


st.subheader("Model Performance")

results = pd.DataFrame({
    "Model": list(model_scores.keys()),
    "Accuracy (%)": [
        round(score * 100, 2)
        for score in model_scores.values()
    ]
})

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

st.write(
    f"Best Model: {best_model_name} "
    f"with {model_scores[best_model_name] * 100:.2f}% accuracy."
)


st.subheader("Model Accuracy Comparison")

fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(
    results["Model"],
    results["Accuracy (%)"]
)

ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Titanic Survival Model Accuracy")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)


st.subheader("Confusion Matrix")

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.imshow(cm)

ax2.set_title(
    f"Confusion Matrix - {best_model_name}"
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])

ax2.set_xticklabels([
    "Not Survived",
    "Survived"
])

ax2.set_yticklabels([
    "Not Survived",
    "Survived"
])

for i in range(2):
    for j in range(2):
        ax2.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

st.pyplot(fig2)


with st.expander("Classification Report"):

    report = classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Not Survived",
            "Survived"
        ]
    )

    st.text(report)


st.divider()

st.subheader("Project Information")

st.write(
    "Dataset: Titanic passenger dataset"
)

st.write(
    "Features used: Passenger Class, Sex, Age, "
    "Siblings/Spouse, Parents/Children, Fare and "
    "Port of Embarkation."
)

st.write(
    "Models used: Logistic Regression, Decision Tree, "
    "Random Forest and Gradient Boosting."
)

st.write(
    "Preprocessing includes missing-value handling, "
    "feature scaling and categorical encoding."
)
