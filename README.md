# Titanic Survival Prediction

> A Machine Learning project that predicts whether a passenger on the Titanic survived or did not survive based on passenger information.

## Live Demo

**Streamlit Application:**  
https://titanicsurvivalprediction-ioudftgjjda28fku2kcgu8.streamlit.app

**GitHub Repository:**  
https://github.com/Naina137/Titanic_Survival_Prediction

### Application Preview

![Streamlit Application](streamlit_app.png)

---

## Project Overview

Titanic Survival Prediction is an end-to-end Machine Learning project developed using Python and Scikit-learn.

The project covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, model serialization, and deployment.

The trained model is integrated into an interactive Streamlit web application that allows users to enter passenger details and obtain a survival prediction.

---

## Objectives

- Analyze and preprocess the Titanic passenger dataset
- Perform exploratory data analysis and visualization
- Identify important factors associated with passenger survival
- Develop classification models for survival prediction
- Evaluate and compare model performance
- Save the trained model using Joblib
- Develop an interactive Streamlit application
- Deploy the Machine Learning model as a web application

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## Dataset

The project uses the Titanic passenger dataset.

### Features

| Feature | Description |
|---|---|
| Pclass | Passenger class |
| Age | Passenger age |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Fare | Passenger fare |
| Sex | Passenger gender |
| Embarked | Port of embarkation |
| Survived | Target variable |

### Target Variable

- `0` — Did not survive
- `1` — Survvived

## Project Structure

```text
Titanic_Survival_Prediction/
│
├── data/
│   └── train.csv
│
├── images/
│   ├── streamlit_app.png
│   ├── survival_distribution.png
│   ├── survival_gender.png
│   ├── survival_class.png
│   ├── random_forest_cm.png
│   └── logistic_regression_cm.png
│
├── Titanic_Survival_Prediction.ipynb
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand passenger characteristics and identify patterns related to survival.

### Survival Distribution

![Survival Distribution](survival_distribution.png)

### Survival by Gender

![Survival by Gender](survival_gender.png)

### Survival by Passenger Class

![Survival by Passenger Class](survival_class.png)

## Key Findings

- Female passengers had a higher survival rate than male passengers.
- Passenger class had a strong relationship with survival.
- First-class passengers generally had better survival outcomes than lower-class passengers.
- Age, family size and fare also contributed to the prediction of survival.
- The dataset contained more passengers who did not survive than passengers who survived.

## Machine Learning Models

The following classification models were implemented and compared:

### Logistic Regression

Used as a baseline classification model because it is simple and easy to interpret.

### Decision Tree

Used to capture decision-based relationships between passenger features and survival.

### Random Forest

An ensemble learning model that combines multiple decision trees to improve prediction performance.

### Gradient Boosting

An ensemble technique that builds models sequentially to improve prediction accuracy.

## Data Preprocessing

The following preprocessing steps were applied:

- Missing value handling
- Numerical feature scaling
- Categorical feature encoding
- Train-test split

A Scikit-learn preprocessing pipeline was used to keep the preprocessing and model training consistent.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

- ### Random Forest Confusion Matrix

![Random Forest Confusion Matrix](random_forest_cm.png)

### Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](logistic_regression_cm.png)

Model accuracy was compared to identify the best-performing classification model.

## Streamlit Web Application

The trained machine learning model was integrated into a Streamlit web application.

Users can enter passenger details such as:

- Passenger Class
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Sex
- Port of Embarkation

The application processes the entered information and predicts whether the passenger is likely to survive.

## Workflow

```text
Titanic Dataset
       ↓
Data Collection
       ↓
Data Understanding & Exploration
       ↓
Data Preprocessing
       ↓
Missing Value Handling
       ↓
Categorical Feature Encoding
       ↓
Numerical Feature Scaling
       ↓
Train-Test Split
       ↓
Model Training
       ↓
 ┌───────────────────────────────┐
 │ Logistic Regression           │
 │ Decision Tree                 │
 │ Random Forest                 │
 │ Gradient Boosting             │
 └───────────────────────────────┘
       ↓
Model Evaluation
       ↓
Accuracy Comparison
       ↓
Select Best Performing Model
       ↓
Passenger Details as Input
       ↓
Survival Prediction
       ↓
Streamlit Web Application
       ↓
Deployment on Streamlit Cloud
```

## Author

**Naina Kumari**

Data Science Undergraduate | Machine Learning Enthusiast | Data Analyst

This project was developed as part of my hands-on learning journey in
Data Science and Machine Learning, with the goal of understanding the
complete process of developing, evaluating, and deploying a Machine
Learning application.

### Connect with Me

- GitHub: https://github.com/Naina137
- LinkedIn: https://www.linkedin.com/in/naina-kumari-06373132b
- 

## Project Conclusion

This project demonstrates the complete Machine Learning workflow, from data preprocessing and exploratory analysis to model training, evaluation and deployment using Streamlit.
