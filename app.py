# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Load and Prepare Data
@st.cache
def load_data():
    url = 'https://web.stanford.edu/class/cs102/datasets/Titanic.csv'
    data = pd.read_csv(url)
    return data

# Step 2: Preprocess Data
@st.cache
def preprocess_data(data):
    data['age'].fillna(data['age'].median(), inplace=True)
    data.dropna(subset=['embarked'], inplace=True)

    label_encoders = {}
    for column in ['gender', 'embarked']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data[['age', 'class', 'fare', 'embarked', 'gender']]
    y = data['survived'].apply(lambda x: 1 if x == 'yes' else 0)
    return X, y, label_encoders

# Step 3: Train Model
@st.cache
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit app layout
st.title("Titanic Survival Prediction")
st.write("This app predicts whether a passenger survived the Titanic disaster.")

# Load and preprocess data
data = load_data()
X, y, label_encoders = preprocess_data(data)

# Split and standardize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = train_model(X_train, y_train)

# Step 4: Model Evaluation and Visualization
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model evaluation metrics
st.subheader("Model Evaluation")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Survived', 'Survived'])
disp.plot(cmap='Blues', ax=ax)
st.pyplot(fig)

# Interactive Prediction Form
st.subheader("Make Your Prediction")
age = st.slider("Age", int(data['age'].min()), int(data['age'].max()), int(data['age'].median()))
pclass = st.selectbox("Passenger Class", sorted(data['class'].unique()))
fare = st.number_input("Fare", min_value=float(data['fare'].min()), max_value=float(data['fare'].max()), value=float(data['fare'].median()))
embarked = st.selectbox("Embarked (0=Cherbourg, 1=Queenstown, 2=Southampton)", [0, 1, 2])
gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])

# Prepare input for prediction
input_data = pd.DataFrame([[age, pclass, fare, embarked, gender]], columns=['age', 'class', 'fare', 'embarked', 'gender'])
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.write(f"The model predicts: **{result}**")
