import streamlit as st
import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load('iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Title of the app
st.title("Iris Species Classifier")

# Input fields for sepal and petal dimensions
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")

# Predict button
if st.button("Classify"):
    # Create a feature array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the species
    prediction = model.predict(features)
    species = label_encoder.inverse_transform(prediction)[0]

    # Display the result
    st.success(f"The predicted species is: **{species}**")