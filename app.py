import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('models/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Web App Title
st.title("🌸 Iris Flower Classification App")
st.write("Enter the measurements of the flower to predict its species:")

# UI Sliders for user inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction Button
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    species = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    result = species[prediction[0]]
    
    # Display the output on screen
    st.success(f"The predicted species is: **{result}**")