#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing required libraries
import streamlit as st
#model is downloaded & loading here
import joblib

#Mathematical operation
import numpy as np

# Load model and scaler (Loading files into model & scaler variables)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# In[6]:


# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="+")
st.title("🩺 Diabetes Prediction")
st.write("Enter the feature values below to predict the diagnosis.")

# User Inputs
Pregnancies = st.number_input("Pregnancies", min_value=0.0, format="%.2f")
Glucose = st.number_input("Glucose", min_value=0.0, format="%.2f")
BloodPressure = st.number_input("BloodPressure", min_value=0.0, format="%.2f")
SkinThickness = st.number_input("SkinThickness", min_value=0.0, format="%.2f")
Insulin = st.number_input("Insulin", min_value=0.0, format="%.2f")
BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.2f")
Age = st.number_input("Age", min_value=0.0, format="%.2f")

if st.button("Predict Diagnosis"):
    # Create input array
    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, 
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Display Result
    if prediction[0] == 1:
        st.error("🔴 Diagnosis: Diabetes - Start immediate Treatment")
    else:
        st.success("🟢 Diagnosis: No - Cool")  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




