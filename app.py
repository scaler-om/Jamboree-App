import numpy as np
import pickle
import pandas as pd
import streamlit as st

scaler = pickle.load(open("./model/scaler.pkl", 'rb'))
model = pickle.load(open("./model/model.pkl", 'rb'))


# data = {'GRE Score': [318.0],
#         'TOEFL Score': [106.0],
#         'University Rating': [2.0],
#         'SOP': [4.0],
#         'LOR': [4.0],
#         'CGPA': [7.92],
#         'Research': [1.0]}

def admission_prediction(data):
    df = pd.DataFrame(data=data)
    df = scaler.transform(df)
    prediction = model.predict(df)
    return prediction

st.title("Admission Probability Prediction App")

# GRE Score -> 0 to 340
# TOEFL Score -> 0 to 120
# University Rating -> 0 to 5
# SOP -> 0 to 5
# LOR -> 0 to 5
# CGPA -> 0 to 10
# Research -> 0 or 1

gre_score = st.number_input("GRE Score[0, 340]", min_value=0, max_value=340)
toefl_score = st.number_input("TOEFL Score[0, 120]", min_value=0, max_value=120)
uni_rating = st.selectbox("University Rating", (0, 1, 2, 3, 4, 5))
sop = st.selectbox("SOP", (0, 1, 2, 3, 4, 5))
lor = st.selectbox("LOR", (0, 1, 2, 3, 4, 5))
cgpa = st.selectbox("CGPA", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
research = st.selectbox("Research", (0, 1))

data = {'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [uni_rating],
        'SOP': [sop],
        'LOR': [lor],
        'CGPA': [cgpa],
        'Research': [research]}


if st.button('Predict'):
    probability = admission_prediction(data)
    st.success(round(probability[0][0],2))


