import pickle
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
    data = scaler.transform(data)
    prediction = model.predict(data)
    return round(prediction[0][0], 2)

st.title("Admission Probability Prediction App")

gre_score = st.number_input("GRE Score[0, 340]", min_value=0, max_value=340)
toefl_score = st.number_input("TOEFL Score[0, 120]", min_value=0, max_value=120)
uni_rating = st.selectbox("University Rating", (0, 1, 2, 3, 4, 5))
sop = st.selectbox("SOP", (0, 1, 2, 3, 4, 5))
lor = st.selectbox("LOR", (0, 1, 2, 3, 4, 5))
cgpa = st.number_input("CGPA [0, 10]", min_value=0, max_value=10)
research = st.selectbox("Research", (0, 1))

# data = {'GRE Score': [gre_score],
#         'TOEFL Score': [toefl_score],
#         'University Rating': [uni_rating],
#         'SOP': [sop],
#         'LOR': [lor],
#         'CGPA': [cgpa],
#         'Research': [research]}

data = [[gre_score, toefl_score, uni_rating, sop, lor, cgpa, research]]

if st.button('Predict'):
    prediction = admission_prediction(data)
    if(prediction >= 0.5):
        st.success(f"Admission probability is high: {prediction}")
    else:
        st.error(f"Admission probability is low: {prediction}")


