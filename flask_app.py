import pickle
from flask import Flask, render_template, request

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

app = Flask(__name__)
@app.route('/')
def default():
    return render_template('index.html')
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    #print(request.form)
    data = [[float(x) for x in request.form.values()]]
    prediction = admission_prediction(data)
    if prediction>0.5:
        return render_template('index.html', pred=f"Admission probability is high: {prediction}")
    else:
        return render_template('index.html', pred=f"Admission probability is low: {prediction}")


