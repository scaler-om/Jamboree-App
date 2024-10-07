import pytest
from flask_app import app
import json

@pytest.fixture
def client():
    return app.test_client()

def test_predict_high_probability(client):
    # Provide form data manually
    form_data = {'GRE Score': '318.0',
                 'TOEFL Score': '106.0',
                 'University Rating': '2.0',
                 'SOP': '4.0',
                 'LOR': '4.0',
                 'CGPA': '7.92',
                 'Research': '1.0'}

    # Simulate POST request with form data
    resp = client.get('/predict', data=form_data)

    # Assert the response is 200 OK
    assert resp.status_code == 200

    # Check if the correct message is rendered
    assert b"Admission probability is high: 0.66" in resp.data

def test_predict_low_probability(client):
    # Provide form data manually
    form_data = {'GRE Score': '100.0',
                 'TOEFL Score': '40.0',
                 'University Rating': '2.0',
                 'SOP': '2.0',
                 'LOR': '2.0',
                 'CGPA': '4.5',
                 'Research': '0.0'}

    # Simulate POST request with form data
    resp = client.get('/predict', data=form_data)

    # Assert the response is 200 OK
    assert resp.status_code == 200

    # Check if the correct message is rendered
    assert b"Admission probability is low: -0.45" in resp.data