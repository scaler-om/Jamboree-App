import pytest
from flask_app import app
import json

@pytest.fixture
def client():
    return app.test_client()

def test_home(client):
    resp = client.get('/')
    assert resp.status_code == 200