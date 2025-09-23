from fastapi.testclient import TestClient
from main import app  # import your FastAPI app from main.py

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SmartPen FastAPI Server Running"}

def test_imu_data_prediction():
    sample_data = {
        "accel": [0.1, 0.0, 0.2, 0.1, -0.1, 0.0],
        "gyro": [0.0, 0.0, 0.05, 0.01, 0.0, -0.02]
    }
    response = client.post("/imu-data/", json=sample_data)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_class" in json_resp
    assert isinstance(json_resp["predicted_class"], int)
