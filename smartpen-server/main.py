import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import random

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:19006",
    "http://127.0.0.1",
    "http://10.37.28.160",  # Replace with your PC IP address
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IMUData(BaseModel):
    accel: List[float]
    gyro: List[float]

class CNNLSTM(nn.Module):
    def __init__(self, n_features=6, n_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, features, seq_length]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [B, seq_length, features]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last time step
        return self.fc(x)

model = CNNLSTM(n_features=6, n_classes=10)
model.load_state_dict(torch.load("cnn_lstm_imu_model.pth"))
model.eval()

def preprocess(data: IMUData):
    combined = np.array(data.accel + data.gyro).reshape(1, -1, 6)  # mock batch, seq, feature
    return torch.tensor(combined).float()

@app.get("/")
async def root():
    return {"message": "SmartPen FastAPI Server Running"}

@app.post("/imu-data/")
async def imu_data_endpoint(data: IMUData):
    try:
        x = preprocess(data)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return {"predicted_class": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Uncomment below block when ready to receive real IMU data from client
            # data = await websocket.receive_json()
            # imu_data = IMUData(**data)
            # x = preprocess(imu_data)
            # logits = model(x)
            # pred = torch.argmax(logits, dim=1).item()
            # await manager.send_personal_message(str(pred), websocket)

            # Dummy IMU data sending test
            dummy_data = {
                "accel": [random.uniform(-1, 1) for _ in range(6)],
                "gyro": [random.uniform(-1, 1) for _ in range(6)]
            }
            imu_data = IMUData(**dummy_data)
            x = preprocess(imu_data)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            await manager.send_personal_message(str(pred), websocket)

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
