import asyncio
import websockets
import json

async def send_imu_data():
    uri = "ws://10.37.28.160:8000/ws"  # Replace with your FastAPI server IP & port
    async with websockets.connect(uri) as websocket:
        imu_data = {
            "accel": [0.1, 0.0, 0.2, 0.1, -0.1, 0.0],
            "gyro": [0.0, 0.0, 0.05, 0.01, 0.0, -0.02]
        }
        while True:
            await websocket.send(json.dumps(imu_data))
            prediction = await websocket.recv()
            print(f"Prediction from server: {prediction}")
            await asyncio.sleep(1)  # Send data every second

if __name__ == "__main__":
    asyncio.run(send_imu_data())
