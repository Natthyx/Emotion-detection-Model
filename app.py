from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from collections import Counter
import os
import requests
import time
from threading import Thread

app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Error loading Haar Cascade")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
BACKEND_URL = 'https://emotion-backend-sh1h.onrender.com' 
PING_INTERVAL = 300  # 5 minutes in seconds

def ping_backend():
    while True:
        try:
            response = requests.get(f'{BACKEND_URL}/ping')
            if response.status_code == 200:
                print(f"Ping to backend successful: {response.json()}")
            else:
                print(f"Ping to backend failed: Status {response.status_code}")
        except Exception as e:
            print(f"Ping to backend error: {e}")
        time.sleep(PING_INTERVAL)

@app.get("/ping")
async def ping():
    return JSONResponse(content={"status": "Emotion server is active"}, status_code=200)

@app.post("/predict")
async def predict_emotion(frames: list[UploadFile] = File(...)):
    emotion_counts = Counter()

    for frame in frames:
        if not frame.content_type.startswith('image/'):
            continue
        contents = await frame.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        if len(faces) == 0:
            emotion_counts['No face detected'] += 1
            continue
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        if face_img.size == 0:
            emotion_counts['Invalid face region'] += 1
            continue
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], face_img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        emotion = emotions[np.argmax(prediction)]
        emotion_counts[emotion] += 1

    valid_emotions = [e for e in emotion_counts if e in emotions]
    if not valid_emotions:
        return JSONResponse(content={"emotion": "No valid emotions detected"}, status_code=200)
    most_common = max(valid_emotions, key=lambda e: emotion_counts[e])
    return JSONResponse(content={"emotion": most_common}, status_code=200)

Thread(target=ping_backend, daemon=True).start()
