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
special_emotions = ["No face detected", "Invalid face region"]
all_possible_emotions = emotions + special_emotions

EMOTION_URL = 'https://emotion-backend-sh1h.onrender.com'
PING_INTERVAL = 300

latest_emotion = "Waiting for emotion..."

def ping_emotion_server():
    while True:
        try:
            response = requests.get(f'{EMOTION_URL}/ping')
            if response.status_code == 200:
                print(f"Ping to emotion-detector successful: {response.json()}")
            else:
                print(f"Ping to emotion-detector failed: Status {response.status_code}")
        except Exception as e:
            print(f"Ping to emotion-detector error: {e}")
        time.sleep(PING_INTERVAL)

@app.get("/ping")
async def ping():
    print("Received ping request")
    return JSONResponse(content={"status": "Emotion server is active"}, status_code=200)

@app.get("/emotion")
async def get_emotion():
    print(f"Returning latest emotion: {latest_emotion}")
    return JSONResponse(content={"emotion": latest_emotion}, status_code=200)

@app.post("/predict")
async def predict_emotion(frames: list[UploadFile] = File(...)):
    global latest_emotion
    print(f"Received {len(frames)} frames for processing")
    emotion_counts = Counter()
    warnings = []

    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}: {frame.filename}")
        if not frame.content_type.startswith('image/'):
            warnings.append(f"Skipped non-image file: {frame.filename}, type: {frame.content_type}")
            print(f"Warning: {warnings[-1]}")
            continue

        contents = await frame.read()
        print(f"Read {len(contents)} bytes")
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            if len(faces) == 0:
                emotion_counts['No face detected'] += 1
                print("No faces detected in frame")
                continue

            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            if face_img.size == 0:
                emotion_counts['Invalid face region'] += 1
                print("Invalid face region")
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
            print(f"Predicted emotion: {emotion}")
        except Exception as e:
            warnings.append(f"Error processing frame {frame.filename}: {str(e)}")
            print(f"Warning: {warnings[-1]}")
            continue

    print(f"Emotion counts: {dict(emotion_counts)}")

    # Consider all possible emotions including special cases
    valid_emotions = [e for e in emotion_counts if e in all_possible_emotions]
    if not valid_emotions:
        emotion = "No valid emotions detected"
        print("No valid emotions detected")
    else:
        emotion = max(valid_emotions, key=lambda e: emotion_counts[e])
        print(f"Dominant emotion: {emotion}")

    latest_emotion = emotion
    print(f"Stored latest emotion: {latest_emotion}")

    response_content = {"emotion": emotion, "counts": dict(emotion_counts)}
    if warnings:
        response_content["warnings"] = warnings
        print(f"Warnings: {warnings}")

    print("Returning response")
    return JSONResponse(content=response_content, status_code=200)

Thread(target=ping_emotion_server, daemon=True).start()
