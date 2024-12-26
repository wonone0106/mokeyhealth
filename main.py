from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import shutil

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


class PushupResult(BaseModel):
    angle: float
    state: str


@app.post("/analyze-pushup/", response_model=PushupResult)
async def analyze_pushup(video_frame: UploadFile):
    # Save the uploaded frame temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(video_frame.file, tmp)
        temp_path = tmp.name

    try:
        # Load the frame and process
        frame = cv2.imread(temp_path)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_RGB)

        # Check if landmarks were detected
        if not results.pose_landmarks:
            return {"angle": -1, "state": "error"}  # No landmarks detected

        # Extract landmarks
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Convert normalized coordinates to pixel values
        shoulder_xy = (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0]))
        elbow_xy = (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0]))
        wrist_xy = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

        # Calculate angle
        angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)

        # Determine state
        if angle > 130:
            state = "down"
        elif angle < 90:
            state = "up"
        else:
            state = "down"

        return {"angle": angle, "state": state}

    except Exception as e:
        return {"angle": -1, "state": "error"}  # Error handling case
    finally:
        # Clean up temporary file
        video_frame.file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)