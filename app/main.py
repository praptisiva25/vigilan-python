from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import os
import cv2
from ultralytics import YOLO

app = FastAPI()

class VideoRequest(BaseModel):
    s3_bucket: str
    s3_key: str

@app.post("/process-video")
def process_video(request: VideoRequest):
    try:
        os.makedirs("temp", exist_ok=True)
        local_path = os.path.join("temp", "video.mp4")

        # Download from S3
        s3 = boto3.client("s3")
        s3.download_file(request.s3_bucket, request.s3_key, local_path)

        # Open video
        cap = cv2.VideoCapture(local_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"status": "error", "message": "Could not read frame"}

        # Load YOLO model (first time will download weights)
        model = YOLO("yolov8n.pt")

        # Run detection
        results = model(frame)

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detections.append({
                    "object": class_name,
                    "confidence": round(confidence, 2)
                })

        return {
            "status": "detection_complete",
            "detections": detections
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}