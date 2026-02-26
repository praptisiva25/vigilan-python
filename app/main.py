import os
import time
import requests
import boto3
import cv2
import json
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

app = FastAPI()

BACKEND_BASE_URL = "http://localhost:8080/api"


# ----------------------------
# Request Model
# ----------------------------
class MonitoringJobMessage(BaseModel):
    jobId: int
    videoPath: str
    mode: str


# ----------------------------
# Backend Helpers
# ----------------------------
def get_job_context(job_id):
    r = requests.get(f"{BACKEND_BASE_URL}/monitoring/{job_id}/context")
    r.raise_for_status()
    return r.json()


def update_job(job_id, status=None, progress=None):
    payload = {}
    if status:
        payload["status"] = status
    if progress is not None:
        payload["progress"] = progress

    requests.patch(
        f"{BACKEND_BASE_URL}/monitoring/{job_id}",
        json=payload
    )


def send_intrusion(data):
    requests.post(
        f"{BACKEND_BASE_URL}/intrusions",
        json=data
    )


# ----------------------------
# Zone Utilities
# ----------------------------
def build_polygons(hazard_zones):
    zone_map = {}

    for zone in hazard_zones:
        coords = json.loads(zone["polygonCoordinates"])

        # Convert {x,y} → (x,y)
        converted = [(p["x"], p["y"]) for p in coords]

        print("ZONE POLYGON:", converted)

        polygon = Polygon(converted)

        # Convert blocked string to proper list
        blocked_raw = zone["blockedObjects"] or ""
        blocked_list = [
            x.strip() for x in blocked_raw.split(",") if x.strip()
        ]

        zone_map[zone["id"]] = {
            "polygon": polygon,
            "blocked": blocked_list
        }

    return zone_map


# ----------------------------
# Main Processing Endpoint
# ----------------------------
@app.post("/process-job")
def process_job(message: MonitoringJobMessage):

    job_id = message.jobId
    video_path = message.videoPath
    mode = message.mode

    try:
        context = get_job_context(job_id)
        hazard_zones = context["hazardZones"]

        zone_map = build_polygons(hazard_zones)

        update_job(job_id, status="RUNNING", progress=0)

        os.makedirs("temp", exist_ok=True)
        local_path = os.path.join("temp", "video.mp4")

        # S3 support
        if video_path.startswith("s3://"):
            bucket = video_path.split("/")[2]
            key = "/".join(video_path.split("/")[3:])
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, local_path)
        else:
            local_path = video_path

        cap = cv2.VideoCapture(local_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        model = YOLO("yolov8n.pt")

        frame_number = 0
        active_intrusions = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            timestamp = frame_number / fps
            progress = int((frame_number / total_frames) * 100)

            if frame_number % 30 == 0:
                update_job(job_id, progress=progress)

            results = model.track(frame, persist=True)

            for result in results:
                for box in result.boxes:

                    if box.id is None:
                        continue

                    object_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    x1, y1, x2, y2 = box.xyxy[0]

                    # 🔥 Normalize center to 0–1
                    center_x = float((x1 + x2) / 2) / frame_width
                    center_y = float((y1 + y2) / 2) / frame_height

                    print("OBJECT CENTER:", center_x, center_y)

                    point = Point(center_x, center_y)

                    for zone_id, zone_data in zone_map.items():

                        polygon = zone_data["polygon"]
                        blocked_list = zone_data["blocked"]

                        inside = polygon.contains(point)

                        if inside and class_name in blocked_list:

                            if object_id not in active_intrusions:
                                active_intrusions[object_id] = {
                                    "zoneId": zone_id,
                                    "entryTime": timestamp
                                }

                        # Exit logic
                        elif object_id in active_intrusions:

                            entry_time = active_intrusions[object_id]["entryTime"]
                            duration = timestamp - entry_time

                            screenshot_path = f"temp/intrusion_{object_id}.jpg"
                            cv2.imwrite(screenshot_path, frame)

                            send_intrusion({
                                "monitoringJobId": job_id,
                                "hazardZoneId": active_intrusions[object_id]["zoneId"],
                                "objectId": object_id,
                                "entryTimeSeconds": entry_time,
                                "exitTimeSeconds": timestamp,
                                "durationSeconds": duration,
                                "screenshotUrl": screenshot_path
                            })

                            del active_intrusions[object_id]

            if mode == "LIVE":
                time.sleep(1 / fps)

        cap.release()

        update_job(job_id, status="COMPLETED", progress=100)

        return {"status": "completed"}

    except Exception as e:
        print("PYTHON ERROR:", str(e))
        import traceback
        traceback.print_exc()
        update_job(job_id, status="FAILED", progress=0)