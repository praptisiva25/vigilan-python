import os
import time
import json
import requests
import boto3
import cv2
import tempfile
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# ---------------- CONFIG ---------------- #

BACKEND_BASE_URL = "http://localhost:8080/api"
SQS_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/591789164799/vigilan-monitoring-queue"
AWS_REGION = "ap-south-1"
S3_BUCKET = "vigilan-intrusions"

sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Load YOLO model ONCE (important for performance)
model = YOLO("yolov8n.pt")


# ---------------- BACKEND COMMUNICATION ---------------- #

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

    r = requests.patch(
        f"{BACKEND_BASE_URL}/monitoring/{job_id}",
        json=payload
    )
    r.raise_for_status()


def send_intrusion(data):
    r = requests.post(
        f"{BACKEND_BASE_URL}/intrusions",
        json=data
    )
    r.raise_for_status()


# ---------------- ZONE BUILDER ---------------- #

def build_polygons(hazard_zones):
    zone_map = {}

    for zone in hazard_zones:
        coords = json.loads(zone["polygonCoordinates"])
        converted = [(p["x"], p["y"]) for p in coords]
        polygon = Polygon(converted)
        zone_map[zone["id"]] = polygon

    return zone_map


# ---------------- SCREENSHOT UPLOAD ---------------- #

def upload_screenshot(frame, job_id, object_id, timestamp):

    filename = f"job_{job_id}_obj_{object_id}_{int(timestamp)}.jpg"

    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, filename)

    cv2.imwrite(local_path, frame)

    if not os.path.exists(local_path):
        raise Exception("Screenshot not saved locally")

    s3_key = f"intrusions/{filename}"

    s3_client.upload_file(
        local_path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print("Uploaded to S3:", s3_key)

    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


# ---------------- MAIN JOB PROCESSING ---------------- #

def process_job(message):

    job_id = message["jobId"]
    video_path = message["videoPath"]

    print(f"Starting job {job_id}")

    context = get_job_context(job_id)
    zone_map = build_polygons(context["hazardZones"])

    update_job(job_id, status="RUNNING", progress=0)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_number = 0
    active_intrusions = {}
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp = frame_number / fps

        # Safe progress update (never hit 100 during processing)
        if total_frames > 0 and frame_number % 30 == 0:
            progress = min(int((frame_number / total_frames) * 100), 99)
            update_job(job_id, progress=progress)

        results = model.track(frame, persist=True)

        for result in results:
            for box in result.boxes:

                if box.id is None:
                    continue

                object_id = int(box.id[0])

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = float((x1 + x2) / 2) / frame_width
                center_y = float((y1 + y2) / 2) / frame_height

                point = Point(center_x, center_y)

                inside_any_zone = False
                current_zone_id = None

                for zone_id, polygon in zone_map.items():
                    if polygon.contains(point):
                        inside_any_zone = True
                        current_zone_id = zone_id
                        break

                # ENTRY
                if inside_any_zone:
                    if object_id not in active_intrusions:

                        screenshot_url = upload_screenshot(
                            frame, job_id, object_id, timestamp
                        )

                        active_intrusions[object_id] = {
                            "zoneId": current_zone_id,
                            "entryTime": timestamp,
                            "screenshotUrl": screenshot_url
                        }

                # EXIT
                else:
                    if object_id in active_intrusions:

                        entry_time = active_intrusions[object_id]["entryTime"]
                        duration = timestamp - entry_time

                        send_intrusion({
                            "monitoringJobId": job_id,
                            "hazardZoneId": active_intrusions[object_id]["zoneId"],
                            "objectId": object_id,
                            "entryTimeSeconds": entry_time,
                            "exitTimeSeconds": timestamp,
                            "durationSeconds": duration,
                            "screenshotUrl": active_intrusions[object_id]["screenshotUrl"]
                        })

                        del active_intrusions[object_id]

    # Flush remaining intrusions at video end
    for object_id, data in active_intrusions.items():
        send_intrusion({
            "monitoringJobId": job_id,
            "hazardZoneId": data["zoneId"],
            "objectId": object_id,
            "entryTimeSeconds": data["entryTime"],
            "exitTimeSeconds": timestamp,
            "durationSeconds": timestamp - data["entryTime"],
            "screenshotUrl": data["screenshotUrl"]
        })

    cap.release()

    update_job(job_id, status="COMPLETED", progress=100)

    print(f"Job {job_id} completed successfully")


# ---------------- SQS POLLING ---------------- #

def poll_sqs():

    print("SQS Worker started...")

    while True:
        response = sqs_client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )

        messages = response.get("Messages", [])

        for msg in messages:
            body = json.loads(msg["Body"])

            try:
                process_job(body)

                sqs_client.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )

            except Exception as e:
                print("Job failed:", e)

                update_job(body["jobId"], status="FAILED")

                # Do NOT delete message → allow retry

        time.sleep(1)


if __name__ == "__main__":
    poll_sqs()