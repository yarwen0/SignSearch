"""MediaPipe Tasks VIDEO extraction. Invoked by prepare.py's process pool."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path
from time import perf_counter
import numpy as np
from signsearch.data import POSE, normalize


def extract_clip(root, row):
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    root = Path(root)
    video = root / row["video_path"]
    output = root / row["keypoints"]
    cap = cv2.VideoCapture(str(video))
    points, masks, timings = [], [], []
    try:
        if not cap.isOpened():
            raise ValueError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video frame rate")
        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        hands_options = vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(root / "models/hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(root / "models/pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        with vision.HandLandmarker.create_from_options(hands_options) as hands, vision.PoseLandmarker.create_from_options(pose_options) as pose:
            frame_index = 0
            last_timestamp = -1
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                height, width = bgr.shape[:2]
                if max(height, width) > 640:
                    bgr = cv2.resize(bgr, (round(width * 640 / max(height, width)), round(height * 640 / max(height, width))))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                timestamp = max(last_timestamp + 1, round(frame_index * 1000 / fps))
                last_timestamp = timestamp
                start = perf_counter()
                hand_result = hands.detect_for_video(image, timestamp)
                pose_result = pose.detect_for_video(image, timestamp)
                timings.append((perf_counter() - start) * 1000)
                xy = np.zeros((49, 2), np.float32)
                present = np.zeros(3, bool)
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks[0]
                    if min(landmarks[i].visibility for i in (11, 12)) >= .5:
                        xy[:7] = [(landmarks[i].x * width, landmarks[i].y * height) for i in POSE]
                        present[0] = True
                confidence = [-1., -1.]
                for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    category = handedness[0]
                    side = 0 if category.category_name == "Left" else 1
                    if category.score > confidence[side]:
                        confidence[side] = category.score
                        xy[7 + side * 21:28 + side * 21] = [(p.x * width, p.y * height) for p in landmarks]
                        present[1 + side] = True
                points.append(xy)
                masks.append(present)
                frame_index += 1
        if not points:
            raise ValueError("No decoded frames")
        features = normalize(np.stack(points), np.stack(masks))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial.npz")
        np.savez_compressed(temporary, features=features.astype(np.float16), frame_ms=np.asarray(timings, np.float32), fps=np.float32(fps))
        temporary.replace(output)
        # Delete only this pipeline's video after the landmark artifact is safely written.
        video.unlink()
        return {"video": row["video"], "split": row["split"], "frames": len(features), "extraction_ms": float(np.sum(timings))}
    finally:
        cap.release()
