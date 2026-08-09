import cv2
import numpy as np
import os

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# ───────────────────────────────────────────────
# פונקציה לזיהוי גבות
# ───────────────────────────────────────────────
def create_eyebrow_mask_mediapipe(image_path, output_path='eyebrow_mask.jpg'):
    model_path = 'face_landmarker.task'  # הנתיב למודל שלך
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return False

    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255  # לבן = רקע

    # הגדרת FaceLandmarker
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            # Convert numpy array to MediaPipe Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
            detection_result = landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                print("No face landmarks detected.")
                return False

            face_landmarks = detection_result.face_landmarks[0]

            # אינדקסים של הגבות
            LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

            for brow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
                pts = []
                for idx in brow_indices:
                    lm = face_landmarks[idx]
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    pts.append([x, y])
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

                if len(pts) >= 3:
                    pts_array = np.array(pts, np.int32)
                    cv2.fillPoly(mask, [pts_array], 0)  # שחור = אזור הגבה

        cv2.imwrite(output_path, mask)
        print(f"Eyebrow mask saved to: {output_path}")

        viz_path = output_path.replace('.jpg','_viz.jpg').replace('.png','_viz.png')
        viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.addWeighted(image, 0.65, viz, 0.35, 0)
        cv2.imwrite(viz_path, combined)
        print(f"Visualization saved to: {viz_path}")

        return True

    except Exception as e:
        print(f"MediaPipe error: {e}")
        return False

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":
    input_image = r"tryIn\000001.jpg"
    output_mask = r"tryOut\eyebrow_mask.jpg"

    if not os.path.exists(input_image):
        print(f"Error: Input file not found: {input_image}")
    else:
        create_eyebrow_mask_mediapipe(input_image, output_mask)
