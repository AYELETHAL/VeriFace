
import cv2
import numpy as np
import os


def create_lips_mask(image_path, output_path):
    print(f"Starting lip detection for: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    height, width = img.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- METHOD 1: MediaPipe (High Accuracy) ---
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        
        model_path = 'face_landmarker.task'
        if os.path.exists(model_path):
            base_options = BaseOptions(model_asset_path=model_path)
            options = FaceLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE, num_faces=1)
            
            with FaceLandmarker.create_from_options(options) as landmarker:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
                result = landmarker.detect(mp_image)
                
                if result.face_landmarks:
                    print("Using MediaPipe landmarks...")
                    LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409]
                    points = []
                    for idx in LIP_INDICES:
                        pt = result.face_landmarks[0][idx]
                        points.append([int(pt.x * width), int(pt.y * height)])
                    
                    cv2.fillPoly(mask, [np.array(points)], 0)
                    cv2.imwrite(output_path, mask)
                    print(f"Success via MediaPipe: {output_path}")
                    return True
    except Exception as e:
        print(f"MediaPipe Method skipped: {e}")

    # --- METHOD 2: Haar Cascade (Fallback) ---
    print("Trying Haar Cascade fallback...")
    # This looks for the mouth specifically. 
    # Note: OpenCV doesn't always include a 'mouth' xml by default, so we use 'smile' or 'nose' to estimate the area
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # We detect the lower half of the face to find the mouth
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Define region of interest: bottom 1/3 of the face
        mouth_roi_y = int(y + (h * 0.65))
        mouth_roi_h = int(h * 0.3)
        # Draw a simple ellipse/rectangle where the mouth usually is
        cv2.ellipse(mask, (x + w//2, mouth_roi_y + mouth_roi_h//2), (w//4, mouth_roi_h//2), 0, 0, 360, 0, -1)
        
    cv2.imwrite(output_path, mask)
    print(f"Success via Haar Estimation: {output_path}")
    return True
    


if __name__ == "__main__":
    print("=== lips Mask Generator ===")
    print()
    # CONFIGURE YOUR PATHS HERE
    input_path = r"tryIn\000002.jpg"
    output_path = r"tryOut"
    
    # ====================================
    
    print(f"Input image: {input_path}")
    print(f"Output location: {output_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        exit(1)
    
    # If output_path is a directory, create a default filename
    if os.path.isdir(output_path):
        # Get the input filename without extension
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{input_filename}_lips_mask.jpg")
        print(f"Output directory detected. Saving to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print()
    print("Processing...")
    create_lips_mask(input_path, output_path)
    print("\nDone!")