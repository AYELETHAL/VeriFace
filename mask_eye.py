import cv2
import numpy as np
import os

def create_pupil_mask_dlib_method(image_path, output_path='pupil_mask.jpg'):
    """
    Creates a mask using OpenCV's built-in eye detection.
    This is a fallback method that doesn't require MediaPipe.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create white mask (all white initially)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No face detected in the image")
        return False
    
    pupils_found = 0
    
    # For each face, detect eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Only look for eyes in the upper 60% of the face (to exclude mouth)
        upper_face_height = int(h * 0.6)
        roi_gray_upper = roi_gray[0:upper_face_height, :]
        
        # Detect eyes in the upper face region with adjusted parameters
        eyes = eye_cascade.detectMultiScale(roi_gray_upper, 1.1, 3, minSize=(30, 30))
        
        # Store eye centers to avoid duplicates
        eye_centers = []
        
        for (ex, ey, ew, eh) in eyes:
            # Filter: Eyes should be in upper half and horizontally separated
            # Skip if detection is too low (likely mouth)
            if ey > upper_face_height * 0.7:
                continue
            
            # Calculate pupil position (center of detected eye region)
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            
            # Check if this eye is too close to an already detected eye (avoid duplicates)
            is_duplicate = False
            for prev_x, prev_y in eye_centers:
                distance = np.sqrt((eye_center_x - prev_x)**2 + (eye_center_y - prev_y)**2)
                if distance < ew * 0.5:  # If centers are very close, it's likely a duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                eye_centers.append((eye_center_x, eye_center_y))
                
                # Estimate pupil radius (about 30% of eye width)
                pupil_radius = int(ew * 0.3)
                
                # Draw black circle for pupil on mask
                cv2.circle(mask, (eye_center_x, eye_center_y), pupil_radius, 0, -1)
                pupils_found += 1
        
        # If we detected exactly 2 eyes, we're good
        # If we detected less than 2, try with different parameters
        if len(eye_centers) < 2:
            # Try with more sensitive parameters
            eyes2 = eye_cascade.detectMultiScale(roi_gray_upper, 1.05, 2, minSize=(25, 25))
            
            for (ex, ey, ew, eh) in eyes2:
                # Filter: Skip if too low in face
                if ey > upper_face_height * 0.7:
                    continue
                
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                
                # Check if this is a new eye
                is_duplicate = False
                for prev_x, prev_y in eye_centers:
                    distance = np.sqrt((eye_center_x - prev_x)**2 + (eye_center_y - prev_y)**2)
                    if distance < ew * 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(eye_centers) < 2:
                    eye_centers.append((eye_center_x, eye_center_y))
                    pupil_radius = int(ew * 0.3)
                    cv2.circle(mask, (eye_center_x, eye_center_y), pupil_radius, 0, -1)
                    pupils_found += 1
    
    if pupils_found == 0:
        print("No eyes/pupils detected in the image")
        return False
    
    # Save the mask
    cv2.imwrite(output_path, mask)
    print(f"Pupil mask saved to: {output_path}")
    print(f"Detected {pupils_found} pupil(s)")
    
    # Create visualization
    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
    combined = cv2.addWeighted(image, 0.7, visualization, 0.3, 0)
    cv2.imwrite(output_viz, combined)
    print(f"Visualization saved to: {output_viz}")
    
    return True


def create_pupil_mask_mediapipe_new(image_path, output_path='pupil_mask.jpg'):
    """
    Creates a mask using the new MediaPipe API (0.10.8+).
    """
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
    except ImportError:
        print("New MediaPipe API not available, falling back to OpenCV method")
        return create_pupil_mask_dlib_method(image_path, output_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create white mask
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = python.vision.Image(image_format=python.vision.ImageFormat.SRGB, data=image_rgb)
    
    # Create FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks:
                print("No face detected, trying OpenCV fallback method")
                return create_pupil_mask_dlib_method(image_path, output_path)
            
            face_landmarks = detection_result.face_landmarks[0]
            
            # Eye landmarks (approximate indices for iris/pupil)
            # Left eye: indices around 468-473
            # Right eye: indices around 473-478
            left_eye_center = face_landmarks[468]
            right_eye_center = face_landmarks[473]
            
            # Convert normalized coordinates to pixel coordinates
            left_x = int(left_eye_center.x * width)
            left_y = int(left_eye_center.y * height)
            right_x = int(right_eye_center.x * width)
            right_y = int(right_eye_center.y * height)
            
            # Estimate pupil radius
            eye_distance = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            pupil_radius = int(eye_distance * 0.08)
            
            # Draw black circles for pupils
            cv2.circle(mask, (left_x, left_y), pupil_radius, 0, -1)
            cv2.circle(mask, (right_x, right_y), pupil_radius, 0, -1)
            
            # Save results
            cv2.imwrite(output_path, mask)
            print(f"Pupil mask saved to: {output_path}")
            
            visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
            combined = cv2.addWeighted(image, 0.7, visualization, 0.3, 0)
            cv2.imwrite(output_viz, combined)
            print(f"Visualization saved to: {output_viz}")
            
            return True
            
    except Exception as e:
        print(f"MediaPipe error: {e}")
        print("Falling back to OpenCV method")
        return create_pupil_mask_dlib_method(image_path, output_path)


def create_pupil_mask(image_path, output_path='pupil_mask.jpg'):
    """
    Main function that tries different methods to create pupil mask.
    """
    print("Attempting to create pupil mask...")
    
    # Try OpenCV method (most reliable and doesn't need extra models)
    success = create_pupil_mask_dlib_method(image_path, output_path)
    
    if not success:
        print("\nFailed to create pupil mask. Please ensure:")
        print("  1. The image contains a clearly visible face")
        print("  2. The eyes are open and well-lit")
        print("  3. The face is roughly frontal")
    
    return success
from PIL import Image
import cv2
import numpy as np


def create_mask(input_image_path, output_mask_path):
    """
    Detect eyes and create mask:
    eyes = black
    rest = white
    """

    print(f"Creating mask for {input_image_path}")

    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar cascade לזיהוי עיניים
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # מסכה לבנה
    mask = np.ones_like(gray) * 255

    for (x, y, w, h) in eyes:
        # מציירים אזור שחור בעיניים
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")


if __name__ == "__main__":
    print("=== Pupil Mask Generator ===")
    print()
    2
    # CONFIGURE YOUR PATHS HERE
    input_path = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\tryIn\000002.jpg"
    output_path = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\tryOut"
    
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
        output_path = os.path.join(output_path, f"{input_filename}_mask.jpg")
        print(f"Output directory detected. Saving to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print()
    print("Processing...")
    create_pupil_mask(input_path, output_path)
    
    print("\nDone!")