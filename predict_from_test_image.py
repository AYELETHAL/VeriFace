import cv2
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torchvision.models as models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    from pipeline_lips import process_single_image_lips
    from pipeline_eye_color import process_single_image_eyes
    from pipeline_nose import process_single_image_nose         
    from pipeline_eyebrows import process_single_image_eyebrows 
    from remove_padding import remove_white_padding
except ImportError:
    print("Warning: Compilation pipelines couldn't be imported. Make sure paths are correct.")

# =========================
# Settings
# =========================
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding",
    "output_faces_change_nose_no_padding",      
    "output_faces_change_eyebrows_no_padding"   
]

IMG_SIZE = 224 
REFERENCE_IMAGE_PATH_RESIZE = r"C:\Users\97258\engineering_try_2\Engineering-Project\tryIn\000001.jpg"
REFERENCE_IMAGE_PATH_PAD = r"C:\Users\97258\engineering_try_2\Engineering-Project\output_faces_change_lip_color_with_padding\600_lips_edited.png"
JSON_PATH = r"C:\Users\97258\engineering_try_2\Engineering-Project\test_paths.json"
OUTPUT_DIR = "gradcam_results" # תיקייה לשמירת תוצאות ה-heatmap

os.makedirs(OUTPUT_DIR, exist_ok=True)

# הגדרת המכשיר
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# טעינת ובניית המודל
# =========================
print("Loading ResNet18 model...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(CATEGORIES))
)

model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# =========================
# פונקציות עזר מהקוד שלך
# =========================
def resize_to_reference(img, reference_path):
    ref_img = cv2.imread(reference_path)
    if ref_img is None: raise ValueError("Reference image not found")
    target_h, target_w = ref_img.shape[:2]
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

def pad_to_reference_size(img, reference_path):
    ref_img = cv2.imread(reference_path)
    if ref_img is None: raise ValueError("Reference image not found")
    target_h, target_w = ref_img.shape[:2]
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w, delta_h = target_w - new_w, target_h - new_h
    top, left = delta_h // 2, delta_w // 2
    bottom, right = delta_h - top, delta_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def remove_padding_simple(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white = gray < 245
    rows, cols = np.any(non_white, axis=1), np.any(non_white, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0,-1]]
        x_min, x_max = np.where(cols)[0][[0,-1]]
        return image[y_min:y_max, x_min:x_max]
    return image


# =========================
# Prediction & Grad-CAM Heatmap (Save Only)
# =========================
def predict_and_save_heatmap(image, img_num, true_label, image_name="image"):
    # 1. עיבוד מקדים עבור המודל
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype("float32") / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_tensor_input = (image_normalized - mean) / std
    
    image_transposed = np.transpose(image_tensor_input, (2, 0, 1))
    image_tensor = torch.tensor(image_transposed).unsqueeze(0).to(device)

    # 2. הרצת חיזוי
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    predicted_class_idx = predicted_idx.item()
    result_category = CATEGORIES[predicted_class_idx]

    # 3. יצירת ה-Heatmap באמצעות Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class_idx)]
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] 

    # 4. שילוב מפת החום על גבי התמונה
    visualization = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    """
    # הוספת טקסט על גבי התמונה שמתעדת את הלייבל והפרדיקציה (נשמר ישירות לתוך הקובץ)
    cv2.putText(visualization_bgr, f"True: {true_label}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization_bgr, f"Pred: {result_category}", (10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    """
    # הדפסה קשיחה לטרמינל כדי שתוכלי לעקוב אחרי הקצב
    print(f"[RESULT] #{img_num} | True Label: {true_label} | Prediction: {result_category}")
    
    # 5. שמירת התמונה ישירות לדיסק בפורמט המבוקש ללא תצוגה
    filename = f"{img_num:03d}_Label_{true_label}_Pred_{result_category}.jpg"
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, visualization_bgr)
    print(f"Saved heatmap to: {output_path}")

# =========================
# JSON Pipeline Loop
# =========================
# =========================
# JSON Pipeline Loop
# =========================
def run_json_pipeline(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # בניית רשימה אחידה של טאפלים: (נתיב_תמונה, לייבל_אמיתי)
    image_items = []
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "path" in item:
                image_items.append((item["path"], item.get("label", "unknown_label")))
            else:
                image_items.append((str(item), "unknown_label"))
                
    elif isinstance(data, dict):
        if "paths" in data and isinstance(data["paths"], list):
            for p in data["paths"]:
                image_items.append((p, "unknown_label"))
        else:
            for k, v in data.items():
                if os.path.exists(str(k)) or "/" in str(k) or "\\" in str(k):
                    image_items.append((str(k), str(v)))
                else:
                    image_items.append((str(v), str(k)))

    if not image_items:
        print("Unknown JSON format or empty file.")
        return

    print(f"Found {len(image_items)} images in JSON. Starting batch pipeline...")

    # ריצה בלולאה עם מונה (idx + 1) שמייצג את מספר התמונה
    for idx, (img_path, true_label) in enumerate(image_items):
        img_num = idx + 1
        
        # --- תיקון אוטומטי לחילוץ הלייבל מהנתיב במידה והוא לא נמצא ב-JSON ---
        if true_label == "unknown_label":
            # לוקח את שם התיקייה שבה נמצא הקובץ (למשל No_Manipulation)
            folder_name = os.path.basename(os.path.dirname(img_path))
            # בודק אם שם התיקייה הוא אחד מהקטיגוריות המוכרות שלך
            if folder_name in CATEGORIES:
                true_label = folder_name
            else:
                # גיבוי: אם התיקייה האמא היא לא הלייבל, ננסה לבדוק אם אחת הקטגוריות מוזכרת בתוך הנתיב השלם
                for cat in CATEGORIES:
                    if cat in img_path:
                        true_label = cat
                        break

        print(f"\n--- Processing [{img_num}/{len(image_items)}]: {img_path} ---")
        
        if not os.path.exists(img_path):
            print(f"Skipping: Image path does not exist: {img_path}")
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping: Failed to load image: {img_path}")
            continue

        try:
            # 1. שינוי גודל וריפוד (Padding) 
            resizes_image = resize_to_reference(image, REFERENCE_IMAGE_PATH_RESIZE)
            padded = pad_to_reference_size(resizes_image, REFERENCE_IMAGE_PATH_PAD)
            
            # 2. הסרת הריפוד 
            final_img = remove_padding_simple(padded)
            
            # 3. הרצת החיזוי, שמירה והצגה
            img_base_name = os.path.basename(img_path)
            predict_and_save_heatmap(final_img, img_num, true_label, image_name=img_base_name)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print("\nFinished processing all images! Check the 'gradcam_results' folder.")

# =========================
# Run
# =========================
if __name__ == "__main__":
    run_json_pipeline(JSON_PATH)