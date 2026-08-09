
import os
#from Predict_for_one_image import apply_mask
from mask_lips import create_lips_mask  # פונקציה שיוצרת מסכה
#from change_lips_color import change_lip_color    # פונקציה שמשנה צבע שפתיים
from change_lips_color import change_lip_color    # פונקציה שמשנה צבע שפתיים


# ====== SETTINGS ======
INPUT_FOLDER = r"input_faces"
OUTPUT_FOLDER = r"output_faces_change_lip_color"
MASK_FOLDER = r"lip_masks"
NEW_LIP_COLOR = "Modify ONLY the masked area to contain the new lip color. Change only the lip color to matte soft pink lipstick. Keep the skin tone, lighting, texture and natural shadows unchanged. Make it realistic and blended naturally. Do not change, regenerate, or alter any pixels outside of the masked region. Keep the background identical to the original image."

# ======================

# צור תיקיות אם לא קיימות
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

import cv2
import numpy as np


def apply_mask(face_path, mask_path, save_path='result.png'):
    # טוענים את תמונת הפנים והמסכה
    face = cv2.imread(face_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # גרייסקייל

    if face is None or mask is None:
        #print("Error: Couldn't load image or mask.")
        return

    # מוודאים שהמסכה בגודל זהה לתמונה
    mask = cv2.resize(mask, (face.shape[1], face.shape[0]))

    # יוצרים תמונה חדשה שמתחילה עם התמונה המקורית
    result = face.copy()

    # האזור שבו המסכה שחורה יהיה שחור בתמונה
    result[mask == 0] = 0  # אפס = שחור

    # החלק הלבן נשאר כפי שהוא (mask==255)
    
    # שומרים את התוצאה
    cv2.imwrite(save_path, result)
    print(f"Saved masked image to {save_path}")

# שימוש
apply_mask('face.png', 'mask.png', 'masked_face.png')

def process_single_image_lips(image_path):
    img_name = os.path.basename(image_path)

    # ===== 1. CREATE MASK =====
    mask_output_path = os.path.join(
        MASK_FOLDER, img_name.split('.')[0] + "_lips_mask.png"
    )
    print("Creating lips mask...")
    create_lips_mask(image_path, mask_output_path)
    apply_mask(image_path, mask_output_path, save_path='image_with_mask.png')
    # ===== 2. CHANGE LIP COLOR =====
    edited_output_path = os.path.join(
        OUTPUT_FOLDER, img_name.split('.')[0] + "_lips_edited.png"
    )
    print("Changing lip color...")
    change_lip_color(image_path, mask_output_path, NEW_LIP_COLOR, edited_output_path)

    print(f"✅ Done: {img_name}")

    return edited_output_path


def process_all_images():
    print("\n🚀 Starting batch lip color pipeline...\n")

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print("No images found in input folder.")
        return

    for img_name in images:
        try:
            print(f"\n🖼 Processing: {img_name}")

            input_path = os.path.join(INPUT_FOLDER, img_name)

            # ===== 1. CREATE MASK =====
            mask_output_path = os.path.join(MASK_FOLDER, img_name.split('.')[0] + "_lips_mask.png")
            print("Creating lips mask...")
            create_lips_mask(input_path, mask_output_path)

            # ===== 2. CHANGE LIP COLOR =====
            edited_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_lips_edited.png")
            print("Changing lip color...")
            change_lip_color(input_path, mask_output_path, NEW_LIP_COLOR, edited_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
   # process_single_image_lips(r"C:\Users\97258\engineering_try_2\Engineering-Project\input_faces\demo_image.jpeg")
    process_all_images()

