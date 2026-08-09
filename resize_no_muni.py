import cv2
import os

# ===== נתיבים =====
folder_path = r"C:\Users\97258\engineering_try_2\Engineering-Project\try_resize"
output_folder = os.path.join(folder_path, "resized_845_1024")
os.makedirs(output_folder, exist_ok=True)

# סיומות קבצים
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# ===== פונקציית resize =====
def resize_to_845_1024(img):
    target_w = 845
    target_h = 1024
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

# ===== מעבר על כל התמונות =====
images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

for filename in images:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {filename}")
        continue

    resized_img = resize_to_845_1024(img)

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, resized_img)

    print(f"Processed: {filename}")

print("All images resized to 845x1024 ✅")