# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import base64

# ייבוא הפונקציות המקוריות שלך מקובץ החיזוי
from Predict_one_img_cnn import run_manipulation_pipeline, predict_image, pad_to_reference_size, resize_to_reference, remove_padding_simple

app = FastAPI()

# מאפשר ל-React (שפועל בפורט אחר) לגשת לשרת הפייתון
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # בסביבת פרודקשן כדאי להגביל לפורט של React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# נתיבים זמניים לשמירת תמונות במהלך העיבוד
UPLOAD_DIR = "temp_ui_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def image_to_base64(img_np):
    """פונקציית עזר להמרת מטריצת OpenCV למחרוזת Base64 כדי ש-React יוכל להציג אותה"""
    _, buffer = cv2.imencode('.jpg', img_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.post("/process")
async def process_and_predict(
    file: UploadFile = File(...),
    manipulation: str = Form(...) # מקבל את שם המניפולציה מה-UI
):
    try:
        # 1. קריאת הקובץ שנשלח מה-UI והפיכתו לתמונת OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. שמירה זמנית של תמונת המקור לצורך ה-pipeline הקיים שלך
        orig_path = os.path.join(UPLOAD_DIR, "orig.jpg")
        cv2.imwrite(orig_path, image)
        
        # 3. הרצת ה-Pipeline הקיים שלך (Resize + Padding)
        # (משתמש בנתיבי הרפרנס שהגדרת ב-Settings של קובץ החיזוי)
        from Predict_for_one_image import REFERENCE_IMAGE_PATH_RESIZE, REFERENCE_IMAGE_PATH_PAD
        resized_img = resize_to_reference(image, REFERENCE_IMAGE_PATH_RESIZE)
        padded_img = pad_to_reference_size(resized_img, REFERENCE_IMAGE_PATH_PAD)
        
        padded_path = os.path.join(UPLOAD_DIR, "padded.jpg")
        cv2.imwrite(padded_path, padded_img)
        
        # 4. הרצת המניפולציה שנבחרה ב-UI (שפתיים, עיניים, אף או גבות)
        manipulated_path = run_manipulation_pipeline(manipulation, padded_path)
        
        # 5. קריאת התוצאה המניפולטיבית והסרת פדינג
        final_img = cv2.imread(manipulated_path)
        final_img_clean = remove_padding_simple(final_img)
        
        # 6. הרצת מודל ה-ResNet18 לחיזוי + הפקת מפת החום (Grad-CAM)
        # *שימי לב:* בקוד הקודם שילבנו את ה-Grad-CAM בתוך predict_image והוא שומר קובץ בשם "heatmap_result.jpg"
        prediction_result = predict_image(final_img_clean)
        
        # 7. קריאת תמונת ה-Heatmap שנוצרה בדיסק בדיסק
        heatmap_img = cv2.imread("heatmap_result.jpg")
        
        # 8. המרת התמונות לפורמט ש-React מבין (Base64)
        final_base64 = image_to_base64(final_img_clean)
        heatmap_base64 = image_to_base64(heatmap_img) if heatmap_img is not None else None
        
        # 9. החזרת כל המידע ל-UI
        return JSONResponse({
            "status": "success",
            "prediction": prediction_result,
            "manipulatedImage": final_base64,
            "heatmapImage": heatmap_base64
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)