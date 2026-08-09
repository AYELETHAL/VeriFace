import base64

from openai import OpenAI
import requests
from PIL import Image
import os

# ============= CONFIGURATION =============
API_KEY = r"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"
IMAGE_PATH = r"tryIn\000002.jpg"
MASK_PATH = r"tryOut\000002_eyebrows_mask.jpg"
NEW_STYLE = "thick black eyebrows" 
OUTPUT_PATH = r"tryOut\000002_edited_eyebrows.png"
# =========================================

def prepare_eyebrow_mask(mask_path):
    """
    DALL-E 2 edits areas where alpha = 0 (Transparent).
    This function converts your black eyebrow mask to transparent.
    """
    print(f"Preparing eyebrow edit mask...")
    with Image.open(mask_path).convert("L") as mask:
        w, h = mask.size
        # Create a white background RGBA image
        new_mask = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        
        for y in range(h):
            for x in range(w):
                value = mask.getpixel((x, y))
                # Value < 100 means the black eyebrow detection
                if value < 100:  
                    new_mask.putpixel((x, y), (255, 255, 255, 0)) # Transparent = Edit Area
                else:
                    new_mask.putpixel((x, y), (255, 255, 255, 255)) # Opaque = Keep Original
        
        new_path = os.path.splitext(mask_path)[0] + "_processed_mask.png"
        new_mask.save(new_path, "PNG")
    return new_path

def convert_to_rgba(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        new_path = os.path.splitext(image_path)[0] + "_rgba_temp.png"
        img.save(new_path, "PNG")
    return new_path

def change_eyebrow_style(image_path, mask_path, style_desc, output_path):
    client = OpenAI(api_key=API_KEY)
    image_temp = None
    mask_temp = None

    try:
        image_temp = convert_to_rgba(image_path)
        mask_temp = prepare_eyebrow_mask(mask_path)

        print(f"Sending Eyebrow Edit request: {style_desc}...")

  
        prompt_text = f"Photorealistic edit. Change ONLY the eyebrows to {NEW_STYLE}. Keep face, skin, lighting and photo identical."
            
        with open(image_temp, "rb") as img_file, open(mask_temp, "rb") as mask_file:
            response = client.images.edit(
                model="gpt-image-2",   # or "gpt-image-1.5" if available
                image=img_file,
                mask=mask_file,
                prompt= (prompt_text),
                size="1024x1024",
            )


        # GPT Image models return b64_json by default
        b64_data = response.data[0].b64_json
        img_data = base64.b64decode(b64_data)
        
        with open(output_path, "wb") as f:
            f.write(img_data)

        # Resize back to original dimensions
        with Image.open(image_path) as original:
            orig_w, orig_h = original.size

        # After downloading the img_data from the API:
        with open(output_path, "wb") as f:
            f.write(img_data)

        # Resize the saved result back to original dimensions
        with Image.open(output_path) as edited_img:
            final_img = edited_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            final_img.save(output_path)

        print(f"✅ Success! Saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        for f in [image_temp, mask_temp]:
            if f and os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    change_eyebrow_style(IMAGE_PATH, MASK_PATH, NEW_STYLE, OUTPUT_PATH)