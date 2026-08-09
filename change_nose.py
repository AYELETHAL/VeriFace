
import base64
import os
import time
import requests
from PIL import Image
from openai import OpenAI

API_KEY = r"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"


def prepare_mask(mask_path):
    """
    Black in mask → transparent → area to edit
    White in mask → opaque → keep original
    """
    print(f"Preparing edit mask from: {mask_path}")

    with Image.open(mask_path).convert("L") as mask:
        w, h = mask.size
        new_mask = Image.new("RGBA", (w, h), (255, 255, 255, 255))

        for y in range(h):
            for x in range(w):
                value = mask.getpixel((x, y))
                if value < 100:
                    new_mask.putpixel((x, y), (255, 255, 255, 0))
                else:
                    new_mask.putpixel((x, y), (255, 255, 255, 255))

        new_path = os.path.splitext(mask_path)[0] + "_editmask.png"
        new_mask.save(new_path, "PNG")

    print(f"Mask ready: {new_path}")
    return new_path


def convert_image_to_rgba(image_path):
    print(f"Converting image to RGBA: {image_path}")

    with Image.open(image_path) as img:
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        new_path = os.path.splitext(image_path)[0] + "_temp.png"
        img.save(new_path, "PNG")

    return new_path


def safe_remove(path):
    try:
        time.sleep(0.5)
        if path and os.path.exists(path):
            os.remove(path)
    except:
        pass


def change_nose(image_path, mask_path, output_path):
    client = OpenAI(api_key=API_KEY)

    image_temp = None
    mask_temp = None

    try:
        image_temp = convert_image_to_rgba(image_path)
        mask_temp = prepare_mask(mask_path)

        print("\nSending nose edit request...")

        prompt = (
            "Photorealistic facial edit. Modify the nose shape subtly and naturally. "
            "Keep realistic human anatomy, skin texture, pores, shadows, and lighting. "
            "Do not make the nose look plastic, airbrushed, or surgically altered. "
            "Blend changes smoothly with surrounding facial features. "
            "Preserve identity and facial proportions. "
            "No beauty filter effect, no smoothing, no glow."
        )

        with open(image_temp, "rb") as img_file, open(mask_temp, "rb") as mask_file:
            response = client.images.edit(
                model="gpt-image-2",   # or "gpt-image-1.5" if available
                image=img_file,
                mask=mask_file,
                prompt= (prompt),
                size="1024x1024",
            )


        # GPT Image models return b64_json by default
        b64_data = response.data[0].b64_json
        img_data = base64.b64decode(b64_data)

        print("Edit success. Downloading...")

        with open(output_path, "wb") as f:
            f.write(img_data)

        print(f"\n✅ Saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

    finally:
        safe_remove(image_temp)
        safe_remove(mask_temp)


# ================= RUN =================
if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\97258\project111Try\Engineering-Project\input_faces\DONE\WhatsApp Image 2026-01-28 at 11.18.38.jpeg"
    MASK_PATH = r"C:\Users\97258\project1\Engineering-Project\tryOut\WhatsApp Image 2026-01-28 at 11.18.38_nose_mask.jpg"
    OUTPUT_PATH = r"C:\Users\97258\project111Try\Engineering-Project\try_out_nose\output_ME.png"

    print("Starting nose edit...\n")

    result = change_nose(IMAGE_PATH, MASK_PATH, OUTPUT_PATH)

    if result:
        print("🎉 Done!")
    else:
        print("✗ Failed.")
