
import cv2
import os
import numpy as np

# Settings
INPUT_FOLDER = r"C:\Users\97258\engineering_try_2\Engineering-Project\output_faces_change_eye_color"
OUTPUT_FOLDER = r"C:\Users\97258\engineering_try_2\Engineering-Project\output_faces_change_eye_color"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to find and remove white padding
def remove_white_padding(image):
    """
    Remove white padding from an image by cropping
    """
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where white pixels are 255 and others are 0
    # White is around 255, so we look for pixels close to 255
    lower_white = np.array([200])
    upper_white = np.array([255])
    
    # Threshold to create binary mask (non-white pixels)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect the main content area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the actual image content)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding margin to keep content
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        return cropped
    else:
        # If no contour found, return original
        return image


# Alternative function - simpler approach (find non-white boundaries)
def remove_padding_simple(image):
    """
    Remove padding by finding the bounding box of non-white pixels
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find pixels that are not white (threshold at 245)
    non_white = gray < 245
    
    # Get rows and columns with non-white pixels
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    
    if rows.any() and cols.any():
        # Get boundaries
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add small margin
        margin = 5
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        
        # Crop
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
    else:
        return image


# Process all images
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
processed_count = 0
total_count = 0

#print("=" * 60)
#print("Removing White Padding from Images")
#print("=" * 60)
#print(f"Input folder: {INPUT_FOLDER}")
#print(f"Output folder: {OUTPUT_FOLDER}")
#print()

for img_name in os.listdir(INPUT_FOLDER):
    if not img_name.lower().endswith(image_extensions):
        continue
    
    total_count += 1
    
    try:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"❌ Failed to read: {img_name}")
            continue
        
        # Remove padding
        cropped = remove_padding_simple(image)
        
        # Save cropped image
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, cropped)
        
        original_size = image.shape
        new_size = cropped.shape
        #print(f"✓ {img_name}: {original_size[1]}x{original_size[0]} → {new_size[1]}x{new_size[0]}")
        processed_count += 1
        
    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")

#print()
#print("=" * 60)
#print(f"Processing complete!")
#print(f"Successfully processed: {processed_count}/{total_count} images")
#print(f"Output folder: {OUTPUT_FOLDER}")
#print("=" * 60)
