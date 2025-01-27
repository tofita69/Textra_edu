import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

# Use the full path to your image file here
image_path = "data /x0 newton3.PNG"
output_file = "data /img_extr.txt"

full_text = ""

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")

    # Enhance sharpness and contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)

    # Apply thresholding to make the text stand out
    img = img.point(lambda x: 0 if x < 140 else 255, '1')

    # Additional filters if needed
    img = img.filter(ImageFilter.SHARPEN)

    img.show()
    return img

# Open and process the image at the specified path
img = Image.open(image_path)
processed_img = preprocess_image(img)

# Extract text using pytesseract
text = pytesseract.image_to_string(processed_img)
full_text += f"\n\n--- Text from {os.path.basename(image_path)} ---\n{text}"

# Save the extracted text
with open(output_file, 'w') as f:
    f.write(full_text)

print(f"Text extracted and saved to {output_file}")
