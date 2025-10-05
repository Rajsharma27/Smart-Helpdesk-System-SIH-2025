import base64
import io
import logging
from PIL import Image
import pytesseract

def process_image(image_data: str) -> str:
    """
    Decodes base64 image and runs OCR (pytesseract).
    Returns extracted text.
    """
    try:
        # handle data URI prefix (e.g., "data:image/png;base64,iVBORw...")
        image_bytes = base64.b64decode(image_data.split(",")[-1])
        image = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip() or "No readable text found in screenshot."
    except Exception as e:
        logging.error(f"Image processing failed: {e}", exc_info=True)
        return "Error: Unable to process screenshot."
