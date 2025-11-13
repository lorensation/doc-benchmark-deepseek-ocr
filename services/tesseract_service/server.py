from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io

app = FastAPI(title="Tesseract OCR Service", version="0.1.0")

class OCRResponse(BaseModel):
    text: str

@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    text = pytesseract.image_to_string(image)
    return OCRResponse(text=text)
