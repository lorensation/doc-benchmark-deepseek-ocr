from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import logging
import datetime as dt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "tesseract"}'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tesseract OCR Service", version="0.2.0")

class OCRResponse(BaseModel):
    text: str
    confidence: float = 1.0

@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing OCR request for file: {file.filename}")
        
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file received"
            )
        
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Extract text with Tesseract
        text = pytesseract.image_to_string(image)
        
        # Get confidence data (optional)
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [float(c) for c in data['conf'] if c != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        except:
            avg_confidence = 1.0
        
        logger.info(f"OCR completed for {file.filename}: {len(text)} characters extracted, confidence: {avg_confidence:.2f}")
        return OCRResponse(text=text, confidence=avg_confidence / 100.0)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "tesseract-ocr",
        "version": "0.2.0",
        "tesseract_version": pytesseract.get_tesseract_version(),
        "timestamp": dt.datetime.utcnow().isoformat()
    }


@app.get("/ready")
def readiness_check():
    """Readiness check - verifies Tesseract is accessible."""
    try:
        version = pytesseract.get_tesseract_version()
        return {
            "status": "ready",
            "tesseract_version": str(version),
            "timestamp": dt.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Tesseract readiness check failed: {e}")
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": dt.datetime.utcnow().isoformat()
        }, 503
