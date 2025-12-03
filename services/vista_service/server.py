import io
import logging
import os
import datetime as dt
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from PIL import Image
import torch # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"time":"%(asctime)s", '
        '"level":"%(levelname)s", '
        '"service":"vista-ocr", '
        '"message":"%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("VISTA_MODEL_NAME", "mPLUG/DocOwl2"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
ATTN_IMPL = os.getenv("ATTN_IMPL", "eager")
PROMPT = os.getenv(
    "OCR_PROMPT",
    "You are an OCR system. Transcribe the document image to well-structured markdown, "
    "preserving layout, tables, and lists.",
)

tokenizer: Optional[AutoTokenizer] = None
model: Optional[torch.nn.Module] = None
device: Optional[torch.device] = None
model_loaded = False
model_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device, model_loaded, model_error

    try:
        logger.info(f"Loading VISTA-OCR model '{MODEL_NAME}'...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        
        # Initialize the processor inside the model (DocOwl2 pattern)
        model.init_processor(tokenizer=tokenizer, basic_image_size=504, crop_anchors="grid_12")
        
        if hasattr(model, "eval"):
            model.eval()
        model_loaded = True
        logger.info("VISTA-OCR model loaded successfully.")

    except Exception as exc:
        model_loaded = False
        model_error = str(exc)
        logger.error("Failed to load VISTA-OCR model", exc_info=True)

    yield

    logger.info("Shutting down VISTA-OCR service.")


app = FastAPI(
    title="VISTA-OCR Service",
    version="1.0.0",
    lifespan=lifespan,
)


class OCRResponse(BaseModel):
    text: str
    confidence: float = 1.0


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. Error: {model_error}",
        )

    try:
        logger.info(f"Received OCR request: {file.filename}")
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file received.",
            )

        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {err}",
            ) from err

        # Use temporary directory for image processing
        # DocOwl2 model.chat() expects file paths (strings), not PIL Image objects
        with tempfile.TemporaryDirectory() as tmp:
            # Save PIL image to temporary file
            temp_image_path = os.path.join(tmp, "temp_image.jpg")
            image.save(temp_image_path, format="JPEG")
            
            # Prepare messages following official DocOwl2 pattern
            images = [temp_image_path]  # List of file paths (strings)
            query = PROMPT
            messages = [{"role": "USER", "content": "<|image|>" * len(images) + query}]
            
            # Use model.chat() method as in official implementation
            with torch.inference_mode():
                text = model.chat(
                    messages=messages, 
                    images=images, 
                    tokenizer=tokenizer
                )
            
            logger.info(f"OCR complete ({len(text)} chars).")
            return OCRResponse(text=text, confidence=1.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"OCR processing error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR failed: {exc}",
        ) from exc


@app.get("/health")
def health_check():
    info = {
        "status": "healthy",
        "service": "vista-ocr",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "device": str(device),
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
    else:
        info["gpu"] = {"available": False}
    return info


@app.get("/ready")
def readiness_check():
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "model_loaded": False,
                "error": model_error,
                "timestamp": dt.datetime.utcnow().isoformat(),
            },
        )

    info = {
        "status": "ready",
        "model_loaded": True,
        "model": MODEL_NAME,
        "device": str(device),
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    if torch.cuda.is_available():
        info["gpu_memory"] = {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return info
