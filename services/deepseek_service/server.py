import io
import os
import logging
import tempfile
import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoTokenizer # type: ignore
import torch
import glob


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"time":"%(asctime)s", '
        '"level":"%(levelname)s", '
        '"service":"deepseek", '
        '"message":"%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)


# ============================================================
# Global State
# ============================================================
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")

tokenizer = None
model = None
device = None
model_loaded = False
model_error = None


# ============================================================
# FastAPI Lifespan (Startup / Shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device, model_loaded, model_error

    try:
        logger.info(f"Loading DeepSeek-OCR model '{MODEL_NAME}'...")

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"Total GPU memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # ----------------------
        # Load tokenizer
        # ----------------------
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        logger.info("Tokenizer loaded.")

        # ----------------------
        # Load model
        # ----------------------
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,  # Use float32 for compatibility
            device_map="auto",
            attn_implementation="flash_attention_2",  # requires flash-attn
        )
        model.eval()

        model_loaded = True
        logger.info("DeepSeek-OCR model loaded successfully.")

    except Exception as e:
        model_loaded = False
        model_error = str(e)
        logger.error("Model failed to load", exc_info=True)

    yield

    logger.info("Shutting down DeepSeek-OCR service.")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="DeepSeek-OCR Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Schemas
# ============================================================
class OCRResponse(BaseModel):
    text: str
    confidence: float = 1.0


def _read_first_text_file(directory: str) -> str | None:
    """Find and read the first markdown/text-like file in a directory (recursive)."""
    patterns = [
        "**/*.mmd",
        "**/*.md",
        "**/*.txt",
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))

    if not files:
        return None

    # deterministic order
    files = sorted(files)
    chosen = files[0]
    logger.info(f"Found output file: {chosen}")
    with open(chosen, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# OCR Endpoint
# ============================================================
@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    """OCR endpoint for image files."""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. Error: {model_error}",
        )

    try:
        logger.info(f"Received OCR request: {file.filename}")

        # Read file bytes
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file received.",
            )

        # Convert to image
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            logger.error(f"Invalid image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}",
            )

        # DeepSeek-OCR prompt
        prompt = "<image>\n<|grounding|>Convert the document to markdown."

        # Run inference
        with tempfile.TemporaryDirectory() as tmp:
            # Save PIL image to temporary file
            temp_image_path = os.path.join(tmp, "temp_image.jpg")
            image.save(temp_image_path, format="JPEG")

            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=tmp,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,  # Save results to file
                test_compress=False,
            )

            text: str | None = None

            # Direct return content
            if isinstance(result, dict):
                text = result.get("text")
                if text is None and "output_file" in result:
                    maybe_path = result["output_file"]
                    if os.path.exists(maybe_path):
                        with open(maybe_path, "r", encoding="utf-8") as f:
                            text = f.read()
            elif result is not None:
                text = str(result)

            # Fallback: scan output directory
            if not text:
                text = _read_first_text_file(tmp)

            if not text:
                text = "OCR completed but no output file found"

        logger.info(f"OCR complete ({len(text)} chars).")

        return OCRResponse(text=text, confidence=1.0)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"OCR processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR failed: {str(e)}",
        )


# ============================================================
# Health Checks
# ============================================================
@app.get("/health")
def health_check():
    info = {
        "status": "healthy",
        "service": "deepseek-ocr",
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
