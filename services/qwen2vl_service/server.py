import io
import logging
import os
import datetime as dt
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from PIL import Image
import torch # type: ignore
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"time":"%(asctime)s", '
        '"level":"%(levelname)s", '
        '"service":"qwen2-vl", '
        '"message":"%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
PROMPT = os.getenv(
    "OCR_PROMPT",
    "Read the document image and return a clean markdown transcription that preserves structure.",
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1536"))
ATTN_IMPL = os.getenv("ATTN_IMPL", "eager")

processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForVision2Seq] = None
device: Optional[torch.device] = None
model_loaded = False
model_error: Optional[str] = None


def _load_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _decode_output(generated_ids, inputs):
    if processor is None:
        return ""

    input_ids = None
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")

    if input_ids is not None:
        trimmed = []
        for in_ids, out_ids in zip(input_ids, generated_ids):
            trimmed.append(out_ids[len(in_ids) :])
        generated_ids = trimmed

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return text[0] if text else ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device, model_loaded, model_error

    try:
        logger.info(f"Loading Qwen2-VL model '{MODEL_NAME}' in 4-bit...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

        load_kwargs = {
            "trust_remote_code": True,
            "quantization_config": _load_quant_config(),
            "device_map": "auto",
        }
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
        if ATTN_IMPL:
            load_kwargs["attn_implementation"] = ATTN_IMPL

        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            **load_kwargs,
        )
        model.eval()
        model_loaded = True
        logger.info("Qwen2-VL model loaded successfully.")

    except Exception as exc:
        model_loaded = False
        model_error = str(exc)
        logger.error("Failed to load Qwen2-VL model", exc_info=True)

    yield

    logger.info("Shutting down Qwen2-VL service.")


app = FastAPI(
    title="Qwen2-VL OCR Service",
    version="1.0.0",
    lifespan=lifespan,
)


class OCRResponse(BaseModel):
    text: str
    confidence: float = 1.0


def _prepare_inputs(image: Image.Image):
    assert processor is not None
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]

    if hasattr(processor, "apply_chat_template"):
        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_prompt],
            images=[image],
            return_tensors="pt",
        )
    else:
        inputs = processor(
            images=[image],
            text=[PROMPT],
            return_tensors="pt",
        )
    return inputs


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    if not model_loaded or model is None or processor is None:
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

        inputs = _prepare_inputs(image)
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        text = _decode_output(generated_ids, inputs)
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
        "service": "qwen2-vl",
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
