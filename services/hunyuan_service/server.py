import io
import logging
import os
import datetime as dt
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"time":"%(asctime)s", '
        '"level":"%(levelname)s", '
        '"service":"hunyuan-ocr", '
        '"message":"%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv("MODEL_NAME", "tencent/HunyuanOCR")
PROMPT = os.getenv(
    "OCR_PROMPT",
    "检测并识别图片中的文字，将文本内容用 Markdown 输出，并保持表格和段落结构。",
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
ATTN_IMPL = os.getenv("ATTN_IMPL", "eager")

processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForCausalLM] = None
device: Optional[torch.device] = None
model_loaded = False
model_error: Optional[str] = None


def _clean_repeated_substrings(text: str) -> str:
    """Trim pathological repetitions sometimes produced in long generations."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


def _decode_output(generated_ids, inputs):
    if processor is None:
        return ""

    input_ids = None
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids") or inputs.get("inputs")

    if input_ids is not None:
        trimmed = []
        for in_ids, out_ids in zip(input_ids, generated_ids):
            trimmed.append(out_ids[len(in_ids) :])
        generated_ids = trimmed

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output = text[0] if text else ""
    return _clean_repeated_substrings(output)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device, model_loaded, model_error

    try:
        logger.info(f"Loading HunyuanOCR model '{MODEL_NAME}'...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation=ATTN_IMPL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        model_loaded = True
        logger.info("HunyuanOCR model loaded successfully.")

    except Exception as exc:
        model_loaded = False
        model_error = str(exc)
        logger.error("Failed to load HunyuanOCR model", exc_info=True)

    yield

    logger.info("Shutting down HunyuanOCR service.")


app = FastAPI(
    title="HunyuanOCR Service",
    version="1.0.0",
    lifespan=lifespan,
)


class OCRResponse(BaseModel):
    text: str
    confidence: float = 1.0


def _prepare_inputs(image: Image.Image):
    assert processor is not None
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text],
        images=image,
        padding=True,
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
                do_sample=False,
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
        "service": "hunyuan-ocr",
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
