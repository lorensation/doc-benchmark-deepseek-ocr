from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import io
import torch
import os
import tempfile

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")

app = FastAPI(title="DeepSeek-OCR Service", version="0.1.0")

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    # CPU-only, no flash-attn, no CUDA
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True
    )
    model.eval()


class OCRResponse(BaseModel):
    text: str


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    """
    Accepts an image or PDF page (already rasterized).
    For simplicity here: assume it's an image.
    """
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    # DeepSeek-OCR API uses a custom infer() method added via trust_remote_code
    with tempfile.TemporaryDirectory() as tmpdir:
        out = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=None,         # we pass PIL instead below if supported
            image=image,             # custom param used by remote code
            output_path=tmpdir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False
        )

    # infer() usually returns a dict; we'll be defensive
    if isinstance(out, dict) and "text" in out:
        text = out["text"]
    else:
        text = str(out)

    return OCRResponse(text=text)
