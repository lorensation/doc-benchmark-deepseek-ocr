from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
import json
import datetime as dt
import requests

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/data/uploads")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/data/results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEEPSEEK_URL = "http://deepseek:9000/ocr"
TESSERACT_URL = "http://tesseract:9001/ocr"

app = FastAPI(title="Doc Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BenchmarkResult(BaseModel):
    run_id: str
    filename: str
    deepseek_text: str
    tesseract_text: str
    length_deepseek: int
    length_tesseract: int
    created_at: str


class RunListItem(BaseModel):
    run_id: str
    filename: str
    created_at: str


@app.post("/upload_and_benchmark", response_model=BenchmarkResult)
async def upload_and_benchmark(file: UploadFile = File(...)):
    # Save file
    run_id = str(uuid.uuid4())
    filename = f"{run_id}_{file.filename}"
    dest_path = os.path.join(UPLOAD_DIR, filename)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Call DeepSeek
    with open(dest_path, "rb") as f:
        ds_resp = requests.post(DEEPSEEK_URL, files={"file": (filename, f)})
    ds_resp.raise_for_status()
    ds_text = ds_resp.json().get("text", "")

    # Call Tesseract
    with open(dest_path, "rb") as f:
        ts_resp = requests.post(TESSERACT_URL, files={"file": (filename, f)})
    ts_resp.raise_for_status()
    ts_text = ts_resp.json().get("text", "")

    created_at = dt.datetime.utcnow().isoformat()

    result = {
        "run_id": run_id,
        "filename": filename,
        "deepseek_text": ds_text,
        "tesseract_text": ts_text,
        "length_deepseek": len(ds_text),
        "length_tesseract": len(ts_text),
        "created_at": created_at,
    }

    out_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return BenchmarkResult(**result)


@app.get("/runs", response_model=List[RunListItem])
def list_runs():
    items: List[RunListItem] = []
    for fn in os.listdir(RESULTS_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fn), "r", encoding="utf-8") as f:
            d = json.load(f)
        items.append(
            RunListItem(
                run_id=d["run_id"],
                filename=d["filename"],
                created_at=d["created_at"],
            )
        )
    # sort by date desc
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items


@app.get("/runs/{run_id}", response_model=BenchmarkResult)
def get_run(run_id: str):
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return BenchmarkResult(
            run_id=run_id,
            filename="",
            deepseek_text="",
            tesseract_text="",
            length_deepseek=0,
            length_tesseract=0,
            created_at="",
        )
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return BenchmarkResult(**d)
