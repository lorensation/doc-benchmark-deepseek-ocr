import os
import json
import uuid
import datetime as dt
import requests

UPLOAD_DIR = "/app/data/uploads"
RESULTS_DIR = "/app/data/results"

DEEPSEEK_URL = "http://deepseek:9000/ocr"
TESSERACT_URL = "http://tesseract:9001/ocr"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def call_deepseek(filepath: str) -> str:
    """Send image to DeepSeek-OCR service."""
    with open(filepath, "rb") as f:
        resp = requests.post(DEEPSEEK_URL, files={"file": (os.path.basename(filepath), f)})
    resp.raise_for_status()
    return resp.json().get("text", "")


def call_tesseract(filepath: str) -> str:
    """Send image to Tesseract service."""
    with open(filepath, "rb") as f:
        resp = requests.post(TESSERACT_URL, files={"file": (os.path.basename(filepath), f)})
    resp.raise_for_status()
    return resp.json().get("text", "")


def compute_metrics(ds_text: str, ts_text: str) -> dict:
    """Simple benchmark metrics (can extend later)."""
    return {
        "length_deepseek": len(ds_text),
        "length_tesseract": len(ts_text),
        "char_difference": abs(len(ds_text) - len(ts_text)),
    }


def run_benchmark(filepath: str) -> dict:
    """Run DeepSeek + Tesseract on a single file."""
    run_id = str(uuid.uuid4())
    filename = os.path.basename(filepath)
    created_at = dt.datetime.utcnow().isoformat()

    ds_text = call_deepseek(filepath)
    ts_text = call_tesseract(filepath)

    metrics = compute_metrics(ds_text, ts_text)

    result = {
        "run_id": run_id,
        "filename": filename,
        "created_at": created_at,
        "deepseek_text": ds_text,
        "tesseract_text": ts_text,
        **metrics,
    }

    outpath = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark worker for OCR evaluation.")
    parser.add_argument("filepath", help="Path to an image inside data/uploads/")
    args = parser.parse_args()

    print("Running benchmark…")
    results = run_benchmark(args.filepath)
    print(json.dumps(results, indent=2))
