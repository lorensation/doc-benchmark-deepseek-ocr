import datetime as dt
import json
import json as json_module
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException

from metrics.llm_integration import calculate_field_extraction_f1, calculate_token_efficiency
from metrics.text_correctness import calculate_all_text_metrics, calculate_ser
from utils import load_sroie_dataset


def _guess_data_dir() -> Path:
    """Find an existing data directory, preferring a mounted /app/data."""
    env_data = os.getenv("DATA_DIR")
    if env_data:
        return Path(env_data)

    here = Path(__file__).resolve().parent
    repo_root_candidate = here.parent.parent / "data"
    candidates = [
        Path.cwd() / "data",
        Path("/app/data"),
        repo_root_candidate,
        here.parent / "data",
        here / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("/app/data")


DATA_DIR = _guess_data_dir()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", DATA_DIR / "results"))
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", DATA_DIR / "datasets"))
SROIE_DATASET_PATH = Path(os.getenv("SROIE_DATASET_PATH", DATASETS_DIR / "SROIE2019"))

IN_DOCKER = Path("/.dockerenv").exists()

# External OCR endpoints with host-aware fallbacks
DEFAULT_DEEPSEEK = "http://deepseek:9000/ocr" if IN_DOCKER else "http://localhost:9000/ocr"
DEFAULT_TESSERACT = "http://tesseract:9001/ocr" if IN_DOCKER else "http://localhost:9001/ocr"
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", DEFAULT_DEEPSEEK)
TESSERACT_URL = os.getenv("TESSERACT_URL", DEFAULT_TESSERACT)
DEFAULT_VISTA = "http://vista:9002/ocr" if IN_DOCKER else "http://localhost:9002/ocr"
DEFAULT_HUNYUAN = "http://hunyuan:9003/ocr" if IN_DOCKER else "http://localhost:9003/ocr"
DEFAULT_QWEN = "http://qwen2vl:9004/ocr" if IN_DOCKER else "http://localhost:9004/ocr"
VISTA_URL = os.getenv("VISTA_URL", DEFAULT_VISTA)
HUNYUAN_URL = os.getenv("HUNYUAN_URL", DEFAULT_HUNYUAN)
QWEN_URL = os.getenv("QWEN_URL", DEFAULT_QWEN)

OCR_ENDPOINTS = {
    "deepseek": DEEPSEEK_URL,
    "tesseract": TESSERACT_URL,
    "vista": VISTA_URL,
    "hunyuan": HUNYUAN_URL,
    "qwen2vl": QWEN_URL,
}

OCR_SERVICE_LABELS = {
    "deepseek": "DeepSeek-OCR",
    "tesseract": "Tesseract",
    "vista": "VISTA-OCR",
    "hunyuan": "HunyuanOCR",
    "qwen2vl": "Qwen2-VL",
}
SERVICE_ORDER = ["deepseek", "tesseract", "vista", "hunyuan", "qwen2vl"]

WORKER_PORT = int(os.getenv("WORKER_PORT", "9100"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Benchmark Worker", version="0.1.0")


def call_ocr_service(service: str, filepath: str) -> str:
    """Send image to one of the OCR microservices."""
    url = OCR_ENDPOINTS[service]
    with open(filepath, "rb") as file:
        resp = requests.post(url, files={"file": (os.path.basename(filepath), file)})
    resp.raise_for_status()
    return resp.json().get("text", "")


def _match_fields_in_text(text: str, ground_truth_fields: Dict[str, str]) -> Dict[str, str]:
    """Naively match known ground-truth fields inside OCR text."""
    matched: Dict[str, str] = {}
    lowered = text.lower()
    for key, value in ground_truth_fields.items():
        value_str = str(value).strip()
        if value_str and value_str.lower() in lowered:
            matched[key] = value_str
    return matched


def compute_model_metrics(
    ocr_text: str,
    ground_truth_text: Optional[str] = None,
    ground_truth_lines: Optional[List[str]] = None,
    ground_truth_fields: Optional[Dict[str, str]] = None,
) -> dict:
    """Compute metrics for a single OCR output."""
    text_metrics = None
    if ground_truth_text:
        text_metrics = calculate_all_text_metrics(ocr_text, ground_truth_text)
        if ground_truth_lines is not None:
            text_metrics["ser"] = calculate_ser(ocr_text.splitlines(), ground_truth_lines)

    llm_metrics = {"token_efficiency": calculate_token_efficiency(ocr_text)}

    if ground_truth_fields:
        extracted_fields = _match_fields_in_text(ocr_text, ground_truth_fields)
        llm_metrics["field_extraction"] = calculate_field_extraction_f1(extracted_fields, ground_truth_fields)

    metrics = {"llm": llm_metrics}
    if text_metrics is not None:
        metrics["text"] = text_metrics

    return metrics


def _parse_box_file(box_path: Path) -> List[str]:
    """Parse SROIE box file into text lines."""
    lines: List[str] = []
    if not box_path.exists():
        return lines
    with box_path.open(encoding="utf-8") as file:
        for raw_line in file:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parts = raw_line.split(",")
            if len(parts) < 9:
                continue
            text = ",".join(parts[8:]).strip()
            if text:
                lines.append(text)
    return lines


def _load_ground_truth_for_file(image_path: str) -> dict:
    """If the image belongs to SROIE, load its ground truth fields and lines."""
    img_path = Path(image_path).resolve()
    stem = img_path.stem

    dataset_root = None
    for parent in img_path.parents:
        if parent.name == "SROIE2019":
            dataset_root = parent
            break
    if not dataset_root:
        return {}

    split_dir = img_path.parent.parent if img_path.parent.name == "img" else None
    if not split_dir:
        return {}
    split = split_dir.name

    entity_path = dataset_root / split / "entities" / f"{stem}.txt"
    box_path = dataset_root / split / "box" / f"{stem}.txt"

    fields: Dict[str, str] = {}
    if entity_path.exists():
        try:
            with entity_path.open(encoding="utf-8") as ef:
                fields = json_module.load(ef)
        except json_module.JSONDecodeError:
            fields = {}

    text_lines = _parse_box_file(box_path)
    ground_truth_text = "\n".join(text_lines) if text_lines else None

    return {
        "ground_truth_text": ground_truth_text,
        "ground_truth_lines": text_lines if text_lines else None,
        "ground_truth_fields": fields if fields else None,
    }


def compute_metrics(
    outputs: Dict[str, str],
    ground_truth_text: Optional[str] = None,
    ground_truth_lines: Optional[List[str]] = None,
    ground_truth_fields: Optional[Dict[str, str]] = None,
) -> dict:
    """Compute metrics for all OCR outputs."""
    lengths = {name: len(text) for name, text in outputs.items()}
    metrics: Dict[str, object] = {"lengths": lengths}

    if ground_truth_text:
        model_metrics = {
            name: compute_model_metrics(text, ground_truth_text, ground_truth_lines, ground_truth_fields)
            for name, text in outputs.items()
        }
        metrics["models"] = model_metrics
        metrics.update(model_metrics)

    if "deepseek" in outputs:
        baseline = outputs["deepseek"]
        metrics["char_difference_vs_deepseek"] = {
            name: abs(len(text) - len(baseline))
            for name, text in outputs.items()
            if name != "deepseek"
        }

    return metrics


def run_benchmark(
    filepath: str,
    ground_truth_text: Optional[str] = None,
    ground_truth_lines: Optional[List[str]] = None,
    ground_truth_fields: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run all OCR engines on a single file."""
    # Auto-load SROIE ground truth if not provided
    if not any([ground_truth_text, ground_truth_lines, ground_truth_fields]):
        gt = _load_ground_truth_for_file(filepath)
        ground_truth_text = gt.get("ground_truth_text")
        ground_truth_lines = gt.get("ground_truth_lines")
        ground_truth_fields = gt.get("ground_truth_fields")

    run_id = str(uuid.uuid4())
    filename = os.path.basename(filepath)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    outputs: Dict[str, str] = {}
    for service in SERVICE_ORDER:
        outputs[service] = call_ocr_service(service, filepath)

    metrics = compute_metrics(outputs, ground_truth_text, ground_truth_lines, ground_truth_fields)

    result = {
        "run_id": run_id,
        "filename": filename,
        "created_at": created_at,
        "ground_truth_available": bool(ground_truth_text),
        "deepseek_text": outputs.get("deepseek", ""),
        "tesseract_text": outputs.get("tesseract", ""),
        "vista_text": outputs.get("vista", ""),
        "hunyuan_text": outputs.get("hunyuan", ""),
        "qwen2vl_text": outputs.get("qwen2vl", ""),
        "outputs": outputs,
        "metrics": metrics,
    }

    target_dir = Path(output_dir) if output_dir else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    outpath = target_dir / f"benchmark_{timestamp}_{run_id}.json"
    with open(outpath, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    result["result_file"] = str(outpath)

    return result


def run_sroie_samples(split: str = "train", limit: int = 3) -> dict:
    """Run benchmarks against SROIE samples for quick validation."""
    samples = load_sroie_dataset(str(SROIE_DATASET_PATH), split=split, limit=limit)
    if not samples:
        raise FileNotFoundError(f"No SROIE samples found in {SROIE_DATASET_PATH}/{split}")

    batch_run_id = str(uuid.uuid4())
    batch_dir = RESULTS_DIR / f"sroie_{split}_{batch_run_id}"
    os.makedirs(batch_dir, exist_ok=True)

    results = []
    for sample in samples:
        result = run_benchmark(
            str(sample.image_path),
            ground_truth_text=sample.full_text,
            ground_truth_lines=sample.text_lines,
            ground_truth_fields=sample.fields,
            output_dir=batch_dir,
        )
        results.append(result)

    result_files = []
    for item in results:
        if "result_file" in item:
            result_files.append(item["result_file"])
            continue

        candidates = list(batch_dir.glob(f"*{item['run_id']}*.json"))
        if candidates:
            result_files.append(str(candidates[0]))
        else:
            result_files.append(str(batch_dir / f"{item['run_id']}.json"))

    summary = {
        "batch_run_id": batch_run_id,
        "split": split,
        "count": len(results),
        "result_files": result_files,
    }

    summary_path = batch_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {"summary": summary, "results": results}


def _result_path(run_id: str) -> Path:
    return RESULTS_DIR / f"{run_id}.json"


def get_result(run_id: str) -> dict:
    path = _result_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Result {run_id} not found")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def list_results(limit: int = 50) -> List[str]:
    run_files = sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    return [f.stem for f in run_files[:limit]]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "deepseek_url": DEEPSEEK_URL,
        "tesseract_url": TESSERACT_URL,
        "vista_url": VISTA_URL,
        "hunyuan_url": HUNYUAN_URL,
        "qwen2vl_url": QWEN_URL,
    }


@app.get("/results")
def api_list_results(limit: int = 50):
    return {"results": list_results(limit=limit)}


@app.get("/results/{run_id}")
def api_get_result(run_id: str):
    try:
        return get_result(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark worker for OCR evaluation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filepath", help="Path to an image to benchmark.")
    group.add_argument(
        "--run-sroie-samples",
        action="store_true",
        help="Run a quick benchmark on SROIE samples (uses --split/--limit).",
    )
    group.add_argument(
        "--serve",
        action="store_true",
        help="Run an HTTP server to expose results.",
    )
    parser.add_argument("--split", default="train", help="Dataset split for SROIE runs.")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of SROIE samples.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server when using --serve.")
    parser.add_argument("--port", type=int, default=WORKER_PORT, help="Port for HTTP server when using --serve.")
    args = parser.parse_args()

    if args.run_sroie_samples:
        print("Running SROIE sample benchmarks...")
        payload = run_sroie_samples(split=args.split, limit=args.limit)
        print(json.dumps(payload["summary"], indent=2))
    elif args.serve:
        import uvicorn

        uvicorn.run("worker:app", host=args.host, port=args.port, reload=False)
    else:
        print("Running benchmark...")
        results = run_benchmark(args.filepath)
        print(json.dumps(results, indent=2))
