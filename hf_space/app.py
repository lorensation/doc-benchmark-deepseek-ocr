"""Streamlit UI for benchmarking DeepSeek-OCR vs Tesseract.

Features:
- Upload a single image with optional ground truth to run a benchmark.
- Run a quick batch benchmark on the bundled SROIE2019 dataset.
- Browse summaries of previous benchmark runs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Ensure we can import the worker package when running in HF Spaces
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from services.benchmark_worker import worker  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and configuration helpers
# ---------------------------------------------------------------------------
DATA_DIR = worker.DATA_DIR
UPLOAD_DIR = worker.UPLOAD_DIR
RESULTS_DIR = worker.RESULTS_DIR
SROIE_PATH = worker.SROIE_DATASET_PATH


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file) -> Path:
    """Persist an uploaded file to the uploads directory and return its path."""
    suffix = Path(file.name).suffix or ".png"
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with open(dest, "wb") as fh:
        fh.write(file.getbuffer())
    return dest


def _parse_ground_truth(gt_text: str, gt_json_file) -> Tuple[Optional[str], Optional[List[str]], Optional[Dict]]:
    """Normalize ground truth inputs from the UI."""
    ground_truth_text = gt_text.strip() if gt_text and gt_text.strip() else None
    ground_truth_lines = ground_truth_text.splitlines() if ground_truth_text else None

    ground_truth_fields = None
    if gt_json_file is not None:
        try:
            ground_truth_fields = json.loads(gt_json_file.getvalue().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Could not parse ground truth JSON: {exc}")

    return ground_truth_text, ground_truth_lines, ground_truth_fields


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_result_details(result: dict) -> None:
    """Pretty-print a single benchmark result."""
    st.success("Benchmark complete")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Run ID", result.get("run_id", "") or "-")
    meta_cols[1].metric("File", result.get("filename", "") or "-")
    meta_cols[2].metric("Created", result.get("created_at", "") or "-")

    st.caption(f"Saved to: {result.get('result_file', 'not saved')}")

    metrics = result.get("metrics", {})
    lengths = metrics.get("lengths") or {
        "deepseek": len(result.get("deepseek_text", "")),
        "tesseract": len(result.get("tesseract_text", "")),
    }
    char_diff = metrics.get("char_difference") or abs(lengths["deepseek"] - lengths["tesseract"])

    with st.container():
        cols = st.columns(3)
        cols[0].metric("DeepSeek length", lengths.get("deepseek", 0))
        cols[1].metric("Tesseract length", lengths.get("tesseract", 0))
        cols[2].metric("Char diff", char_diff)

    st.divider()

    deepseek_metrics = metrics.get("deepseek", {})
    tesseract_metrics = metrics.get("tesseract", {})

    if deepseek_metrics or tesseract_metrics:
        st.subheader("Quality metrics")
        mcols = st.columns(2)
        _render_model_metrics("DeepSeek-OCR", deepseek_metrics, mcols[0])
        _render_model_metrics("Tesseract", tesseract_metrics, mcols[1])

    st.subheader("Extracted text")
    text_cols = st.columns(2)
    text_cols[0].write("DeepSeek-OCR")
    text_cols[0].code(result.get("deepseek_text", ""), language="markdown")

    text_cols[1].write("Tesseract")
    text_cols[1].code(result.get("tesseract_text", ""), language="markdown")

    st.download_button(
        "Download result JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"{result.get('run_id', 'benchmark')}.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_model_metrics(name: str, payload: dict, container) -> None:
    """Render key metrics for a single model."""
    if not payload:
        container.info("No metrics captured.")
        return

    text_metrics = payload.get("text", {})
    llm_metrics = payload.get("llm", {})

    if text_metrics:
        cer = text_metrics.get("cer", {}).get("cer")
        wer = text_metrics.get("wer", {}).get("wer")
        ser = text_metrics.get("ser", {}).get("ser")
        if cer is not None or wer is not None or ser is not None:
            container.metric(f"{name} CER", f"{cer:.2f}%" if cer is not None else "-")
            container.metric(f"{name} WER", f"{wer:.2f}%" if wer is not None else "-")
            container.metric(f"{name} SER", f"{ser:.2f}%" if ser is not None else "-")

    if llm_metrics:
        token_eff = llm_metrics.get("token_efficiency", {})
        fext = llm_metrics.get("field_extraction", {})
        rows = []
        if token_eff:
            rows.append(f"Token density: {token_eff.get('token_density', 0):.2f}")
            rows.append(f"Chars/token: {token_eff.get('chars_per_token', 0):.2f}")
        if fext:
            rows.append(f"Field F1: {fext.get('field_f1', 0):.2f}")
            rows.append(f"Field recall: {fext.get('field_recall', 0):.2f}")
        if rows:
            container.write("\n".join(rows))


def _render_batch_summary(summary: dict, results: List[dict]) -> None:
    st.success("SROIE benchmark finished")
    st.write(
        f"Batch ID: `{summary.get('batch_run_id', '-')}`  | "
        f"Split: `{summary.get('split', '-')}`  | "
        f"Samples: `{summary.get('count', 0)}`"
    )
    if summary.get("result_files"):
        st.caption(f"Stored in: {Path(summary['result_files'][0]).parent}")

    if results:
        st.subheader("Individual results")
        for item in results:
            with st.expander(f"{item.get('filename', 'sample')} — {item.get('run_id', '')}"):
                _render_result_details(item)
    else:
        st.info("No individual results were returned.")


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------
def page_single_run() -> None:
    st.header("Upload and Benchmark")
    st.write(
        "Upload an image and optional ground truth. The worker will call DeepSeek-OCR "
        "and Tesseract services, compute metrics, and save the result to the workspace."
    )

    upload = st.file_uploader(
        "Image file",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )
    gt_text = st.text_area(
        "Ground truth text (optional)",
        placeholder="Paste expected text. Leave empty to auto-use SROIE ground truth when the file belongs to the dataset.",
        height=120,
    )
    gt_json = st.file_uploader(
        "Ground truth fields JSON (optional)",
        type=["json"],
        key="gt_json_single",
        help='Optional key-value ground truth (e.g. {"total": "123.00", "date": "2024-01-01"}).',
    )

    if upload and st.button("Run benchmark", type="primary", use_container_width=True):
        _ensure_dirs()
        with st.spinner("Running benchmark..."):
            img_path = _save_upload(upload)
            gt_text_val, gt_lines, gt_fields = _parse_ground_truth(gt_text, gt_json)
            result = worker.run_benchmark(
                str(img_path),
                ground_truth_text=gt_text_val,
                ground_truth_lines=gt_lines,
                ground_truth_fields=gt_fields,
            )
        _render_result_details(result)
    elif not upload:
        st.info("Upload an image to begin.")


def _count_sroie_samples(split: str) -> int:
    patterns = ("*.jpg", "*.jpeg", "*.png")
    img_dir = Path(SROIE_PATH) / split / "img"
    total = 0
    for pattern in patterns:
        total += len(list(img_dir.glob(pattern)))
    return total


def page_sroie_batch() -> None:
    st.header("SROIE2019 Batch Benchmark")
    st.write(
        "Run a quick benchmark on the bundled SROIE2019 dataset. "
        "Ground truth is auto-loaded from the dataset annotations."
    )

    if not SROIE_PATH.exists():
        st.error(f"SROIE dataset not found at {SROIE_PATH}. Add the dataset before running.")
        return

    split = st.selectbox("Split", ["train", "test"], index=0)
    total = _count_sroie_samples(split)
    st.caption(f"Found {total} images in {SROIE_PATH / split / 'img'}")
    limit = st.slider("Number of samples to run", min_value=1, max_value=max(1, total) if total else 1, value=min(3, total or 1))

    if st.button("Run SROIE benchmark", type="primary", use_container_width=True):
        _ensure_dirs()
        with st.spinner("Processing SROIE samples..."):
            payload = worker.run_sroie_samples(split=split, limit=limit)
        summary = payload.get("summary", {})
        results = payload.get("results", [])
        _render_batch_summary(summary, results)


def _is_batch_summary(data: dict) -> bool:
    return "batch_run_id" in data or ("summary" in data and isinstance(data["summary"], dict))


def _load_result_file(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _collect_history(max_files: int = 200) -> Tuple[List[dict], List[dict]]:
    run_items: List[dict] = []
    batch_items: List[dict] = []

    files = sorted(RESULTS_DIR.rglob("*.json"), key=os.path.getmtime, reverse=True)
    for path in files[:max_files]:
        data = _load_result_file(path)
        if not data:
            continue

        if _is_batch_summary(data):
            summary = data.get("summary", data)
            batch_items.append(
                {
                    "path": str(path),
                    "batch_run_id": summary.get("batch_run_id"),
                    "split": summary.get("split"),
                    "count": summary.get("count"),
                    "saved": summary.get("result_files", []),
                }
            )
        elif "run_id" in data:
            run_items.append(
                {
                    "path": str(path),
                    "run_id": data.get("run_id"),
                    "filename": data.get("filename"),
                    "created_at": data.get("created_at"),
                    "ground_truth": data.get("ground_truth_available", bool(data.get("metrics", {}).get("deepseek", {}).get("text"))),
                    "char_diff": data.get("metrics", {}).get("char_difference"),
                }
            )

    return run_items, batch_items


def page_history() -> None:
    st.header("Benchmark History")
    st.write("Browse results saved under the workspace `data/results` directory.")

    run_items, batch_items = _collect_history()

    if batch_items:
        st.subheader("Batch summaries")
        st.table(batch_items)
    else:
        st.info("No batch summaries found.")

    st.subheader("Individual runs")
    if not run_items:
        st.info("No individual runs found.")
        return

    st.dataframe(run_items, use_container_width=True)

    selected = st.selectbox(
        "Inspect a run",
        options=run_items,
        format_func=lambda item: f"{item['run_id']} — {item['filename']}",
    )

    if selected:
        data = _load_result_file(Path(selected["path"]))
        if data:
            _render_result_details(data)
        else:
            st.error("Could not load the selected result file.")


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="DeepSeek OCR Benchmark", layout="wide")
    st.title("DeepSeek-OCR Benchmark Space")
    st.caption("Compare DeepSeek-OCR against Tesseract, run SROIE batches, and review saved results.")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Upload benchmark",
            "SROIE dataset benchmark",
            "Benchmark history",
        ],
    )
    st.sidebar.markdown(f"**Data dir:** `{DATA_DIR}`")
    st.sidebar.markdown(f"**Results dir:** `{RESULTS_DIR}`")
    st.sidebar.markdown(f"**Uploads dir:** `{UPLOAD_DIR}`")
    st.sidebar.markdown(f"**SROIE path:** `{SROIE_PATH}`")

    if page == "Upload benchmark":
        page_single_run()
    elif page == "SROIE dataset benchmark":
        page_sroie_batch()
    else:
        page_history()


if __name__ == "__main__":
    main()
