"""LLM integration metrics for OCR evaluation."""

import json
from typing import Dict, Optional

from jsonschema import ValidationError, validate

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency optional at runtime
    TIKTOKEN_AVAILABLE = False


def calculate_field_extraction_f1(extracted_fields: Dict[str, str], ground_truth_fields: Dict[str, str]) -> dict:
    """Calculate precision/recall/F1 for structured field extraction."""
    gt_keys = set(ground_truth_fields.keys())
    ext_keys = set(extracted_fields.keys())

    correct = [
        k
        for k in gt_keys & ext_keys
        if str(extracted_fields[k]).strip().lower() == str(ground_truth_fields[k]).strip().lower()
    ]

    precision = len(correct) / len(ext_keys) if ext_keys else 0
    recall = len(correct) / len(gt_keys) if gt_keys else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "correct_fields": correct,
        "missing_fields": list(gt_keys - ext_keys),
        "incorrect_fields": [k for k in gt_keys & ext_keys if k not in correct],
        "extra_fields": list(ext_keys - gt_keys),
    }


def validate_json_output(ocr_output: str, schema: dict) -> dict:
    """Validate OCR structured output against JSON schema."""
    try:
        parsed = json.loads(ocr_output)
        is_valid = True
        parsing_error: Optional[str] = None
    except json.JSONDecodeError as exc:
        return {
            "is_valid": False,
            "is_schema_compliant": False,
            "parsing_error": str(exc),
            "validation_errors": [],
            "completeness_score": 0.0,
            "field_count": 0,
        }

    try:
        validate(instance=parsed, schema=schema)
        is_schema_compliant = True
        validation_errors = []
    except ValidationError as exc:
        is_schema_compliant = False
        validation_errors = [str(exc)]

    required_fields = schema.get("required", [])
    present_fields = sum(1 for field in required_fields if field in parsed)
    completeness = present_fields / len(required_fields) if required_fields else 1.0

    return {
        "is_valid": is_valid,
        "is_schema_compliant": is_schema_compliant,
        "parsing_error": parsing_error,
        "validation_errors": validation_errors,
        "completeness_score": completeness,
        "field_count": len(parsed) if isinstance(parsed, dict) else 0,
    }


def calculate_token_efficiency(ocr_output: str, model: str = "gpt-4") -> dict:
    """Calculate token efficiency metrics for LLM consumption."""
    if not TIKTOKEN_AVAILABLE:
        return {
            "error": "tiktoken not installed",
            "total_tokens": 0,
            "meaningful_tokens": 0,
            "token_density": 0,
            "compression_ratio": 0,
            "estimated_cost_usd": 0,
            "chars_per_token": 0,
        }

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = len(encoding.encode(ocr_output))
    cleaned = " ".join(ocr_output.split())
    cleaned_tokens = len(encoding.encode(cleaned))
    meaningful_tokens = len(encoding.encode(cleaned.strip()))

    token_density = meaningful_tokens / total_tokens if total_tokens > 0 else 0
    compression_ratio = cleaned_tokens / total_tokens if total_tokens > 0 else 1.0
    estimated_cost = (total_tokens / 1000) * 0.01
    chars_per_token = len(ocr_output) / total_tokens if total_tokens > 0 else 0

    return {
        "total_tokens": total_tokens,
        "meaningful_tokens": meaningful_tokens,
        "token_density": token_density,
        "compression_ratio": compression_ratio,
        "estimated_cost_usd": estimated_cost,
        "chars_per_token": chars_per_token,
    }


def evaluate_downstream_task(task_type: str, result: dict, ground_truth: dict) -> dict:
    """Evaluate downstream LLM task success for extraction/QA/classification."""
    if task_type == "extraction":
        required_fields = set(ground_truth.keys())
        extracted_fields = set(result.keys())
        matches = sum(
            1
            for key in required_fields & extracted_fields
            if str(result[key]).strip().lower() == str(ground_truth[key]).strip().lower()
        )
        quality = matches / len(required_fields) if required_fields else 0
        success = quality >= 0.8
        error_type = None if success else "extraction_incomplete"

    elif task_type == "qa":
        pred_answer = str(result.get("answer", "")).strip().lower()
        gt_answer = str(ground_truth.get("answer", "")).strip().lower()
        quality = 1.0 if pred_answer == gt_answer else 0.0
        success = quality >= 0.5
        error_type = None if success else "answer_incorrect"

    elif task_type == "classification":
        success = result.get("class") == ground_truth.get("class")
        quality = 1.0 if success else 0.0
        error_type = None if success else "classification_wrong"

    else:
        success = False
        quality = 0.0
        error_type = "unknown_task_type"

    return {
        "task_success": success,
        "quality_score": quality,
        "error_type": error_type,
    }
