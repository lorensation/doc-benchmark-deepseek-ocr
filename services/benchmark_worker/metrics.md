# OCR Benchmarking Metrics Specification (Simplified)

**Document Version:** 2.0 (Simplified)
**Last Updated:** November 30, 2025
**Project:** doc-benchmark-deepseek-ocr

---

## Table of Contents

1. [Overview](#overview)
2. [Text Correctness Metrics](#1-text-correctness-metrics)
3. [Token Output Metrics for Multimodal LLM Integration](#2-token-output-metrics-for-multimodal-llm-integration)
4. [Implementation Guide](#implementation-guide)
5. [Benchmark Protocol](#benchmark-protocol)
6. [References](#references)

---

## Overview

This document defines **core OCR evaluation metrics** for comparing DeepSeek-OCR and Tesseract in production environments. The focus is on:

1. **Text Correctness**: How accurate is the extracted text?
2. **LLM Integration**: How well does OCR output work with downstream LLM tasks?

### Scope

**Models Compared:**
- DeepSeek-OCR (advanced vision-language model)
- Tesseract (classic OCR baseline)

**Key Metrics:**
- 5 text correctness metrics
- 4 LLM integration metrics

**Deployment:**
- Docker Compose environment
- HTTP API-based OCR workers
- Python-based metric computation

---

## 1. Text Correctness Metrics

### 1.1 Character Error Rate (CER)

**Definition**: Measures the rate of erroneous characters in OCR output compared to ground truth.

**Formula**:
```
CER = (I + D + S) / N × 100%
```

Where:
- `I` = Insertions (extra characters in OCR output)
- `D` = Deletions (missing characters from reference)
- `S` = Substitutions (characters replaced incorrectly)
- `N` = Total number of characters in ground truth

**Characteristics**:
- Range: [0, ∞) (can exceed 100% with many insertions)
- Lower is better (0% = perfect)
- Character-level granularity
- Calculated using Levenshtein distance

**Implementation**:

**Input**:
```python
{
  "ground_truth": str,  # Reference text
  "predicted": str      # OCR output
}
```

**Output**:
```python
{
  "cer": float,         # 0-100+ (percentage)
  "insertions": int,
  "deletions": int,
  "substitutions": int,
  "total_chars": int
}
```

**Python Code**:
```python
import jiwer

def calculate_cer(predicted: str, ground_truth: str) -> dict:
    """Calculate Character Error Rate."""
    cer = jiwer.cer(ground_truth, predicted)

    # Get detailed error counts
    measures = jiwer.compute_measures(ground_truth, predicted)

    return {
        "cer": cer * 100,  # Convert to percentage
        "insertions": measures["insertions"],
        "deletions": measures["deletions"],
        "substitutions": measures["substitutions"],
        "total_chars": len(ground_truth)
    }

# Example usage:
ground_truth = "INVOICE #12345"
predicted = "INV0ICE #12345"

result = calculate_cer(predicted, ground_truth)
# {"cer": 7.14, "insertions": 0, "deletions": 0, "substitutions": 1, "total_chars": 14}
```

**Use Cases**:
- Primary metric for OCR accuracy evaluation
- Fine-grained character-level comparison
- Error pattern identification

---

### 1.2 Word Error Rate (WER)

**Definition**: Evaluates accuracy at word level by measuring the proportion of incorrectly recognized words.

**Formula**:
```
WER = (S + D + I) / N × 100%
```

Where:
- `S` = Substituted words
- `D` = Deleted words
- `I` = Inserted words
- `N` = Total number of words in ground truth

**Characteristics**:
- Range: [0, ∞)
- Lower is better
- One wrong character errors the entire word
- WER typically higher than CER (5% CER ≈ 25% WER)

**Implementation**:

**Input**:
```python
{
  "ground_truth": str,
  "predicted": str
}
```

**Output**:
```python
{
  "wer": float,              # 0-100+ (percentage)
  "word_substitutions": int,
  "word_deletions": int,
  "word_insertions": int,
  "total_words": int
}
```

**Python Code**:
```python
import jiwer

def calculate_wer(predicted: str, ground_truth: str) -> dict:
    """Calculate Word Error Rate."""
    wer = jiwer.wer(ground_truth, predicted)

    measures = jiwer.compute_measures(ground_truth, predicted)

    return {
        "wer": wer * 100,
        "word_substitutions": measures["substitutions"],
        "word_deletions": measures["deletions"],
        "word_insertions": measures["insertions"],
        "total_words": len(ground_truth.split())
    }

# Example usage:
ground_truth = "TOTAL AMOUNT DUE"
predicted = "TOTAL AMUNT DUE"

result = calculate_wer(predicted, ground_truth)
# {"wer": 33.33, "word_substitutions": 1, "word_deletions": 0, "word_insertions": 0, "total_words": 3}
```

**Use Cases**:
- Document transcription evaluation
- Business document processing
- Word-level accuracy assessment

---

### 1.3 Sentence Error Rate (SER)

**Definition**: Percentage of sentence/line segments with at least one error.

**Formula**:
```
SER = (Sentences with errors) / (Total sentences) × 100%
```

**Characteristics**:
- Range: [0, 100]
- Binary per sentence (error or no error)
- Useful for structured documents (invoices, forms)

**Implementation**:

**Input**:
```python
{
  "ground_truth_sentences": List[str],
  "predicted_sentences": List[str]
}
```

**Output**:
```python
{
  "ser": float,                # 0-100
  "total_sentences": int,
  "error_sentences": int,
  "correct_sentences": int,
  "error_indices": List[int]   # Which sentences had errors
}
```

**Python Code**:
```python
def calculate_ser(predicted_sentences: list[str], ground_truth_sentences: list[str]) -> dict:
    """Calculate Sentence Error Rate."""

    # Ensure same length (pad if needed)
    max_len = max(len(predicted_sentences), len(ground_truth_sentences))
    pred = predicted_sentences + [""] * (max_len - len(predicted_sentences))
    gt = ground_truth_sentences + [""] * (max_len - len(ground_truth_sentences))

    # Count sentences with errors
    error_indices = []
    for i, (p, g) in enumerate(zip(pred, gt)):
        if p.strip() != g.strip():
            error_indices.append(i)

    error_count = len(error_indices)
    total = max_len

    return {
        "ser": (error_count / total * 100) if total > 0 else 0,
        "total_sentences": total,
        "error_sentences": error_count,
        "correct_sentences": total - error_count,
        "error_indices": error_indices
    }

# Example usage:
ground_truth_lines = [
    "INVOICE NUMBER: INV-2024-001",
    "DATE: 2024-03-15",
    "TOTAL: $150.00"
]
predicted_lines = [
    "INVOICE NUMBER: INV-2024-001",
    "DATE: 2024-03-15",
    "TOTAL: $15O.OO"  # Error: 0 vs O
]

result = calculate_ser(predicted_lines, ground_truth_lines)
# {"ser": 33.33, "total_sentences": 3, "error_sentences": 1, "correct_sentences": 2, "error_indices": [2]}
```

**Use Cases**:
- Multi-line document evaluation
- Invoice line items
- Form field accuracy

---

### 1.4 Token-Level Accuracy

**Definition**: Measures correctness at token level with precision, recall, and F1 score.

**Metrics**:
1. **Exact Match Rate (EMR)**: Percentage of perfect token matches
2. **Token-Level F1**: Harmonic mean of precision and recall

**Formula**:
```
Precision = Correct Tokens / Predicted Tokens
Recall = Correct Tokens / Ground Truth Tokens
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Implementation**:

**Input**:
```python
{
  "ground_truth": str,        # Or List[str] of tokens
  "predicted": str
}
```

**Output**:
```python
{
  "exact_match_rate": float,  # 0-100
  "token_precision": float,   # 0-1
  "token_recall": float,      # 0-1
  "token_f1": float,         # 0-1
  "correct_tokens": int,
  "total_gt_tokens": int,
  "total_pred_tokens": int
}
```

**Python Code**:
```python
def calculate_token_accuracy(predicted: str, ground_truth: str) -> dict:
    """Calculate token-level accuracy metrics."""

    # Tokenize (split by whitespace)
    pred_tokens = predicted.split()
    gt_tokens = ground_truth.split()

    # Convert to sets for unique tokens
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    # Count matches
    correct = len(pred_set & gt_set)

    # Calculate metrics
    precision = correct / len(pred_set) if pred_set else 0
    recall = correct / len(gt_set) if gt_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Exact match rate (how many tokens are identical in order)
    exact_matches = sum(1 for p, g in zip(pred_tokens, gt_tokens) if p == g)
    max_len = max(len(pred_tokens), len(gt_tokens))
    emr = (exact_matches / max_len * 100) if max_len > 0 else 0

    return {
        "exact_match_rate": emr,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "correct_tokens": correct,
        "total_gt_tokens": len(gt_tokens),
        "total_pred_tokens": len(pred_tokens)
    }

# Example usage:
ground_truth = "Invoice Number INV-2024-001 Total Amount $150.00"
predicted = "Invoice Number INV-2024-OO1 Total Amount $150.00"  # Error in '001'

result = calculate_token_accuracy(predicted, ground_truth)
# {"exact_match_rate": 83.33, "token_precision": 0.857, "token_recall": 0.857, "token_f1": 0.857, ...}
```

**Use Cases**:
- Structured data extraction
- Key-value pair extraction
- Named entity recognition

---

### 1.5 Character Confusion Matrix

**Definition**: Matrix showing which characters are commonly misrecognized as other characters.

**Purpose**:
- Identify OCR engine error patterns
- Understand common confusions (e.g., 'O' ↔ '0', 'I' ↔ 'l', '1' ↔ 'l')
- Compare error signatures between DeepSeek and Tesseract

**Implementation**:

**Input**:
```python
{
  "ground_truth": str,
  "predicted": str
}
```

**Output**:
```python
{
  "confusion_matrix": Dict[str, Dict[str, int]],  # gt_char -> {pred_char: count}
  "total_errors": int,
  "top_confusions": List[Tuple[str, str, int]],  # (gt, pred, count)
  "confusion_rate": float                         # errors / total chars
}
```

**Python Code**:
```python
from collections import defaultdict
import Levenshtein

def calculate_confusion_matrix(predicted: str, ground_truth: str) -> dict:
    """Calculate character confusion matrix."""

    # Get character-level alignment using Levenshtein
    ops = Levenshtein.editops(ground_truth, predicted)

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    total_errors = 0

    for op_type, gt_idx, pred_idx in ops:
        if op_type == 'replace':
            gt_char = ground_truth[gt_idx]
            pred_char = predicted[pred_idx]
            confusion_matrix[gt_char][pred_char] += 1
            total_errors += 1
        elif op_type == 'delete':
            gt_char = ground_truth[gt_idx]
            confusion_matrix[gt_char]['<DELETE>'] += 1
            total_errors += 1
        elif op_type == 'insert':
            pred_char = predicted[pred_idx]
            confusion_matrix['<INSERT>'][pred_char] += 1
            total_errors += 1

    # Convert to regular dict
    confusion_matrix = {k: dict(v) for k, v in confusion_matrix.items()}

    # Get top confusions
    top_confusions = []
    for gt_char, pred_dict in confusion_matrix.items():
        for pred_char, count in pred_dict.items():
            top_confusions.append((gt_char, pred_char, count))
    top_confusions.sort(key=lambda x: x[2], reverse=True)

    return {
        "confusion_matrix": confusion_matrix,
        "total_errors": total_errors,
        "top_confusions": top_confusions[:10],  # Top 10
        "confusion_rate": (total_errors / len(ground_truth) * 100) if ground_truth else 0
    }

# Example usage:
ground_truth = "INVOICE #12345 TOTAL: $150.00"
predicted = "INV0ICE #I2345 T0TAL: $15O.OO"

result = calculate_confusion_matrix(predicted, ground_truth)
# {
#   "confusion_matrix": {"O": {"0": 3}, "1": {"I": 1}},
#   "total_errors": 4,
#   "top_confusions": [("O", "0", 3), ("1", "I", 1)],
#   "confusion_rate": 13.79
# }
```

**Use Cases**:
- OCR engine comparison (which model confuses which characters)
- Error pattern analysis
- Post-processing optimization (correct common confusions)

---

## 2. Token Output Metrics for Multimodal LLM Integration

### 2.1 Field Extraction F1

**Definition**: Measures quality of structured field extraction from documents (invoices, receipts, forms).

**Formula**:
```
Precision = Correctly Extracted Fields / Total Extracted Fields
Recall = Correctly Extracted Fields / Total Ground Truth Fields
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Implementation**:

**Input**:
```python
{
  "ground_truth_fields": Dict[str, str],  # field_name -> value
  "extracted_fields": Dict[str, str]
}
```

**Output**:
```python
{
  "field_precision": float,        # 0-1
  "field_recall": float,           # 0-1
  "field_f1": float,              # 0-1
  "correct_fields": List[str],
  "missing_fields": List[str],
  "incorrect_fields": List[str],
  "extra_fields": List[str]
}
```

**Python Code**:
```python
def calculate_field_extraction_f1(extracted_fields: dict, ground_truth_fields: dict) -> dict:
    """Calculate F1 score for field extraction."""

    gt_keys = set(ground_truth_fields.keys())
    ext_keys = set(extracted_fields.keys())

    # Count correct extractions (both key and value must match)
    correct = []
    for key in gt_keys & ext_keys:
        if str(extracted_fields[key]).strip().lower() == str(ground_truth_fields[key]).strip().lower():
            correct.append(key)

    # Calculate metrics
    precision = len(correct) / len(ext_keys) if ext_keys else 0
    recall = len(correct) / len(gt_keys) if gt_keys else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Identify error types
    missing_fields = list(gt_keys - ext_keys)
    extra_fields = list(ext_keys - gt_keys)
    incorrect_fields = [k for k in gt_keys & ext_keys if k not in correct]

    return {
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "correct_fields": correct,
        "missing_fields": missing_fields,
        "incorrect_fields": incorrect_fields,
        "extra_fields": extra_fields
    }

# Example usage:
ground_truth = {
    "invoice_number": "INV-2024-001",
    "date": "2024-03-15",
    "total": "$150.00",
    "vendor": "Acme Corp"
}

extracted = {
    "invoice_number": "INV-2024-001",
    "date": "2024-03-15",
    "total": "$15O.OO",  # Error: O instead of 0
    "customer": "John Doe"  # Extra field
}

result = calculate_field_extraction_f1(extracted, ground_truth)
# {
#   "field_precision": 0.5,  # 2 correct out of 4 extracted
#   "field_recall": 0.5,     # 2 correct out of 4 ground truth
#   "field_f1": 0.5,
#   "correct_fields": ["invoice_number", "date"],
#   "missing_fields": ["vendor"],
#   "incorrect_fields": ["total"],
#   "extra_fields": ["customer"]
# }
```

**Use Cases**:
- Invoice data extraction
- Receipt parsing
- Form field recognition

---

### 2.2 JSON/XML Validity Rate

**Definition**: Percentage of OCR outputs that produce valid, parseable structured data.

**Metrics**:
1. **Parsing Success Rate**: Can the output be parsed as valid JSON/XML?
2. **Schema Compliance Rate**: Does it match the expected schema?
3. **Field Completeness**: Are all required fields present?

**Implementation**:

**Input**:
```python
{
  "ocr_output": str,            # Raw OCR text or structured output
  "expected_schema": dict,      # JSON Schema definition
  "output_format": str          # "json" or "xml"
}
```

**Output**:
```python
{
  "is_valid": bool,
  "is_schema_compliant": bool,
  "parsing_error": Optional[str],
  "validation_errors": List[str],
  "completeness_score": float,  # 0-1 (required fields present)
  "field_count": int
}
```

**Python Code**:
```python
import json
from jsonschema import validate, ValidationError

def validate_json_output(ocr_output: str, schema: dict) -> dict:
    """Validate JSON output from OCR."""

    # Try to parse JSON
    try:
        parsed = json.loads(ocr_output)
        is_valid = True
        parsing_error = None
    except json.JSONDecodeError as e:
        return {
            "is_valid": False,
            "is_schema_compliant": False,
            "parsing_error": str(e),
            "validation_errors": [],
            "completeness_score": 0.0,
            "field_count": 0
        }

    # Validate against schema
    try:
        validate(instance=parsed, schema=schema)
        is_schema_compliant = True
        validation_errors = []
    except ValidationError as e:
        is_schema_compliant = False
        validation_errors = [str(e)]

    # Calculate completeness
    required_fields = schema.get("required", [])
    present_fields = sum(1 for f in required_fields if f in parsed)
    completeness = present_fields / len(required_fields) if required_fields else 1.0

    return {
        "is_valid": is_valid,
        "is_schema_compliant": is_schema_compliant,
        "parsing_error": parsing_error,
        "validation_errors": validation_errors,
        "completeness_score": completeness,
        "field_count": len(parsed) if isinstance(parsed, dict) else 0
    }

# Example usage:
schema = {
    "type": "object",
    "required": ["invoice_number", "date", "total"],
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total": {"type": "string"}
    }
}

ocr_json = '{"invoice_number": "INV-001", "date": "2024-03-15", "total": "$150.00"}'

result = validate_json_output(ocr_json, schema)
# {
#   "is_valid": True,
#   "is_schema_compliant": True,
#   "parsing_error": None,
#   "validation_errors": [],
#   "completeness_score": 1.0,
#   "field_count": 3
# }
```

**Use Cases**:
- LLM-based structured extraction validation
- API response validation
- Data pipeline quality checks

---

### 2.3 LLM Prompt Token Efficiency

**Definition**: Measures how efficiently OCR output can be tokenized for LLM consumption.

**Metrics**:
1. **Token Count**: Total tokens in OCR output
2. **Token Density**: Meaningful tokens / Total tokens
3. **Compression Ratio**: Cleaned tokens / Raw tokens
4. **Estimated Cost**: Based on LLM pricing

**Implementation**:

**Input**:
```python
{
  "ocr_output": str,
  "tokenizer": str = "gpt-4"  # or "claude", "llama", etc.
}
```

**Output**:
```python
{
  "total_tokens": int,
  "meaningful_tokens": int,
  "token_density": float,        # 0-1
  "compression_ratio": float,
  "estimated_cost_usd": float,   # Based on pricing
  "chars_per_token": float
}
```

**Python Code**:
```python
import tiktoken

def calculate_token_efficiency(ocr_output: str, model: str = "gpt-4") -> dict:
    """Calculate token efficiency for LLM consumption."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (GPT-4 encoding)
        encoding = tiktoken.get_encoding("cl100k_base")

    # Raw tokens
    total_tokens = len(encoding.encode(ocr_output))

    # Cleaned tokens (remove excessive whitespace)
    cleaned = " ".join(ocr_output.split())
    cleaned_tokens = len(encoding.encode(cleaned))

    # Meaningful content (strip and deduplicate)
    meaningful = cleaned.strip()
    meaningful_tokens = len(encoding.encode(meaningful))

    # Calculate metrics
    token_density = meaningful_tokens / total_tokens if total_tokens > 0 else 0
    compression_ratio = cleaned_tokens / total_tokens if total_tokens > 0 else 1.0

    # Estimate cost (example: $0.01 per 1K tokens for GPT-4)
    cost_per_1k = 0.01  # Adjust based on actual pricing
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    chars_per_token = len(ocr_output) / total_tokens if total_tokens > 0 else 0

    return {
        "total_tokens": total_tokens,
        "meaningful_tokens": meaningful_tokens,
        "token_density": token_density,
        "compression_ratio": compression_ratio,
        "estimated_cost_usd": estimated_cost,
        "chars_per_token": chars_per_token
    }

# Example usage:
ocr_output = """INVOICE
Invoice Number: INV-2024-001
Date: 2024-03-15
Total Amount: $150.00"""

result = calculate_token_efficiency(ocr_output)
# {
#   "total_tokens": 25,
#   "meaningful_tokens": 25,
#   "token_density": 1.0,
#   "compression_ratio": 1.0,
#   "estimated_cost_usd": 0.00025,
#   "chars_per_token": 3.2
# }
```

**Use Cases**:
- LLM pipeline cost estimation
- OCR quality assessment (noise reduction)
- Preprocessing optimization

---

### 2.4 Downstream Task Success Rate

**Definition**: Percentage of documents where LLM successfully completes downstream tasks using OCR output.

**Tasks**:
- Data extraction (structured fields)
- Question answering
- Document classification
- Summarization

**Implementation**:

**Input**:
```python
{
  "ocr_output": str,
  "task_type": str,              # "extraction", "qa", "classification"
  "task_result": dict,
  "ground_truth_result": dict
}
```

**Output**:
```python
{
  "task_success": bool,
  "success_rate": float,          # Across batch
  "error_type": Optional[str],
  "quality_score": float          # Task-specific (0-1)
}
```

**Python Code**:
```python
def evaluate_downstream_task(task_type: str, result: dict, ground_truth: dict) -> dict:
    """Evaluate downstream task success."""

    if task_type == "extraction":
        # Data extraction task
        required_fields = set(ground_truth.keys())
        extracted_fields = set(result.keys())

        # Count matches
        matches = sum(
            1 for k in required_fields & extracted_fields
            if str(result[k]).strip().lower() == str(ground_truth[k]).strip().lower()
        )

        quality = matches / len(required_fields) if required_fields else 0
        success = quality >= 0.8  # 80% threshold
        error_type = None if success else "extraction_incomplete"

    elif task_type == "qa":
        # Question answering task
        pred_answer = str(result.get("answer", "")).strip().lower()
        gt_answer = str(ground_truth.get("answer", "")).strip().lower()

        # Exact match or high similarity
        quality = 1.0 if pred_answer == gt_answer else 0.0
        success = quality >= 0.5
        error_type = None if success else "answer_incorrect"

    elif task_type == "classification":
        # Document classification
        pred_class = result.get("class")
        gt_class = ground_truth.get("class")

        success = pred_class == gt_class
        quality = 1.0 if success else 0.0
        error_type = None if success else "classification_wrong"

    else:
        success = False
        quality = 0.0
        error_type = "unknown_task_type"

    return {
        "task_success": success,
        "quality_score": quality,
        "error_type": error_type
    }

# Example usage:
task_result = {
    "invoice_number": "INV-2024-001",
    "date": "2024-03-15",
    "total": "$150.00"
}

ground_truth = {
    "invoice_number": "INV-2024-001",
    "date": "2024-03-15",
    "total": "$150.00",
    "vendor": "Acme Corp"  # Missing in result
}

result = evaluate_downstream_task("extraction", task_result, ground_truth)
# {
#   "task_success": True,  # 3/4 = 75%, meets 80% threshold? No
#   "quality_score": 0.75,
#   "error_type": "extraction_incomplete"
# }
```

**Use Cases**:
- End-to-end pipeline validation
- OCR quality impact measurement
- Production automation rate tracking

---

## Implementation Guide

### Dependencies

Create `services/benchmark_worker/requirements.txt`:

```txt
# Text correctness metrics
jiwer>=3.0.0                # CER, WER
python-Levenshtein>=0.21.0  # Confusion matrix, edit distance

# LLM integration metrics
jsonschema>=4.19.0          # JSON validation
tiktoken>=0.5.0             # Token counting (OpenAI)

# Utilities
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.66.0
```

Install dependencies:
```bash
cd services/benchmark_worker
pip install -r requirements.txt
```

---

### Module Structure

Organize code into clean modules:

```
services/benchmark_worker/
├── worker.py                      # Main orchestrator
├── metrics/
│   ├── __init__.py
│   ├── text_correctness.py        # CER, WER, SER, token accuracy, confusion matrix
│   └── llm_integration.py         # Field F1, JSON validation, token efficiency
├── utils/
│   ├── __init__.py
│   └── data_loader.py             # Load datasets and ground truth
├── requirements.txt
└── metrics.md                      # This file
```

---

### Implementation Example

**File: `services/benchmark_worker/metrics/text_correctness.py`**

```python
"""Text correctness metrics for OCR evaluation."""

import jiwer
import Levenshtein
from collections import defaultdict
from typing import Dict, List, Tuple


def calculate_cer(predicted: str, ground_truth: str) -> dict:
    """Calculate Character Error Rate."""
    cer = jiwer.cer(ground_truth, predicted)
    measures = jiwer.compute_measures(ground_truth, predicted)

    return {
        "cer": cer * 100,
        "insertions": measures["insertions"],
        "deletions": measures["deletions"],
        "substitutions": measures["substitutions"],
        "total_chars": len(ground_truth)
    }


def calculate_wer(predicted: str, ground_truth: str) -> dict:
    """Calculate Word Error Rate."""
    wer = jiwer.wer(ground_truth, predicted)
    measures = jiwer.compute_measures(ground_truth, predicted)

    return {
        "wer": wer * 100,
        "word_substitutions": measures["substitutions"],
        "word_deletions": measures["deletions"],
        "word_insertions": measures["insertions"],
        "total_words": len(ground_truth.split())
    }


def calculate_ser(predicted_sentences: List[str], ground_truth_sentences: List[str]) -> dict:
    """Calculate Sentence Error Rate."""
    max_len = max(len(predicted_sentences), len(ground_truth_sentences))
    pred = predicted_sentences + [""] * (max_len - len(predicted_sentences))
    gt = ground_truth_sentences + [""] * (max_len - len(ground_truth_sentences))

    error_indices = [i for i, (p, g) in enumerate(zip(pred, gt)) if p.strip() != g.strip()]
    error_count = len(error_indices)

    return {
        "ser": (error_count / max_len * 100) if max_len > 0 else 0,
        "total_sentences": max_len,
        "error_sentences": error_count,
        "correct_sentences": max_len - error_count,
        "error_indices": error_indices
    }


def calculate_token_accuracy(predicted: str, ground_truth: str) -> dict:
    """Calculate token-level accuracy."""
    pred_tokens = predicted.split()
    gt_tokens = ground_truth.split()

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    correct = len(pred_set & gt_set)

    precision = correct / len(pred_set) if pred_set else 0
    recall = correct / len(gt_set) if gt_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    exact_matches = sum(1 for p, g in zip(pred_tokens, gt_tokens) if p == g)
    max_len = max(len(pred_tokens), len(gt_tokens))
    emr = (exact_matches / max_len * 100) if max_len > 0 else 0

    return {
        "exact_match_rate": emr,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "correct_tokens": correct,
        "total_gt_tokens": len(gt_tokens),
        "total_pred_tokens": len(pred_tokens)
    }


def calculate_confusion_matrix(predicted: str, ground_truth: str) -> dict:
    """Calculate character confusion matrix."""
    ops = Levenshtein.editops(ground_truth, predicted)

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    total_errors = 0

    for op_type, gt_idx, pred_idx in ops:
        if op_type == 'replace':
            gt_char = ground_truth[gt_idx]
            pred_char = predicted[pred_idx]
            confusion_matrix[gt_char][pred_char] += 1
            total_errors += 1
        elif op_type == 'delete':
            gt_char = ground_truth[gt_idx]
            confusion_matrix[gt_char]['<DELETE>'] += 1
            total_errors += 1
        elif op_type == 'insert':
            pred_char = predicted[pred_idx]
            confusion_matrix['<INSERT>'][pred_char] += 1
            total_errors += 1

    confusion_matrix = {k: dict(v) for k, v in confusion_matrix.items()}

    top_confusions = []
    for gt_char, pred_dict in confusion_matrix.items():
        for pred_char, count in pred_dict.items():
            top_confusions.append((gt_char, pred_char, count))
    top_confusions.sort(key=lambda x: x[2], reverse=True)

    return {
        "confusion_matrix": confusion_matrix,
        "total_errors": total_errors,
        "top_confusions": top_confusions[:10],
        "confusion_rate": (total_errors / len(ground_truth) * 100) if ground_truth else 0
    }


def calculate_all_text_metrics(predicted: str, ground_truth: str) -> dict:
    """Calculate all text correctness metrics."""
    return {
        "cer": calculate_cer(predicted, ground_truth),
        "wer": calculate_wer(predicted, ground_truth),
        "token_accuracy": calculate_token_accuracy(predicted, ground_truth),
        "confusion_matrix": calculate_confusion_matrix(predicted, ground_truth)
    }
```

**File: `services/benchmark_worker/metrics/llm_integration.py`**

```python
"""LLM integration metrics for OCR evaluation."""

import json
from jsonschema import validate, ValidationError
from typing import Dict, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def calculate_field_extraction_f1(extracted_fields: dict, ground_truth_fields: dict) -> dict:
    """Calculate F1 score for field extraction."""
    gt_keys = set(ground_truth_fields.keys())
    ext_keys = set(extracted_fields.keys())

    correct = [
        k for k in gt_keys & ext_keys
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
        "extra_fields": list(ext_keys - gt_keys)
    }


def validate_json_output(ocr_output: str, schema: dict) -> dict:
    """Validate JSON output."""
    try:
        parsed = json.loads(ocr_output)
        is_valid = True
        parsing_error = None
    except json.JSONDecodeError as e:
        return {
            "is_valid": False,
            "is_schema_compliant": False,
            "parsing_error": str(e),
            "validation_errors": [],
            "completeness_score": 0.0,
            "field_count": 0
        }

    try:
        validate(instance=parsed, schema=schema)
        is_schema_compliant = True
        validation_errors = []
    except ValidationError as e:
        is_schema_compliant = False
        validation_errors = [str(e)]

    required_fields = schema.get("required", [])
    present_fields = sum(1 for f in required_fields if f in parsed)
    completeness = present_fields / len(required_fields) if required_fields else 1.0

    return {
        "is_valid": is_valid,
        "is_schema_compliant": is_schema_compliant,
        "parsing_error": parsing_error,
        "validation_errors": validation_errors,
        "completeness_score": completeness,
        "field_count": len(parsed) if isinstance(parsed, dict) else 0
    }


def calculate_token_efficiency(ocr_output: str, model: str = "gpt-4") -> dict:
    """Calculate token efficiency."""
    if not TIKTOKEN_AVAILABLE:
        return {
            "error": "tiktoken not installed",
            "total_tokens": 0,
            "meaningful_tokens": 0,
            "token_density": 0,
            "compression_ratio": 0,
            "estimated_cost_usd": 0,
            "chars_per_token": 0
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
        "chars_per_token": chars_per_token
    }


def evaluate_downstream_task(task_type: str, result: dict, ground_truth: dict) -> dict:
    """Evaluate downstream task success."""
    if task_type == "extraction":
        required_fields = set(ground_truth.keys())
        extracted_fields = set(result.keys())
        matches = sum(
            1 for k in required_fields & extracted_fields
            if str(result[k]).strip().lower() == str(ground_truth[k]).strip().lower()
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
        "error_type": error_type
    }
```

---

## Benchmark Protocol

### Dataset Structure

```
data/
├── datasets/
│   └── {dataset_name}/
│       ├── metadata.json          # Dataset info
│       ├── images/                # Raw images
│       │   ├── doc_001.png
│       │   └── doc_002.jpg
│       └── ground_truth.json      # Reference data
└── results/
    └── benchmarks/
        └── {run_id}/
            ├── config.json        # Run configuration
            ├── results.json       # Per-image results
            └── summary.json       # Aggregated metrics
```

### Ground Truth Format

```json
{
  "doc_001.png": {
    "full_text": "INVOICE\nInvoice Number: INV-2024-001\nDate: 2024-03-15\nTotal: $150.00",
    "fields": {
      "invoice_number": "INV-2024-001",
      "date": "2024-03-15",
      "total": "$150.00"
    }
  }
}
```

### Evaluation Flow

1. **Initialize**: Load dataset and ground truth
2. **Infer**: Call DeepSeek and Tesseract OCR services
3. **Compute**: Calculate all metrics
4. **Store**: Save results to JSON
5. **Report**: Generate summary and comparisons

---

## References

### Libraries
- **jiwer**: [PyPI](https://pypi.org/project/jiwer/) - CER/WER calculation
- **python-Levenshtein**: [PyPI](https://pypi.org/project/python-Levenshtein/) - Edit distance
- **tiktoken**: [GitHub](https://github.com/openai/tiktoken) - Token counting
- **jsonschema**: [PyPI](https://pypi.org/project/jsonschema/) - JSON validation

### Standards
- **ICDAR**: OCR evaluation protocols
- **DocVQA**: Document visual QA benchmarks
- **FUNSD**: Form understanding datasets

---

**End of Specification**

This simplified specification focuses on **9 core metrics** that provide comprehensive OCR evaluation for text correctness and LLM integration use cases.
