"""Text correctness metrics for OCR evaluation."""

from collections import defaultdict
from typing import Dict, List, Tuple

import Levenshtein
import jiwer
from jiwer import process_words


def char_edit_details(ground_truth: str, predicted: str) -> dict:
    """Character-level CER with edit counts (jiwer v3+ no longer provides counts)."""
    ops = Levenshtein.editops(ground_truth, predicted)
    insertions = deletions = substitutions = 0
    for op_type, _, _ in ops:
        if op_type == "insert":
            insertions += 1
        elif op_type == "delete":
            deletions += 1
        elif op_type == "replace":
            substitutions += 1

    cer_value = jiwer.cer(ground_truth, predicted)

    return {
        "cer": cer_value * 100,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "total_chars": len(ground_truth),
    }


def word_edit_details(ground_truth: str, predicted: str) -> dict:
    """Word-level WER with edit counts using jiwer.process_words."""
    measures = process_words(ground_truth, predicted)

    ref_count = (
        getattr(measures, "reference_word_count", None)
        or getattr(measures, "reference_length", None)
        or len(ground_truth.split())
    )
    return {
        "wer": measures.wer * 100,
        "insertions": measures.insertions,
        "deletions": measures.deletions,
        "substitutions": measures.substitutions,
        "total_words": ref_count,
    }


def calculate_ser(predicted_sentences: List[str], ground_truth_sentences: List[str]) -> dict:
    """Calculate Sentence Error Rate based on matching sentence lists."""
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
        "error_indices": error_indices,
    }


def calculate_token_accuracy(predicted: str, ground_truth: str) -> dict:
    """Calculate token-level precision/recall/F1 and exact match rate."""
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
        "total_pred_tokens": len(pred_tokens),
    }


def calculate_confusion_matrix(predicted: str, ground_truth: str) -> dict:
    """Calculate character confusion matrix and top confusions."""
    ops = Levenshtein.editops(ground_truth, predicted)

    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_errors = 0

    for op_type, gt_idx, pred_idx in ops:
        if op_type == "replace":
            gt_char = ground_truth[gt_idx]
            pred_char = predicted[pred_idx]
            confusion_matrix[gt_char][pred_char] += 1
            total_errors += 1
        elif op_type == "delete":
            gt_char = ground_truth[gt_idx]
            confusion_matrix[gt_char]["<DELETE>"] += 1
            total_errors += 1
        elif op_type == "insert":
            pred_char = predicted[pred_idx]
            confusion_matrix["<INSERT>"][pred_char] += 1
            total_errors += 1

    matrix_dict = {k: dict(v) for k, v in confusion_matrix.items()}

    top_confusions: List[Tuple[str, str, int]] = []
    for gt_char, pred_dict in matrix_dict.items():
        for pred_char, count in pred_dict.items():
            top_confusions.append((gt_char, pred_char, count))
    top_confusions.sort(key=lambda x: x[2], reverse=True)

    return {
        "confusion_matrix": matrix_dict,
        "total_errors": total_errors,
        "top_confusions": top_confusions[:10],
        "confusion_rate": (total_errors / len(ground_truth) * 100) if ground_truth else 0,
    }


def compute_text_metrics(predicted: str, ground_truth: str) -> dict:
    """Return CER and WER metrics in a unified structure."""
    return {
        "cer": char_edit_details(ground_truth, predicted),
        "wer": word_edit_details(ground_truth, predicted),
    }


def calculate_all_text_metrics(predicted: str, ground_truth: str) -> dict:
    """Convenience wrapper to compute core text metrics."""
    ground_truth_sentences = ground_truth.splitlines()
    predicted_sentences = predicted.splitlines()

    return {
        "cer": char_edit_details(ground_truth, predicted),
        "wer": word_edit_details(ground_truth, predicted),
        "ser": calculate_ser(predicted_sentences, ground_truth_sentences),
        "token_accuracy": calculate_token_accuracy(predicted, ground_truth),
        "confusion_matrix": calculate_confusion_matrix(predicted, ground_truth),
    }
