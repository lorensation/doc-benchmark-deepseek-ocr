"""Dataset loaders for benchmark worker."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SROIESample:
    """Structured representation of a single SROIE sample."""

    image_path: Path
    fields: Dict[str, str]
    text_lines: List[str]
    full_text: str


def _parse_box_file(box_path: Path) -> List[str]:
    """Parse SROIE box file into a list of text lines."""
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


def load_sroie_dataset(base_path: str, split: str = "train", limit: Optional[int] = None) -> List[SROIESample]:
    """Load SROIE samples with ground truth fields and text."""
    base_dir = Path(base_path)
    img_dir = base_dir / split / "img"
    entity_dir = base_dir / split / "entities"
    box_dir = base_dir / split / "box"

    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(img_dir.glob(pattern))

    samples: List[SROIESample] = []
    for img_path in sorted(image_paths):
        stem = img_path.stem
        entity_path = entity_dir / f"{stem}.txt"
        fields: Dict[str, str] = {}

        if entity_path.exists():
            with entity_path.open(encoding="utf-8") as file:
                try:
                    fields = json.load(file)
                except json.JSONDecodeError:
                    fields = {}

        text_lines = _parse_box_file(box_dir / f"{stem}.txt")
        full_text = "\n".join(text_lines)

        samples.append(
            SROIESample(
                image_path=img_path,
                fields=fields,
                text_lines=text_lines,
                full_text=full_text,
            )
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples
