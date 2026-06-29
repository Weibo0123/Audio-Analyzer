# findings.py

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field

@dataclass
class Finding:
    kind: str                                              # Stable machine-read category tag.
    confidence: float                                      # Detector's own estimate of how likely this finding is positive, in [0.0, 1.0].
    summary: str                                           # A human-read sentence.
    detail: dict[str, Any] = field(default_factory=dict)   # Structured data specific to the "kind".
    concluded_value: Optional[str] = None                  # If the detector fully solved the puzzle, put the final answer or flag here.

    def __post_init__(self) -> None:
        if self.confidence < 0.0:
            self.confidence = 0.0
        if self.confidence > 1.0:
            self.confidence = 1.0

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "detail": self.detail,
        }
        if self.concluded_value is not None:
            out["concluded_value"] = self.concluded_value
        return out

