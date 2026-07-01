# morse.py

from __future__ import annotations
import math
import time
from dataclasses import dataclass
import numpy as np
from audio_analyzer.findings import Finding
from audio_analyzer.io import LoadedAudio
from audio_analyzer.mores_table import MORSE_TO_CHAR


_FRAME_MS = 6.0                 # The length of the audio window covered by each envelope sample point (milliseconds)
_HOP_MS = 2.0                   # The interval between two adjacent envelope sample points

_MAX_SECONDS = 8.0              # The hard time limit for the whole detection

_MIN_ON_RUNS = 2                # Minimum number of "on" runs before decoding. Below 2, there aren't enough structures to say anything meaningful

# The time period that morse code could possibly be
_MIN_PLAUSIBLE_DOT_S = 0.015
_MAX_PLAUSIBLE_DOT_S = 1.0

@dataclass
class _Run:
    is_on: bool
    duration_s: float


def _amplitude_envelope(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    frame_len = max(1, int(round(sample_rate * _FRAME_MS / 1000.0)))
    hop_len = max(1, int(round(sample_rate * _HOP_MS / 1000.0)))

    n = samples.shape[0]
    if n < frame_len:
        # File shorter than one frame.
        return np.array([float(np.sqrt(np.mean(np.square(samples)) + 1e-12))]), n / sample_rate

    n_frames = 1 + (n - frame_len) // hop_len

    squared = samples.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(squared)])

    starts = np.arange(n_frames) * hop_len
    ends = starts + frame_len
    frame_sums = cumsum[ends] - cumsum[starts]
    env = np.sqrt(frame_sums / frame_len + 1e-12)
    return env, hop_len / sample_rate

def _threshold_envelope(env: np.ndarray) -> np.ndarray:
    lo = np.percentile(env, 10)     # Background noise
    hi = float(np.max(env))            # Maximum volume
    spread = hi - lo                   # Calculate the dynamic range

    # It means the whole audio is almost silent. Return False.
    if spread <= 1e-9 or hi <= 1e-9:
        return np.zeros_like(env, dtype=bool)

    # It means the whole audio stays content. Return False.
    if spread < 0.15 * hi:
        return np.zeros_like(env, dtype=bool)

    # Calculate the threshold.
    threshold = lo + 0.35 * spread
    # Return list of true or false
    return env > threshold

def _mask_to_runs(mask: np.ndarray, hop_s: float) -> list[_Run]:
    # If there's nothing, return None
    if mask.size == 0:
        return []
    runs: list[_Run] = []
    current = bool(mask[0])
    count = 1
    for v in mask[1:]:
        v = bool(v)
        # It means the part still continues
        if v == current:
            count += 1
        # It means the part has ended
        else:
            runs.append(_Run(is_on=current, duration_s=count * hop_s))
            current = v
            count = 1
    runs.append(_Run(is_on=current, duration_s=count * hop_s))
    return runs


def _split_two_clusters(values: list[float]) -> tuple[float, list[int]]:
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0), [0] * n

    order = sorted(range(n), key=lambda i: values[i])
    sorted_vals = [values[i] for i in order]

    gaps = [sorted_vals[i] - sorted_vals[i - 1] for i in range(1, n)]
    best_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    best_gap = gaps[best_idx]
    best_split = best_idx + 1

    if best_gap <= 1e-9:
        return sorted_vals[0], [0] * n

    threshold = (sorted_vals[best_split - 1] + sorted_vals[best_split]) / 2.0
    labels = [0] * n
    labels[best_split - 1] = 1
    labels[best_split:] = 2
    return threshold, labels

def _split_three_clusters(values: list[float]) -> tuple[list[float], list[int]]:
    n = len(values)
    if n < 3:
        thr, labels = _split_two_clusters(values)
        return [thr], labels

    first_threshold, first_labels = _split_two_clusters(values)
    lower_idx = [i for i, lab in enumerate(first_labels) if lab == 0]
    upper_idx = [i for i, lab in enumerate(first_labels) if lab == 1]

    if not upper_idx:
        return [first_threshold], first_labels

    def _try_subsplit(idx_group: list[int]) -> tuple[float, list[int]] | None:
        if len(idx_group) < 2:
            return None
        sub_values = [values[i] for i in idx_group]
        sub_thr, sub_labels = _split_two_clusters(sub_values)
        if all(lab == 0 for lab in sub_labels):
            return None  # No significant sub-split found
        return sub_thr, sub_labels

    lower_split = _try_subsplit(lower_idx)
    if lower_split is not None:
        sub_thr, sub_labels = lower_split
        labels = [0] * n
        for rank, orig_idx in enumerate(lower_idx):
            labels[orig_idx] = 0 if sub_labels[rank] == 0 else 1
        for orig_idx in upper_idx:
            labels[orig_idx] = 2
        return [sub_thr, first_threshold], labels

    upper_split = _try_subsplit(upper_idx)
    if upper_split is not None:
        sub_thr, sub_labels = upper_split
        labels = [0] * n
        for orig_idx in lower_idx:
            labels[orig_idx] = 0
        for rank, orig_idx in enumerate(upper_idx):
            labels[orig_idx] = 1 if sub_labels[rank] == 0 else 2
        return [first_threshold, sub_thr], labels

    return [first_threshold], first_labels

def _cluster_tightness(values: list[float], labels: list[int], n_clusters: int) -> float:
    if not values:
        return 0.0
    scores = []
    for c in range(n_clusters):
        members = [v for v, lab in zip(values, labels) if lab == c]
        if len(members) < 2:
            continue
        mean = sum(members) / len(members)
        if mean <= 1e-9:
            continue
        variance = sum((m - mean) ** 2 for m in members) / len(members)
        cv = (variance ** 0.5) / mean
        # Map CV=0 -> 1.0 (perfectly tight), CV>=0.6 -> 0.0 (very loose).
        scores.append(max(0.0, 1.0 - cv / 0.6))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

_MAX_ANALYZED_SECONDS = 600.0 # 10 minutes

def _detect_morse_inner(audio: LoadedAudio, deadline: float) -> list[Finding]:
    samples = audio.samples
    sr = audio.sample_rate

    if samples.size == 0 or sr <= 0:
        return []

    max_samples = int(_MAX_ANALYZED_SECONDS * sr) if sr > 0 else samples.size
    truncated = max_samples > 0 and samples.size > max_samples
    if truncated:
        samples = samples[:max_samples]

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1e-9:
        samples = samples / peak

    env, hop_s = _amplitude_envelope(samples, sr)
    if time.monotonic() > deadline:
        return []

    mask = _threshold_envelope(env)
    runs = _mask_to_runs(mask, hop_s)

    on_durations = [r.duration_s for r in runs if r.is_on]

    interior_off = [r.duration_s for r in runs[1:-1] if not r.is_on] if len(runs) > 2 else []

    if len(on_durations) < _MIN_ON_RUNS:
        return []

    if time.monotonic() > deadline:
        return []

    dot_dash_threshold, on_labels = _split_two_clusters(on_durations)
    dot_values = [d for d, lab in zip(on_durations, on_labels) if lab == 0]
    dash_values = [d for d, lab in zip(on_durations, on_labels) if lab == 1]

    if not dot_values:
        return []
    dot_mean = sum(dot_values) / len(dot_values)
    dash_mean = sum(dash_values) / len(dash_values) if dash_values else dot_mean * 3.0

    if dot_mean < _MIN_PLAUSIBLE_DOT_S or dot_mean > _MAX_PLAUSIBLE_DOT_S:
        return []

    if dash_values and dash_mean <= dot_mean * 1.3:
        return []

    if time.monotonic() > deadline:
        return []

    gap_labels: list[int] = []
    if len(interior_off) >= 3:
        _gap_thresholds, gap_labels = _split_three_clusters(interior_off)
    elif interior_off:
        _thr, gap_labels = _split_two_clusters(interior_off)

    _CANONICAL_GAP_UNITS = {"intra": 1.0, "inter": 3.0, "word": 7.0}

    def _nearest_tier(units: float) -> str:
        if units <= 0:
            return "intra"
        return min(
            _CANONICAL_GAP_UNITS,
            key=lambda tier: abs(math.log(units) - math.log(_CANONICAL_GAP_UNITS[tier])),
        )

    label_to_tier: dict[int, str] = {}
    if interior_off and dot_mean > 1e-9:
        for label in set(gap_labels):
            members = [d for d, lab in zip(interior_off, gap_labels) if lab == label]
            mean_units = (sum(members) / len(members)) / dot_mean
            label_to_tier[label] = _nearest_tier(mean_units)

    symbol_threshold = dot_dash_threshold
    interior_off_position = 0

    def gap_kind(duration: float) -> str:
        nonlocal interior_off_position
        if not label_to_tier:
            return "intra"
        label = gap_labels[interior_off_position]
        interior_off_position += 1
        return label_to_tier[label, "intra"]

    decoded_chars: list[str] = []
    current_group = ""
    recognized = 0
    total_groups = 0

    for run_index, run in enumerate(runs):
        if time.monotonic() > deadline:
            break
        if run.is_on:
            current_group += "." if run.duration_s <= symbol_threshold else "-"
        else:
            is_interior = 0 < run_index < len(runs) - 1
            kind = gap_kind(run.duration_s) if is_interior else "intra"
            if kind == "intra":
                continue
            if current_group:
                total_groups += 1
                ch = MORSE_TO_CHAR.get(current_group)
                if ch is not None:
                    decoded_chars.append(ch)
                    recognized += 1
                else:
                    decoded_chars.append("#")
                current_group = ""
            if kind == "word":
                decoded_chars.append(" ")

    if current_group:
        total_groups += 1
        ch = MORSE_TO_CHAR.get(current_group)
        if ch is not None:
            decoded_chars.append(ch)
            recognized += 1
        else:
            decoded_chars.append("#")

    if total_groups == 0:
        return []

    decoded_text = "".join(decoded_chars)
    recognition_rate = recognized / total_groups if total_groups else 0.0

    on_tightness = _cluster_tightness(on_durations, on_labels, 2)
    off_tightness = _cluster_tightness(interior_off, gap_labels, 3) if interior_off else 0.5
    ratio_quality = min(1.0, max(0.0, (dash_mean / dot_mean - 1.0) / 2.0)) if dash_mean else 0.0

    confidence = (
        0.45 * recognition_rate
        + 0.25 * on_tightness
        + 0.15 * off_tightness
        + 0.15 * ratio_quality
    )
    confidence = max(0.0, min(1.0, confidence))

    if recognized == 0 and on_tightness < 0.2:
        return []

    start_s = 0.0
    elapsed = 0.0
    first_on_start = None
    last_on_end = None
    for run in runs:
        if run.is_on:
            if first_on_start is None:
                first_on_start = elapsed
            last_on_end = elapsed + run.duration_s
        elapsed += run.duration_s
    start_s = first_on_start or 0.0
    end_s = last_on_end or elapsed

    finding = Finding(
        kind="morse_code_candidate",
        confidence=confidence,
        summary=(
                f"Detected on/off amplitude pattern consistent with Morse code "
                f"between {start_s:.1f}s-{end_s:.1f}s"
                + (f"; decoded: {decoded_text!r}" if decoded_text else "")
        ),
        detail={
            "start_s": start_s,
            "end_s": end_s,
            "decoded_text": decoded_text,
            "recognized_groups": recognized,
            "total_groups": total_groups,
            "recognition_rate": round(recognition_rate, 3),
            "estimated_dot_seconds": round(dot_mean, 4),
            "estimated_dash_seconds": round(dash_mean, 4),
            "decoder_used": audio.decoder_used,
            "analysis_truncated_at_seconds": _MAX_ANALYZED_SECONDS if truncated else None,
        },
    )

    if confidence >= 0.75 and recognition_rate >= 0.9 and decoded_text:
        finding.concluded_value = decoded_text

    return [finding]

def detect_morse(audio: LoadedAudio) -> list[Finding]:
    deadline = time.monotonic() + _MAX_SECONDS
    try:
        return _detect_morse_inner(audio, deadline)
    except Exception as e:
        return []
