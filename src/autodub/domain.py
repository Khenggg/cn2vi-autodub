from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    QUEUED = "QUEUED"
    UPLOADING = "UPLOADING"
    PREPARING = "PREPARING"
    V1_RUNNING = "V1_RUNNING"
    V2_RUNNING = "V2_RUNNING"
    ASR = "ASR"
    ALIGNING = "ALIGNING"
    TRANSLATING = "TRANSLATING"
    SEPARATING = "SEPARATING"
    TTS = "TTS"
    TIMING = "TIMING"
    AUDIO_MIX = "AUDIO_MIX"
    QC = "QC"
    PREVIEW_READY = "PREVIEW_READY"
    AWAITING_ROI = "AWAITING_ROI"
    OCR_VERIFY = "OCR_VERIFY"
    TEXT_REMOVAL = "TEXT_REMOVAL"
    SUBTITLE_RENDER = "SUBTITLE_RENDER"
    ENCODING = "ENCODING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    RETRYING = "RETRYING"
    PAUSED = "PAUSED"
    CHECKPOINTED = "CHECKPOINTED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


PIPELINE = [State.QUEUED, State.UPLOADING, State.PREPARING, State.SEPARATING, State.ASR,
            State.ALIGNING, State.TRANSLATING, State.TTS, State.TIMING, State.AUDIO_MIX,
            State.QC, State.PREVIEW_READY, State.AWAITING_ROI, State.OCR_VERIFY,
            State.TEXT_REMOVAL, State.SUBTITLE_RENDER, State.ENCODING, State.COMPLETED]


def check_transition(old: str, new: str) -> None:
    old, new = State(old), State(new)
    allowed = {a: {b} for a, b in zip(PIPELINE, PIPELINE[1:], strict=False)}
    allowed[State.QUEUED].add(State.PREPARING)
    allowed[State.PREPARING].add(State.V1_RUNNING)
    allowed[State.PREPARING].add(State.V2_RUNNING)
    allowed[State.UPLOADING].add(State.QUEUED)
    # Recovery may resume at any durable stage represented by the checkpoint.
    allowed[State.CHECKPOINTED] = {State.QUEUED, State.ASR, State.ALIGNING,
                                   State.TRANSLATING, State.SEPARATING, State.TTS,
                                   State.TIMING, State.AUDIO_MIX, State.QC, State.ENCODING, State.V1_RUNNING, State.V2_RUNNING}
    allowed[State.V1_RUNNING] = {State.ENCODING, State.COMPLETED}
    allowed[State.V2_RUNNING] = {State.ENCODING, State.COMPLETED}
    allowed[State.FAILED] = {State.RETRYING}
    allowed[State.NEEDS_REVIEW] = {State.RETRYING, State.COMPLETED}
    allowed[State.RETRYING] = {State.QUEUED}
    allowed[State.PAUSED] = {State.QUEUED}
    allowed[State.QC].add(State.ENCODING)
    allowed[State.QC].update({State.TEXT_REMOVAL, State.SUBTITLE_RENDER})
    allowed[State.TEXT_REMOVAL].add(State.ENCODING)
    allowed[State.TRANSLATING].add(State.AUDIO_MIX)
    if old not in {State.COMPLETED, State.SKIPPED, State.UPLOADING}:
        allowed.setdefault(old, set()).update({State.FAILED, State.CHECKPOINTED, State.PAUSED, State.NEEDS_REVIEW})
    if new not in allowed.get(old, set()):
        raise ValueError(f"Invalid transition: {old} -> {new}")


def merge_windows(spans: list[tuple[int, int]], duration_ms: int, padding_ms: int = 600,
                  gap_ms: int = 1200) -> list[tuple[int, int]]:
    if duration_ms <= 0 or padding_ms < 0 or gap_ms < 0:
        raise ValueError("Invalid window limits")
    windows = []
    for start, end in sorted(spans):
        if not 0 <= start < end <= duration_ms:
            raise ValueError("Span outside episode")
        left, right = max(0, start - padding_ms), min(duration_ms, end + padding_ms)
        if windows and left - windows[-1][1] <= gap_ms:
            windows[-1] = (windows[-1][0], max(right, windows[-1][1]))
        else:
            windows.append((left, right))
    return windows


def fitting_action(actual_ms: int, target_ms: int, attempts: int = 0) -> str:
    if actual_ms <= 0 or target_ms <= 0:
        raise ValueError("Durations must be positive")
    drift = abs(actual_ms - target_ms) / target_ms
    if drift <= 0.08:
        return "STRETCH"
    if attempts >= 3:
        return "NEEDS_REVIEW"
    return "SPEED" if drift <= 0.20 else "REWRITE"


@dataclass(frozen=True)
class Budget:
    cloud_rate: int = 6000
    target: int = 5000
    warning: int = 6500
    hard_warning: int = 8000

    def project(self, elapsed_ms: int, remaining_ms: int = 0, api_vnd: float = 0) -> dict:
        if min(elapsed_ms, remaining_ms, api_vnd) < 0:
            raise ValueError("Cost inputs must be non-negative")
        total = (elapsed_ms + remaining_ms) * self.cloud_rate / 3_600_000 + api_vnd
        policy = ("REQUIRE_CONTINUE" if total > self.hard_warning else
                  "NO_AUTO_PROPAINTER" if total > self.warning else
                  "REDUCE_FALLBACK" if total > self.target else "NORMAL")
        return {"projected_vnd": round(total, 2), "policy": policy}


def plan_separation_windows(spans: list[tuple[int, int]], duration_ms: int, *,
                            min_ms: int = 30_000, max_ms: int = 120_000,
                            overlap_ms: int = 4_000) -> list[dict]:
    """Plan BandIt windows covering every dubbed utterance span.

    Spans are merged, runs longer than ``max_ms`` are split into overlapping
    pieces, every window is padded to ``min_ms`` (BandIt fader minimum), and
    windows fully contained in another are dropped.
    """
    if duration_ms <= 0 or not 0 < overlap_ms < max_ms or min_ms > max_ms:
        raise ValueError("Invalid window limits")
    merged = merge_windows(spans, duration_ms) if spans else [(0, min(duration_ms, min_ms))]
    pieces: list[tuple[int, int]] = []
    for start, end in merged:
        cursor = start
        while True:
            stop = min(end, cursor + max_ms)
            pieces.append((cursor, stop))
            if stop >= end:
                break
            cursor = stop - overlap_ms
    padded: list[tuple[int, int]] = []
    for start, end in pieces:
        if end - start < min_ms:
            start = max(0, start - (min_ms - (end - start)) // 2)
            end = min(duration_ms, start + min_ms)
            start = max(0, end - min_ms)
        padded.append((start, end))
    kept: list[tuple[int, int]] = []
    for start, end in sorted(set(padded)):
        if not any(a <= start and end <= b for a, b in kept):
            kept = [(a, b) for a, b in kept if not (start <= a and b <= end)] + [(start, end)]
    return [{"start_ms": start, "end_ms": end} for start, end in sorted(kept)]
