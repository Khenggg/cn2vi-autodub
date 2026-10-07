from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeriesCreate(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    priority: int = Field(default=0, ge=0, le=10000)


class SeriesPatch(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    priority: int | None = Field(default=None, ge=0, le=10000)


class EpisodeCreate(StrictModel):
    series_id: str
    ordinal: int = Field(ge=1, le=100000)
    filename: str = Field(min_length=1, max_length=200)
    total_bytes: int = Field(gt=0)
    subtitle_mode: Literal["off", "burn", "replace"] = "replace"


class Roi(StrictModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    scope: Literal["episode", "series"] = "episode"

    @model_validator(mode="after")
    def within_frame(self):
        if self.x + self.w > 1.000001 or self.y + self.h > 1.000001:
            raise ValueError("ROI exceeds frame")
        return self


class GlossaryEntry(StrictModel):
    zh: str = Field(min_length=1, max_length=200)
    vi: str = Field(min_length=1, max_length=300)
    confidence: float = Field(default=1, ge=0, le=1)
    locked_by_user: bool = True


class Word(StrictModel):
    t: str
    s: int = Field(ge=0)
    e: int = Field(gt=0)


class Segment(StrictModel):
    schema_version: int = 1
    id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    zh_text: str = ""
    words: list[Word] = Field(default_factory=list)
    subtitle_vi: str = ""
    dub_vi: str = ""
    emotion: str = "neutral"
    voice_id: str | None = Field(default=None, min_length=1, max_length=80)
    speaker_id: str | None = Field(default=None, min_length=1, max_length=80)
    action: Literal["DUB", "KEEP", "NEEDS_REVIEW"] = "KEEP"
    confidence: dict[str, float | None] = Field(default_factory=dict)
    needs_review: bool = False

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("Empty segment")
        for word in self.words:
            if not self.start_ms <= word.s < word.e <= self.end_ms:
                raise ValueError("Word outside segment")
        if any(value is not None and not 0 <= value <= 1 for value in self.confidence.values()):
            raise ValueError("Confidence must be between zero and one")
        return self


class AsrProvider(Protocol):
    def transcribe(self, audio_path: str) -> list[Segment]: ...


class TtsProvider(Protocol):
    def synthesize(self, text: str, voice_id: str, emotion_hint: str, target_ms: int) -> dict: ...


class TranslationProvider(Protocol):
    def translate(self, segments: list[Segment], glossary: dict[str, str]) -> list[Segment]: ...
