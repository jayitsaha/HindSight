from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

AlertKind = Literal["misplacement", "repeat"]


class Frame(BaseModel):
    id: int
    video_id: str
    t: float  # seconds from start of the recording
    path: str
    width: int
    height: int


class Detection(BaseModel):
    id: int
    frame_id: int
    t: float
    label: str
    score: float
    xyxy: tuple[float, float, float, float]
    object_id: Optional[int] = None  # instance id from the perception layer, if known


class RetrievalHit(BaseModel):
    query: str
    frame_id: int
    t: float
    xyxy: tuple[float, float, float, float]
    score: float
    margin: float  # top-1 minus top-2 match score: how separable the best hit is
    found: bool = True


class PutDownEvent(BaseModel):
    id: int
    object_label: str
    t: float
    frame_id: int
    location_id: int
    unusualness: float  # misplacement score in [0, 1]; high = rare resting place
    state_change: bool  # hand-object contact -> no contact confirmed


class ActionSegment(BaseModel):
    id: int
    label: str
    t_start: float
    t_end: float
    score: float
    is_repeat: bool = False
    repeat_of: Optional[int] = None
    repeat_score: float = 0.0


class Alert(BaseModel):
    kind: AlertKind
    t: float
    severity: float  # [0, 1]
    message: str
    ref_id: Optional[int] = None


class Answer(BaseModel):
    question: str
    t_asked: float
    refused: bool
    confidence: float
    text: str
    window: Optional[tuple[float, float]] = None
    support_frames: list[int] = Field(default_factory=list)


class TimelineSnapshot(BaseModel):
    video_id: str
    frames: int
    detections: int
    putdowns: list[PutDownEvent] = Field(default_factory=list)
    actions: list[ActionSegment] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
