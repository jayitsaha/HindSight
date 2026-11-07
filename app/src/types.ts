// Mirrors the pydantic models in server/hindsight/timeline/models.py.

export type AlertKind = "misplacement" | "repeat";

export interface Detection {
  id: number;
  frame_id: number;
  t: number;
  label: string;
  score: number;
  xyxy: [number, number, number, number];
  object_id?: number | null;
}

export interface RetrievalHit {
  query: string;
  frame_id: number;
  t: number;
  xyxy: [number, number, number, number];
  score: number;
  margin: number;
  found: boolean;
}

export interface PutDownEvent {
  id: number;
  object_label: string;
  t: number;
  frame_id: number;
  location_id: number;
  unusualness: number;
  state_change: boolean;
}

export interface ActionSegment {
  id: number;
  label: string;
  t_start: number;
  t_end: number;
  score: number;
  is_repeat: boolean;
  repeat_of?: number | null;
  repeat_score: number;
}

export interface Alert {
  kind: AlertKind;
  t: number;
  severity: number;
  message: string;
  ref_id?: number | null;
}

export interface Answer {
  question: string;
  t_asked: number;
  refused: boolean;
  confidence: number;
  text: string;
  window?: [number, number] | null;
  support_frames: number[];
}

export interface TimelineSnapshot {
  video_id: string;
  frames: number;
  detections: number;
  putdowns: PutDownEvent[];
  actions: ActionSegment[];
  alerts: Alert[];
}

export interface ProviderStatus {
  configured: string;
  active: string;
  keys_loaded: number;
  embedder: string;
  vlm: string;
  grounder: string;
}

export interface Health {
  status: string;
  provider: ProviderStatus;
}

export interface FrameResponse {
  detections: Detection[];
  alerts: Alert[];
}

export type StreamMessage =
  | ({ type: "alert" } & Alert)
  | ({ type: "timeline" } & TimelineSnapshot);
