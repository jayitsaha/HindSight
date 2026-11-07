# API reference

The server is FastAPI; run it with `uvicorn hindsight.api.app:app`. Responses are
the pydantic models from `hindsight.timeline.models`. A session (`vid`) is one
recording — either a live phone feed or a loaded synthetic day.

| method | path | body | returns |
|--------|------|------|---------|
| GET  | `/health` | — | `{status, provider}` — which models are active (NIM or local) |
| POST | `/session` | — | `{video_id}` |
| POST | `/session/{vid}/frame` | multipart `image`+`t`, or JSON `{t, embedding?, detections?}` | `{detections, alerts}` |
| POST | `/session/{vid}/ask` | `{question, t}` | `Answer` (`refused=true` ⇒ "I don't know") |
| GET  | `/session/{vid}/timeline` | — | `TimelineSnapshot` |
| GET  | `/session/{vid}/alerts` | — | `[Alert]` |
| POST | `/demo/{dataset}` | — | `{video_id}` — loads a synthetic day and runs the pipeline |
| WS   | `/session/{vid}/stream` | `{t, image_b64}` frames | pushes `{type:"alert"|"timeline", ...}` |

The `/demo/{dataset}` route (e.g. `demo_01`) is the no-camera path: it loads a
materialized synthetic video, runs put-down, action, and retrieval over it, and
returns a populated timeline with alerts — enough to drive the whole app without
a phone.

## Data model

`Detection(id, frame_id, t, label, score, xyxy, object_id?)` ·
`PutDownEvent(id, object_label, t, frame_id, location_id, unusualness, state_change)` ·
`ActionSegment(id, label, t_start, t_end, score, is_repeat, repeat_of?, repeat_score)` ·
`Alert(kind, t, severity, message, ref_id?)` ·
`Answer(question, t_asked, refused, confidence, text, window?, support_frames)` ·
`TimelineSnapshot(video_id, frames, detections, putdowns, actions, alerts)`.

## Providers

Heavy models sit behind `hindsight.providers`: `get_embedder`, `get_grounder`,
`get_vlm`. The registry picks NVIDIA NIM when a key authorizes and falls back to
local models otherwise, so the server runs with or without network. (Speech is
transcribed on the phone and posted as text, so there's no server-side ASR.)
Set keys in `.env` (`NVIDIA_API_KEYS=...`); leave empty to run fully local.

## Synthetic data

`hindsight.ingest.synthetic.generate(...)` builds a labeled egocentric day and
`eval_all` trains and scores every module on it. See [DATA](DATA.md) for the
fields each module consumes and [EGO4D](EGO4D.md) for the real-data path.
