"""FastAPI glue: four ML modules behind the HTTP/WebSocket contract.

Run with `uvicorn hindsight.api.app:app`. Models are fit once at startup and
shared; every request resolves a session's TimelineStore and orchestrates the
already-trained services — no model logic lives here.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from ..providers import provider_status
from ..timeline.models import Answer, TimelineSnapshot
from .sessions import Session, SessionManager


@asynccontextmanager
async def lifespan(_: FastAPI):
    SessionManager.instance()  # fit the scorers once before serving
    yield


app = FastAPI(title="HINDSIGHT", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)


def _session(vid: str) -> Session:
    s = SessionManager.instance().get(vid)
    if s is None:
        raise HTTPException(404, f"unknown session {vid}")
    return s


class AskBody(BaseModel):
    question: str
    t: float


@app.get("/health")
def health():
    return {"status": "ok", "provider": provider_status()}


@app.post("/session")
def create_session():
    return {"video_id": SessionManager.instance().new_session()}


@app.post("/session/{vid}/frame")
async def frame(vid: str, request: Request):
    session = _session(vid)  # 404 on unknown session
    mgr = SessionManager.instance()
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        if "t" not in form or "image" not in form:
            raise HTTPException(422, "multipart frame requires 't' and 'image'")
        try:
            t = float(form["t"])
        except (TypeError, ValueError):
            raise HTTPException(422, "'t' must be a number")
        try:
            image = Image.open(io.BytesIO(await form["image"].read()))
        except Exception:
            raise HTTPException(422, "could not decode image")
        dets, alerts = mgr.ingest_image(session, image, t)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(422, "JSON body must be an object")
        dets, alerts = mgr.ingest_json(session, body)
    return {"detections": dets, "alerts": alerts}


@app.post("/session/{vid}/ask", response_model=Answer)
def ask(vid: str, body: AskBody):
    return SessionManager.instance().answer(_session(vid), body.question, body.t)


@app.get("/session/{vid}/timeline", response_model=TimelineSnapshot)
def timeline(vid: str):
    s = _session(vid)
    return s.store.snapshot(s.video_id)


@app.get("/session/{vid}/alerts")
def alerts(vid: str):
    return _session(vid).store.alerts()


@app.post("/demo/{dataset}")
def demo(dataset: str):
    try:
        return {"video_id": SessionManager.instance().demo(dataset)}
    except FileNotFoundError:
        raise HTTPException(404, f"no materialized dataset {dataset}")


@app.websocket("/session/{vid}/stream")
async def stream(ws: WebSocket, vid: str):
    mgr = SessionManager.instance()
    session = mgr.get(vid)
    if session is None:
        await ws.close(code=1008)  # reject before accepting the handshake
        return
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            t = float(msg.get("t", 0.0))
            b64 = msg.get("image_b64")
            if b64:
                image = Image.open(io.BytesIO(base64.b64decode(b64)))
                _, new_alerts = mgr.ingest_image(session, image, t)
            else:
                _, new_alerts = mgr.ingest_json(session, msg)
            for a in new_alerts:
                await ws.send_json({"type": "alert", **a.model_dump()})
            snap = session.store.snapshot(session.video_id)
            await ws.send_json({"type": "timeline", **snap.model_dump()})
    except WebSocketDisconnect:
        return
