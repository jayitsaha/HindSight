import { API_BASE } from "./config";
import {
  Alert,
  Answer,
  FrameResponse,
  Health,
  StreamMessage,
  TimelineSnapshot,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${init?.method ?? "GET"} ${path} -> ${res.status} ${body}`);
  }
  return res.json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getHealth(): Promise<Health> {
  return request("/health");
}

export function createSession(opts: { minutes?: number; source?: string } = {}): Promise<{ video_id: string }> {
  return post("/session", opts);
}

// Multipart variant: send a captured photo by local uri.
export function postFrameImage(vid: string, t: number, imageUri: string): Promise<FrameResponse> {
  const form = new FormData();
  form.append("t", String(t));
  // React Native's FormData takes {uri, name, type} for file parts.
  form.append("image", {
    uri: imageUri,
    name: `frame_${Math.round(t * 1000)}.jpg`,
    type: "image/jpeg",
  } as unknown as Blob);
  return request(`/session/${vid}/frame`, { method: "POST", body: form });
}

// JSON variant: precomputed embedding and/or detections, no image upload.
export function postFrameJson(
  vid: string,
  t: number,
  payload: { embedding?: number[]; detections?: unknown[] } = {},
): Promise<FrameResponse> {
  return post(`/session/${vid}/frame`, { t, ...payload });
}

export function ask(vid: string, question: string, t: number): Promise<Answer> {
  return post(`/session/${vid}/ask`, { question, t });
}

export function getTimeline(vid: string): Promise<TimelineSnapshot> {
  return request(`/session/${vid}/timeline`);
}

export function getAlerts(vid: string): Promise<Alert[]> {
  return request(`/session/${vid}/alerts`);
}

export function loadDemo(dataset = "demo_01"): Promise<{ video_id: string }> {
  return post(`/demo/${dataset}`, {});
}

export interface FrameStream {
  sendFrame(t: number, imageB64: string): void;
  close(): void;
}

// WS /session/{vid}/stream: push {t, image_b64}, receive alert/timeline messages.
export function openStream(
  vid: string,
  onMessage: (msg: StreamMessage) => void,
  onError?: (err: unknown) => void,
): FrameStream {
  const url = `${API_BASE.replace(/^http/, "ws")}/session/${vid}/stream`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data as string) as StreamMessage);
    } catch (err) {
      onError?.(err);
    }
  };
  ws.onerror = (ev) => onError?.(ev);
  return {
    sendFrame(t, imageB64) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ t, image_b64: imageB64 }));
      }
    },
    close() {
      ws.close();
    },
  };
}
