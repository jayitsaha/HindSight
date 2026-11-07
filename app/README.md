# HINDSIGHT app

Expo client for the HINDSIGHT memory assistant. The phone camera stands in for
smart glasses: the Live tab captures frames and streams them to the server,
which detects objects, put-downs, and repeated actions.

## Run

```
cd app
npm install
npx expo start
```

Scan the QR code with Expo Go, or press `i` / `a` for a simulator.

The server address lives in `src/config.ts` (`API_BASE`, default
`http://localhost:8000`). On a physical device, change `localhost` to your
machine's LAN IP. Start the server first:

```
./.venv/bin/uvicorn hindsight.api.app:app --port 8000
```

## No camera handy?

Open the Status tab and tap "Load demo day" (or "Try a demo day" on Your day).
It calls `POST /demo/demo_01` on the server and fills the timeline from the
synthetic dataset.

## Tabs

- **Live** — camera preview; posts a frame every 3 s to `/session/{vid}/frame`
  and shows detections plus misplacement / repeat alert banners.
- **Your day** — chronological timeline of put-downs, activities, and alerts
  from `/session/{vid}/timeline`.
- **Ask** — ask where something is or what you did; when the server refuses
  (low confidence) the app says "I don't know" plainly instead of guessing.
- **Status** — server health and which model providers are active (NIM vs
  local), plus the demo loader.
