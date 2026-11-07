import { CameraView, useCameraPermissions } from "expo-camera";
import React, { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { createSession, postFrameImage } from "../api";
import { AlertBanner } from "../components/AlertBanner";
import { BigButton, Card, ScreenTitle } from "../components/ui";
import { useSession } from "../session";
import { colors, formatClock, touch, type } from "../theme";
import { Detection } from "../types";

const CAPTURE_INTERVAL_MS = 3000;

export function LiveScreen() {
  const { videoId, setVideoId, alerts, pushAlerts, clearAlerts } = useSession();
  const [permission, requestPermission] = useCameraPermissions();
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [error, setError] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const startedAt = useRef(0);
  const busy = useRef(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => stop, []);

  function stop() {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    setRecording(false);
  }

  async function captureFrame(vid: string) {
    if (busy.current || !cameraRef.current) return;
    busy.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.4,
        skipProcessing: true,
        shutterSound: false,
      });
      const t = (Date.now() - startedAt.current) / 1000;
      const res = await postFrameImage(vid, t, photo.uri);
      setDetections(res.detections);
      pushAlerts(res.alerts);
      setError(null);
    } catch (err) {
      setError("Having trouble reaching the helper. Still trying.");
    } finally {
      busy.current = false;
    }
  }

  async function start() {
    setError(null);
    try {
      let vid = videoId;
      if (!vid) {
        const res = await createSession({ source: "live" });
        vid = res.video_id;
        setVideoId(vid);
      }
      startedAt.current = Date.now();
      setElapsed(0);
      setRecording(true);
      const id = vid;
      timer.current = setInterval(() => {
        setElapsed((Date.now() - startedAt.current) / 1000);
        void captureFrame(id);
      }, CAPTURE_INTERVAL_MS);
    } catch {
      setError("Could not start. Check the server address in Status.");
    }
  }

  const latestAlert = alerts[alerts.length - 1];

  if (!permission?.granted) {
    return (
      <View style={styles.screen}>
        <ScreenTitle
          title="Live"
          subtitle="The camera watches with you and remembers where things go."
        />
        <Card>
          <Text style={styles.body}>
            To help you remember, this app needs to use the camera.
          </Text>
          <BigButton label="Allow camera" onPress={() => void requestPermission()} />
        </Card>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenTitle title="Live" subtitle="Point the camera at what you are doing." />

      {latestAlert && <AlertBanner alert={latestAlert} onDismiss={clearAlerts} />}

      <View style={styles.cameraBox}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        {recording && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>Watching · {formatClock(elapsed)}</Text>
          </View>
        )}
      </View>

      {detections.length > 0 && (
        <View style={styles.chips} accessibilityLabel="Objects seen just now">
          {detections.slice(0, 6).map((d) => (
            <View key={d.id} style={styles.chip}>
              <Text style={styles.chipText}>{d.label}</Text>
            </View>
          ))}
        </View>
      )}

      {error && <Text style={styles.error}>{error}</Text>}

      {recording ? (
        <BigButton label="Stop watching" tone="quiet" onPress={stop} />
      ) : (
        <BigButton label="Start watching" onPress={() => void start()} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, padding: touch.pad, gap: 12, backgroundColor: colors.background },
  body: { fontSize: type.body, color: colors.text, lineHeight: 26 },
  cameraBox: {
    flex: 1,
    borderRadius: touch.radius,
    overflow: "hidden",
    backgroundColor: "#000",
  },
  badge: {
    position: "absolute",
    top: 12,
    left: 12,
    backgroundColor: "rgba(28,36,48,0.75)",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  badgeText: { color: "#FFFFFF", fontSize: type.small, fontWeight: "600" },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    backgroundColor: colors.primarySoft,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  chipText: { color: colors.primary, fontSize: type.small, fontWeight: "600" },
  error: { fontSize: type.small, color: colors.warn },
});
