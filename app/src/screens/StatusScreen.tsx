import React, { useCallback, useEffect, useState } from "react";
import { ScrollView, StyleSheet, Text, View } from "react-native";
import { getHealth, loadDemo } from "../api";
import { BigButton, Card, ScreenTitle } from "../components/ui";
import { API_BASE } from "../config";
import { useSession } from "../session";
import { colors, touch, type } from "../theme";
import { Health } from "../types";

function Row({ name, value }: { name: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowName}>{name}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

export function StatusScreen({ onDemoLoaded }: { onDemoLoaded: () => void }) {
  const { videoId, setVideoId } = useSession();
  const [health, setHealth] = useState<Health | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(async () => {
    setChecking(true);
    setError(null);
    try {
      setHealth(await getHealth());
    } catch {
      setHealth(null);
      setError(`Could not reach the server at ${API_BASE}.`);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  async function runDemo() {
    try {
      const { video_id } = await loadDemo("demo_01");
      setVideoId(video_id);
      onDemoLoaded();
    } catch {
      setError("Could not load the demo session.");
    }
  }

  const online = health?.status === "ok";

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.scroll}>
      <ScreenTitle title="Status" subtitle="How the helper is doing behind the scenes." />

      <Card style={online ? styles.okCard : styles.downCard}>
        <Text style={styles.stateText}>
          {online ? "Connected and ready" : "Not connected"}
        </Text>
        <Text style={styles.meta}>Server: {API_BASE}</Text>
        {error && <Text style={styles.error}>{error}</Text>}
      </Card>

      {health && (
        <Card>
          <Text style={styles.sectionTitle}>Models in use</Text>
          <Row
            name="Mode"
            value={health.provider.active === "nim" ? "NVIDIA NIM (cloud)" : "On this machine"}
          />
          <Row name="Configured" value={health.provider.configured} />
          <Row name="Embedder" value={health.provider.embedder} />
          <Row name="Grounder" value={health.provider.grounder} />
          <Row name="VLM" value={health.provider.vlm} />
          <Row name="API keys loaded" value={String(health.provider.keys_loaded)} />
        </Card>
      )}

      <Card>
        <Text style={styles.sectionTitle}>Session</Text>
        <Text style={styles.meta}>{videoId ? `Active: ${videoId}` : "No session yet."}</Text>
      </Card>

      <BigButton label={checking ? "Checking..." : "Check again"} tone="quiet" onPress={() => void check()} disabled={checking} />
      <BigButton label="Load demo day" onPress={() => void runDemo()} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  scroll: { padding: touch.pad, gap: 12 },
  okCard: { borderColor: colors.good, backgroundColor: colors.goodSoft },
  downCard: { borderColor: colors.alert, backgroundColor: colors.alertSoft },
  stateText: { fontSize: type.heading, fontWeight: "700", color: colors.text },
  sectionTitle: { fontSize: type.body, fontWeight: "700", color: colors.text },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 6,
    gap: 12,
  },
  rowName: { fontSize: type.small, color: colors.textSoft },
  rowValue: { fontSize: type.small, color: colors.text, fontWeight: "600", flexShrink: 1 },
  meta: { fontSize: type.small, color: colors.textSoft },
  error: { fontSize: type.small, color: colors.alert },
});
