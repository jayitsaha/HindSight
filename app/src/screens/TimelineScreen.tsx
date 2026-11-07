import React, { useCallback, useEffect, useState } from "react";
import { FlatList, RefreshControl, StyleSheet, Text, View } from "react-native";
import { getTimeline, loadDemo } from "../api";
import { BigButton, Card, ScreenTitle } from "../components/ui";
import { useSession } from "../session";
import { colors, formatClock, touch, type } from "../theme";
import { ActionSegment, Alert, PutDownEvent, TimelineSnapshot } from "../types";

type Event =
  | { kind: "putdown"; t: number; data: PutDownEvent }
  | { kind: "action"; t: number; data: ActionSegment }
  | { kind: "alert"; t: number; data: Alert };

function mergeEvents(snap: TimelineSnapshot): Event[] {
  const events: Event[] = [
    ...snap.putdowns.map((p): Event => ({ kind: "putdown", t: p.t, data: p })),
    ...snap.actions.map((a): Event => ({ kind: "action", t: a.t_start, data: a })),
    ...snap.alerts.map((a): Event => ({ kind: "alert", t: a.t, data: a })),
  ];
  return events.sort((a, b) => a.t - b.t);
}

function EventCard({ event }: { event: Event }) {
  if (event.kind === "putdown") {
    const p = event.data;
    const unusual = p.unusualness >= 0.5;
    return (
      <Card style={unusual ? styles.unusualCard : undefined}>
        <Text style={styles.time}>{formatClock(p.t)}</Text>
        <Text style={styles.eventTitle}>
          You put down the {p.object_label}
        </Text>
        <Text style={styles.eventBody}>
          {unusual
            ? "This was not its usual place. Worth a look later."
            : "It is in a familiar spot."}
        </Text>
      </Card>
    );
  }
  if (event.kind === "action") {
    const a = event.data;
    return (
      <Card>
        <Text style={styles.time}>
          {formatClock(a.t_start)} – {formatClock(a.t_end)}
        </Text>
        <Text style={styles.eventTitle}>{a.label}</Text>
        {a.is_repeat && (
          <Text style={styles.repeat}>You had already done this a little earlier.</Text>
        )}
      </Card>
    );
  }
  const al = event.data;
  return (
    <Card style={styles.alertCard}>
      <Text style={styles.time}>{formatClock(al.t)}</Text>
      <Text style={styles.eventTitle}>
        {al.kind === "misplacement" ? "Unusual place" : "Repeated activity"}
      </Text>
      <Text style={styles.eventBody}>{al.message}</Text>
    </Card>
  );
}

export function TimelineScreen() {
  const { videoId, setVideoId } = useSession();
  const [snapshot, setSnapshot] = useState<TimelineSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    async (vid: string) => {
      setLoading(true);
      setError(null);
      try {
        setSnapshot(await getTimeline(vid));
      } catch {
        setError("Could not load the day so far. Pull down to try again.");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (videoId) void refresh(videoId);
  }, [videoId, refresh]);

  async function runDemo() {
    setLoading(true);
    setError(null);
    try {
      const { video_id } = await loadDemo("demo_01");
      setVideoId(video_id);
      await refresh(video_id);
    } catch {
      setError("Could not load the demo. Is the server running?");
      setLoading(false);
    }
  }

  if (!videoId || !snapshot) {
    return (
      <View style={styles.screen}>
        <ScreenTitle title="Your day" subtitle="A gentle record of where things went." />
        <Card>
          <Text style={styles.eventBody}>
            {videoId
              ? "Nothing loaded yet."
              : "Start watching on the Live tab, or try the demo day below."}
          </Text>
          {videoId && (
            <BigButton label="Load timeline" onPress={() => void refresh(videoId)} disabled={loading} />
          )}
          <BigButton label="Try a demo day" tone={videoId ? "quiet" : "primary"} onPress={() => void runDemo()} disabled={loading} />
          {error && <Text style={styles.error}>{error}</Text>}
        </Card>
      </View>
    );
  }

  const events = mergeEvents(snapshot);

  return (
    <View style={styles.screen}>
      <ScreenTitle
        title="Your day"
        subtitle={`${snapshot.putdowns.length} things set down · ${snapshot.actions.length} activities`}
      />
      {error && <Text style={styles.error}>{error}</Text>}
      <FlatList
        data={events}
        keyExtractor={(e, i) => `${e.kind}-${i}`}
        renderItem={({ item }) => <EventCard event={item} />}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={() => void refresh(videoId)} />
        }
        ListEmptyComponent={
          <Card>
            <Text style={styles.eventBody}>Nothing recorded yet. That is okay.</Text>
          </Card>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, padding: touch.pad, gap: 12, backgroundColor: colors.background },
  list: { gap: 12, paddingBottom: 24 },
  time: { fontSize: type.small, fontWeight: "700", color: colors.primary },
  eventTitle: { fontSize: type.body, fontWeight: "600", color: colors.text },
  eventBody: { fontSize: type.small, color: colors.textSoft, lineHeight: 24 },
  repeat: { fontSize: type.small, color: colors.warn, fontWeight: "600" },
  unusualCard: { borderColor: colors.warn, backgroundColor: colors.warnSoft },
  alertCard: { borderColor: colors.alert, backgroundColor: colors.alertSoft },
  error: { fontSize: type.small, color: colors.warn },
});
