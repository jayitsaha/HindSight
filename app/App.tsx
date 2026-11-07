import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";
import { Pressable, SafeAreaView, StyleSheet, Text, View } from "react-native";
import { AskScreen } from "./src/screens/AskScreen";
import { LiveScreen } from "./src/screens/LiveScreen";
import { StatusScreen } from "./src/screens/StatusScreen";
import { TimelineScreen } from "./src/screens/TimelineScreen";
import { SessionProvider } from "./src/session";
import { colors, touch, type } from "./src/theme";

type Tab = "live" | "timeline" | "ask" | "status";

const TABS: { key: Tab; label: string }[] = [
  { key: "live", label: "Live" },
  { key: "timeline", label: "Your day" },
  { key: "ask", label: "Ask" },
  { key: "status", label: "Status" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <SessionProvider>
      <SafeAreaView style={styles.root}>
        <StatusBar style="dark" />
        <View style={styles.content}>
          {tab === "live" && <LiveScreen />}
          {tab === "timeline" && <TimelineScreen />}
          {tab === "ask" && <AskScreen />}
          {tab === "status" && <StatusScreen onDemoLoaded={() => setTab("timeline")} />}
        </View>
        <View style={styles.tabBar} accessibilityRole="tablist">
          {TABS.map(({ key, label }) => {
            const active = tab === key;
            return (
              <Pressable
                key={key}
                onPress={() => setTab(key)}
                accessibilityRole="tab"
                accessibilityState={{ selected: active }}
                style={[styles.tab, active && styles.tabActive]}
              >
                <Text style={[styles.tabText, active && styles.tabTextActive]}>{label}</Text>
              </Pressable>
            );
          })}
        </View>
      </SafeAreaView>
    </SessionProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  content: { flex: 1 },
  tabBar: {
    flexDirection: "row",
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: colors.surface,
    paddingVertical: 6,
    paddingHorizontal: 8,
    gap: 6,
  },
  tab: {
    flex: 1,
    minHeight: touch.minHeight,
    borderRadius: touch.radius,
    alignItems: "center",
    justifyContent: "center",
  },
  tabActive: { backgroundColor: colors.primary },
  tabText: { fontSize: type.small, fontWeight: "600", color: colors.textSoft },
  tabTextActive: { color: "#FFFFFF", fontWeight: "700" },
});
