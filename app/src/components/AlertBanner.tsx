import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { colors, formatClock, touch, type } from "../theme";
import { Alert } from "../types";

const kindLabel: Record<Alert["kind"], string> = {
  misplacement: "Unusual place",
  repeat: "Already done",
};

export function AlertBanner({ alert, onDismiss }: { alert: Alert; onDismiss?: () => void }) {
  const strong = alert.severity >= 0.6;
  return (
    <View
      style={[styles.banner, strong ? styles.bannerStrong : styles.bannerMild]}
      accessibilityRole="alert"
    >
      <View style={styles.row}>
        <Text style={[styles.kind, strong && styles.kindStrong]}>
          {kindLabel[alert.kind]} at {formatClock(alert.t)}
        </Text>
        {onDismiss && (
          <Pressable
            onPress={onDismiss}
            accessibilityRole="button"
            accessibilityLabel="Dismiss alert"
            hitSlop={12}
            style={styles.dismiss}
          >
            <Text style={styles.dismissText}>OK</Text>
          </Pressable>
        )}
      </View>
      <Text style={styles.message}>{alert.message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    borderRadius: touch.radius,
    padding: touch.pad,
    borderWidth: 2,
    gap: 6,
  },
  bannerStrong: { backgroundColor: colors.alertSoft, borderColor: colors.alert },
  bannerMild: { backgroundColor: colors.warnSoft, borderColor: colors.warn },
  row: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  kind: { fontSize: type.small, fontWeight: "700", color: colors.warn },
  kindStrong: { color: colors.alert },
  message: { fontSize: type.body, color: colors.text, lineHeight: 26 },
  dismiss: {
    minHeight: 44,
    minWidth: 64,
    borderRadius: 10,
    backgroundColor: colors.surface,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  dismissText: { fontSize: type.small, fontWeight: "700", color: colors.text },
});
