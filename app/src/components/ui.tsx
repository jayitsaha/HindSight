import React from "react";
import { Pressable, StyleSheet, Text, View, ViewStyle } from "react-native";
import { colors, touch, type } from "../theme";

export function BigButton({
  label,
  onPress,
  tone = "primary",
  disabled,
}: {
  label: string;
  onPress: () => void;
  tone?: "primary" | "quiet";
  disabled?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      style={({ pressed }) => [
        styles.button,
        tone === "primary" ? styles.buttonPrimary : styles.buttonQuiet,
        pressed && styles.pressed,
        disabled && styles.disabled,
      ]}
    >
      <Text style={[styles.buttonText, tone === "quiet" && styles.buttonTextQuiet]}>{label}</Text>
    </Pressable>
  );
}

export function Card({ children, style }: { children: React.ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function ScreenTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={styles.header}>
      <Text style={styles.title} accessibilityRole="header">
        {title}
      </Text>
      {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: touch.minHeight,
    borderRadius: touch.radius,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: touch.pad,
  },
  buttonPrimary: { backgroundColor: colors.primary },
  buttonQuiet: { backgroundColor: colors.primarySoft, borderWidth: 1, borderColor: colors.primary },
  pressed: { opacity: 0.8 },
  disabled: { opacity: 0.4 },
  buttonText: { color: "#FFFFFF", fontSize: type.body, fontWeight: "700" },
  buttonTextQuiet: { color: colors.primary },
  card: {
    backgroundColor: colors.surface,
    borderRadius: touch.radius,
    borderWidth: 1,
    borderColor: colors.border,
    padding: touch.pad,
    gap: 8,
  },
  header: { gap: 4, marginBottom: 8 },
  title: { fontSize: type.title, fontWeight: "700", color: colors.text },
  subtitle: { fontSize: type.small, color: colors.textSoft, lineHeight: 22 },
});
