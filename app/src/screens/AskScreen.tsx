import React, { useState } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { ask } from "../api";
import { BigButton, Card, ScreenTitle } from "../components/ui";
import { useSession } from "../session";
import { colors, formatClock, touch, type } from "../theme";
import { Answer } from "../types";

const SUGGESTIONS = ["Where are my keys?", "Did I take my medicine?", "Where is my phone?"];

export function AskScreen() {
  const { videoId } = useSession();
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(q: string) {
    const text = q.trim();
    if (!text || !videoId) return;
    setAsking(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await ask(videoId, text, Date.now() / 1000));
    } catch {
      setError("I could not reach the helper just now. Please try again in a moment.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <ScreenTitle title="Ask" subtitle="Ask about your things or what you did today." />

        {!videoId && (
          <Card>
            <Text style={styles.body}>
              Start watching on the Live tab, or load the demo on Your day, and then ask me
              anything.
            </Text>
          </Card>
        )}

        <TextInput
          style={styles.input}
          value={question}
          onChangeText={setQuestion}
          placeholder="For example: where are my glasses?"
          placeholderTextColor={colors.textSoft}
          accessibilityLabel="Your question"
          returnKeyType="send"
          onSubmitEditing={() => void submit(question)}
          multiline
        />

        <BigButton
          label={asking ? "Thinking..." : "Ask"}
          onPress={() => void submit(question)}
          disabled={asking || !videoId || !question.trim()}
        />

        <View style={styles.suggestions}>
          {SUGGESTIONS.map((s) => (
            <BigButton
              key={s}
              label={s}
              tone="quiet"
              onPress={() => {
                setQuestion(s);
                void submit(s);
              }}
              disabled={asking || !videoId}
            />
          ))}
        </View>

        {error && <Text style={styles.error}>{error}</Text>}

        {answer &&
          (answer.refused ? (
            <Card style={styles.refusedCard}>
              <Text style={styles.answerLead}>I don't know for sure.</Text>
              <Text style={styles.body}>
                I would rather say so than guess and send you the wrong way. If it helps, we
                can look at your day together on the timeline.
              </Text>
            </Card>
          ) : (
            <Card style={styles.answerCard}>
              <Text style={styles.answerLead}>{answer.text}</Text>
              {answer.window && (
                <Text style={styles.meta}>
                  Around {formatClock(answer.window[0])} – {formatClock(answer.window[1])}
                </Text>
              )}
              <Text style={styles.meta}>
                Confidence: {Math.round(answer.confidence * 100)}%
              </Text>
            </Card>
          ))}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  scroll: { padding: touch.pad, gap: 12 },
  body: { fontSize: type.body, color: colors.text, lineHeight: 26 },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 2,
    borderColor: colors.border,
    borderRadius: touch.radius,
    padding: touch.pad,
    fontSize: type.body,
    color: colors.text,
    minHeight: 72,
  },
  suggestions: { gap: 10 },
  answerCard: { borderColor: colors.good, backgroundColor: colors.goodSoft },
  refusedCard: { borderColor: colors.primary, backgroundColor: colors.primarySoft },
  answerLead: { fontSize: type.heading, fontWeight: "700", color: colors.text, lineHeight: 30 },
  meta: { fontSize: type.small, color: colors.textSoft },
  error: { fontSize: type.small, color: colors.warn },
});
