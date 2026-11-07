import React, { createContext, useContext, useMemo, useState } from "react";
import { Alert } from "./types";

interface SessionState {
  videoId: string | null;
  setVideoId(id: string | null): void;
  alerts: Alert[];
  pushAlerts(incoming: Alert[]): void;
  clearAlerts(): void;
}

const SessionContext = createContext<SessionState | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const value = useMemo<SessionState>(
    () => ({
      videoId,
      setVideoId(id) {
        setVideoId(id);
        setAlerts([]);
      },
      alerts,
      pushAlerts(incoming) {
        if (incoming.length) setAlerts((prev) => [...prev, ...incoming].slice(-20));
      },
      clearAlerts: () => setAlerts([]),
    }),
    [videoId, alerts],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession outside SessionProvider");
  return ctx;
}
