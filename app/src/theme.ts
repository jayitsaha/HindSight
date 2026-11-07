// Calm, high-contrast palette for users with cognitive impairment:
// large type, generous spacing, few colors, no flashing.
export const colors = {
  background: "#FBF9F4",
  surface: "#FFFFFF",
  text: "#1C2430",
  textSoft: "#4A5568",
  primary: "#2B5E8C",
  primarySoft: "#E3EDF5",
  good: "#2E7D4F",
  goodSoft: "#E4F2E9",
  warn: "#8C4A1D",
  warnSoft: "#FBEBDD",
  alert: "#9B2C2C",
  alertSoft: "#FBE4E4",
  border: "#D8D3C8",
};

export const type = {
  title: 28,
  heading: 22,
  body: 19,
  small: 16,
};

export const touch = {
  minHeight: 56,
  radius: 14,
  pad: 16,
};

export function formatClock(t: number): string {
  const s = Math.max(0, Math.round(t));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}
