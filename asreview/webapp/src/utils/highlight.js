import React from "react";

// Four opaque highlighter colors with separate values for light and dark
// mode. Calibrated for contrast and color-blind distinguishability.
export const HIGHLIGHT_COLORS = [
  { value: "yellow", label: "Yellow", light: "#ffedc2", dark: "#5c3f00" },
  { value: "green", label: "Green", light: "#b1f6ec", dark: "#004235" },
  { value: "red", label: "Red", light: "#f5cccc", dark: "#7c2c47" },
  { value: "blue", label: "Blue", light: "#d0ddeb", dark: "#2c4a68" },
];

const COLOR_MAP = Object.fromEntries(HIGHLIGHT_COLORS.map((c) => [c.value, c]));

/**
 * Resolve a color value (e.g. "yellow") to an opaque hex for the given
 * palette mode ("light" | "dark"). Falls back to the light value.
 */
export const resolveColor = (value, mode = "light") => {
  const entry = COLOR_MAP[value];
  if (!entry) return value || "#ffedc2";
  return mode === "dark" ? entry.dark : entry.light;
};

const escapeRegex = (s) => s.replace(/[.+?^${}()|[\]\\]/g, "\\$&");

const phraseToRegexSource = (phrase) => {
  const tokens = phrase.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return null;
  const parts = tokens.map((tok) => {
    let suffix = "";
    let core = tok;
    if (core.endsWith("*")) {
      core = core.slice(0, -1);
      suffix = "\\w*";
    }
    if (!core) return null;
    return escapeRegex(core) + suffix;
  });
  if (parts.some((p) => p === null)) return null;
  return parts.join("\\s+");
};

export const buildHighlightPatterns = (groups) => {
  if (!groups) return [];
  const entries = [];
  for (const g of groups) {
    if (!g || !Array.isArray(g.words)) continue;
    for (const w of g.words) {
      const src = phraseToRegexSource(w);
      if (!src) continue;
      entries.push({
        source: src,
        color: g.color,
        length: w.trim().length,
      });
    }
  }
  entries.sort((a, b) => b.length - a.length);
  return entries;
};

export const highlightText = (text, entries, mode = "light") => {
  if (!text || !entries || entries.length === 0) return text;

  const combined = new RegExp(
    "\\b(?:" + entries.map((e) => `(${e.source})`).join("|") + ")\\b",
    "gi",
  );

  const out = [];
  let lastIndex = 0;
  let match;
  let key = 0;
  while ((match = combined.exec(text)) !== null) {
    if (match.index > lastIndex) {
      out.push(text.slice(lastIndex, match.index));
    }
    let entryIdx = -1;
    for (let i = 1; i < match.length; i++) {
      if (match[i] !== undefined) {
        entryIdx = i - 1;
        break;
      }
    }
    const entry = entries[entryIdx];
    out.push(
      <mark
        key={`hl-${key++}`}
        style={{
          backgroundColor: resolveColor(entry.color, mode),
          color: "inherit",
          padding: "0 2px",
          margin: "0 -2px",
          borderRadius: 2,
        }}
      >
        {match[0]}
      </mark>,
    );
    lastIndex = match.index + match[0].length;
    if (match[0].length === 0) combined.lastIndex++;
  }
  if (lastIndex < text.length) out.push(text.slice(lastIndex));
  return out;
};

export const matchesAnyEntry = (text, entries) => {
  if (!text || !entries || entries.length === 0) return null;
  for (const entry of entries) {
    const re = new RegExp("\\b(?:" + entry.source + ")\\b", "i");
    if (re.test(text)) return entry;
  }
  return null;
};
