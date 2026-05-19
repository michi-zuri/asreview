import React from "react";

// Module-level state: persists across RecordCard mount/unmount within a
// single page session, but resets on page reload (since the module is
// re-evaluated on a fresh load).
const state = { value: false, listeners: new Set() };

const subscribe = (listener) => {
  state.listeners.add(listener);
  return () => state.listeners.delete(listener);
};

const setValue = (v) => {
  state.value = !!v;
  state.listeners.forEach((l) => l(state.value));
};

/**
 * Frontend-only toggle for whether highlighting is currently rendered on the
 * RecordCard. Shared across all RecordCards in the session, never persisted.
 */
export const useHighlightToggle = () => {
  const [value, setLocal] = React.useState(state.value);
  React.useEffect(() => subscribe(setLocal), []);
  const toggle = React.useCallback(() => setValue(!state.value), []);
  return [value, toggle];
};
