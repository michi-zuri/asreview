// Collection snapshot state machine (Spec 0001 Part 4).
//
// Pure — no React, no network, no Date.now(), and no input is ever mutated
// (D12). Every function returns a new object. The state is a plain object
// keyed by String(record_id) so this module ports almost line for line to
// Elm, and these Jest tests are the specification for that port.
//
// See specs/0001-collection-snapshot.md, "Client model module".

/**
 * A fresh, empty snapshot state for the given `subset`.
 *
 * @param {string} subset "relevant" | "irrelevant" | "all" | "unlabeled" | …
 * @returns {object} the empty state
 */
export const emptyState = (subset) => ({
  subset,
  order: [],
  rows: {},
  versions: {},
  sortKeys: {},
  boundary: null,
  hasMore: false,
  watermark: null,
  configVersion: null,
  forms: { tags_form: null, lists_form: null },
  users: {},
  stale: false,
  rewound: false,
});

/**
 * The server's sort key for a row: the bare `record_id` on the "unlabeled"
 * subset, otherwise `[state.time ?? null, record_id]`.
 *
 * @param {string} subset the snapshot's subset
 * @param {object} row a `/labeled` row
 * @returns {number | [number | null, number]} the sort key
 */
export const sortKeyOf = (subset, row) => {
  if (subset === "unlabeled") return row.record_id;
  return [row.state?.time ?? null, row.record_id];
};

/**
 * Whether two sort keys are equal. Arrays compare element-wise with `===`;
 * anything else compares with `===`. Null-safe: `sameSortKey(null, 3)` is
 * false.
 *
 * @param {*} a one sort key
 * @param {*} b the other sort key
 * @returns {boolean} whether they are equal
 */
export const sameSortKey = (a, b) => {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  return false;
};

/**
 * Append one `/labeled` page to the snapshot.
 *
 * New ids are pushed in page order; a re-delivered id updates its row,
 * version and sort key but keeps its position. The boundary becomes the
 * page's `next_cursor` ("end" when null), the watermark the LOWEST seen so
 * far (every loaded row is at least that new, D11), and forms/users come
 * from the latest page.
 *
 * @param {object} state the snapshot state
 * @param {object} page a `/labeled` page
 * @returns {object} the new state
 */
export const appendPage = (state, page) => {
  const order = state.order.slice();
  const rows = { ...state.rows };
  const versions = { ...state.versions };
  const sortKeys = { ...state.sortKeys };
  for (const row of page.result || []) {
    const id = row.record_id;
    if (!order.includes(id)) {
      order.push(id);
    }
    rows[id] = row;
    versions[id] = row.version;
    sortKeys[id] = sortKeyOf(state.subset, row);
  }
  return {
    ...state,
    order,
    rows,
    versions,
    sortKeys,
    boundary: page.next_cursor ?? "end",
    hasMore: page.next_cursor != null,
    watermark:
      state.watermark == null
        ? page.watermark
        : Math.min(state.watermark, page.watermark),
    configVersion: page.config_version,
    forms: page.forms,
    users: { ...state.users, ...(page.users || {}) },
  };
};

/**
 * Patch rows in place without moving them (D1/D2).
 *
 * For each row whose id is in `order`, replace `rows[id]` and raise
 * `versions[id]` to `max(old, row.version)`. Order and sortKeys are left
 * untouched — a sort-key change surfaces as staleness on the next poll.
 * Ids not in `order` are ignored. `users` is merged in.
 *
 * @param {object} state the snapshot state
 * @param {object[]} rows the rows to patch
 * @param {object} [users] extra users to merge
 * @returns {object} the new state
 */
export const applyRowUpdates = (state, rows, users = {}) => {
  const newRows = { ...state.rows };
  const newVersions = { ...state.versions };
  for (const row of rows || []) {
    const id = row.record_id;
    if (!state.order.includes(id)) continue;
    newRows[id] = row;
    newVersions[id] = Math.max(newVersions[id] ?? 0, row.version ?? 0);
  }
  return {
    ...state,
    rows: newRows,
    versions: newVersions,
    users: { ...state.users, ...users },
  };
};

/**
 * Apply a `/labeled` page's forms, users and `config_version` to the snapshot
 * (a config change, D5 — not membership, so no banner). The page's rows are
 * ignored.
 *
 * @param {object} state the snapshot state
 * @param {object} page a `/labeled` page (forms, users, config_version)
 * @returns {object} the new state
 */
export const applyForms = (state, page) => ({
  ...state,
  forms: page.forms,
  users: { ...state.users, ...(page.users || {}) },
  configVersion: page.config_version,
});

/**
 * Apply a `/labeled/changes` response to the snapshot.
 *
 * Returns `{state, idsToPatch, configChanged}`. A `reset` only sets `stale`
 * and the new watermark; a reset whose watermark is LOWER than ours (the
 * server's version counter went backwards, e.g. a restored database) also
 * sets the sticky `rewound`: our versions can no longer be compared with the
 * server's, so Refresh must start a fresh snapshot instead of reconciling.
 * Otherwise each change patches an in-snapshot id
 * with a newer version, and marks the snapshot stale when a change's
 * `in_view` disagrees with whether the id is in the snapshot, or when an
 * in-view id's sort key differs from the key at its shown position. `stale`
 * is sticky.
 *
 * @param {object} state the snapshot state
 * @param {object} response a `/labeled/changes` response
 * @returns {{state: object, idsToPatch: number[], configChanged: boolean}}
 */
export const markChanges = (state, response) => {
  if (response.reset) {
    return {
      state: {
        ...state,
        stale: true,
        rewound:
          state.rewound ||
          (state.watermark != null && response.watermark < state.watermark),
        watermark: response.watermark,
      },
      idsToPatch: [],
      configChanged: false,
    };
  }
  const idsToPatch = [];
  let stale = state.stale;
  for (const change of response.changes || []) {
    const id = change.record_id;
    const inSnap = state.order.includes(id);
    if (inSnap && change.version > (state.versions[id] ?? 0)) {
      idsToPatch.push(id);
    }
    if (change.in_view !== inSnap) {
      stale = true;
    }
    if (
      change.in_view &&
      inSnap &&
      !sameSortKey(change.sort_key, state.sortKeys[id])
    ) {
      stale = true;
    }
  }
  return {
    state: {
      ...state,
      stale,
      watermark: response.watermark,
      configVersion: response.config_version,
    },
    idsToPatch,
    configChanged: response.config_version !== state.configVersion,
  };
};

/**
 * Reconcile the snapshot with a `/labeled/ids` response (D4).
 *
 * Returns `{state, idsToFetch, configChanged}`. The new order is the server
 * order, except dirty ids keep their place: each is re-inserted right after
 * its nearest previous neighbour in the old order that is still present (or
 * at the front), whether or not the server still lists it. `idsToFetch` are
 * the non-dirty server ids that are missing or have a newer version, in
 * server order. `stale` is cleared.
 *
 * @param {object} state the snapshot state
 * @param {object} idsResponse a `/labeled/ids` response
 * @param {number[]} [dirtyIds] ids with unsaved edits, kept in place
 * @returns {{state: object, idsToFetch: number[], configChanged: boolean}}
 */
export const reconcile = (state, idsResponse, dirtyIds = []) => {
  const items = idsResponse.items || [];
  const dirtySet = new Set(dirtyIds.filter((id) => state.order.includes(id)));

  let order = items.map((item) => item[0]).filter((id) => !dirtySet.has(id));
  for (const id of state.order) {
    if (!dirtySet.has(id)) continue;
    const idx = state.order.indexOf(id);
    let inserted = false;
    for (let j = idx - 1; j >= 0; j -= 1) {
      const pos = order.indexOf(state.order[j]);
      if (pos !== -1) {
        order.splice(pos + 1, 0, id);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      order.unshift(id);
    }
  }

  const sortKeys = {};
  for (const item of items) {
    if (!dirtySet.has(item[0])) sortKeys[item[0]] = item[2];
  }
  for (const id of state.order) {
    if (dirtySet.has(id)) sortKeys[id] = state.sortKeys[id];
  }

  const idsToFetch = [];
  for (const item of items) {
    const id = item[0];
    if (dirtySet.has(id)) continue;
    if (!(id in state.rows) || item[1] > (state.versions[id] ?? 0)) {
      idsToFetch.push(id);
    }
  }

  const rows = {};
  const versions = {};
  for (const id of order) {
    if (id in state.rows) rows[id] = state.rows[id];
    if (id in state.versions) versions[id] = state.versions[id];
  }

  return {
    state: {
      ...state,
      order,
      rows,
      versions,
      sortKeys,
      boundary: idsResponse.next_cursor ?? "end",
      hasMore: idsResponse.next_cursor != null,
      watermark: idsResponse.watermark,
      stale: false,
      configVersion: idsResponse.config_version,
    },
    idsToFetch,
    configChanged: idsResponse.config_version !== state.configVersion,
  };
};

/**
 * The rows that are in `order` and already loaded (for rendering).
 *
 * @param {object} state the snapshot state
 * @returns {object[]} the loaded rows, in display order
 */
export const visibleRows = (state) =>
  state.order.filter((id) => id in state.rows).map((id) => state.rows[id]);

/**
 * The query params for a snapshot poll/refresh (helper for 10i).
 *
 * @param {object} state the snapshot state
 * @returns {{boundary: string}} the `boundary` to send
 */
export const snapshotParams = (state) => ({
  boundary: state.boundary ?? "end",
});
