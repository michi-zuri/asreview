import {
  appendPage,
  applyForms,
  applyRowUpdates,
  emptyState,
  markChanges,
  reconcile,
  sameSortKey,
  snapshotParams,
  sortKeyOf,
  visibleRows,
} from "./collectionModel";

// --- builders -----------------------------------------------------------

const row = (
  id,
  { version = 1, time = 100 - id, label = 1, note = null } = {},
) => ({
  record_id: id,
  version,
  state: { time, label, note, user_id: null, tags: [], lists: [] },
  title: `title-${id}`,
});

const page = (
  rows,
  {
    next = null,
    wm = 10,
    cv = 0,
    users = {},
    forms = { tags_form: [], lists_form: [] },
  } = {},
) => ({
  count: rows.length,
  next_cursor: next,
  result: rows,
  forms,
  users,
  watermark: wm,
  config_version: cv,
});

const ids = (items, { next = null, wm = 10, cv = 0 } = {}) => ({
  watermark: wm,
  config_version: cv,
  items,
  next_cursor: next,
});

// --- appendPage ----------------------------------------------------------

describe("appendPage", () => {
  it("concatenates two pages, min watermark, merged users and last forms", () => {
    const p1 = page([row(1), row(2)], {
      next: "c1",
      wm: 12,
      users: { 5: { name: "A" } },
      forms: { tags_form: ["t1"], lists_form: null },
    });
    const p2 = page([row(3)], {
      next: null,
      wm: 8,
      users: { 6: { name: "B" } },
      forms: { tags_form: ["t2"], lists_form: ["l"] },
    });
    let state = appendPage(emptyState("all"), p1);
    state = appendPage(state, p2);

    expect(state.order).toEqual([1, 2, 3]);
    expect(state.boundary).toBe("end");
    expect(state.hasMore).toBe(false);
    expect(state.watermark).toBe(8);
    expect(state.users).toEqual({ 5: { name: "A" }, 6: { name: "B" } });
    expect(state.forms).toEqual({ tags_form: ["t2"], lists_form: ["l"] });
  });

  it("keeps a re-delivered id at its position and updates its version", () => {
    let state = appendPage(
      emptyState("all"),
      page([row(1, { version: 1 }), row(2)]),
    );
    state = appendPage(state, page([row(1, { version: 2 })]));

    expect(state.order).toEqual([1, 2]);
    expect(state.versions[1]).toBe(2);
    expect(state.rows[1].version).toBe(2);
  });
});

// --- applyRowUpdates -----------------------------------------------------

describe("applyRowUpdates", () => {
  it("patches the row and version in place; order and sortKeys unchanged", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1, { time: 10 }), row(2, { time: 20 })]),
    );
    const beforeSortKeys = { ...state.sortKeys };

    state = applyRowUpdates(state, [row(1, { version: 2, time: 99 }), row(99)]);

    expect(state.order).toEqual([1, 2]);
    expect(state.sortKeys).toEqual(beforeSortKeys);
    expect(state.rows[1].version).toBe(2);
    expect(state.rows[1].state.time).toBe(99);
    expect(state.versions[1]).toBe(2);
    expect(99 in state.rows).toBe(false);
  });
});

// --- applyForms ----------------------------------------------------------

describe("applyForms", () => {
  it("replaces forms, merges users and sets configVersion, keeping the rest", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1)], {
        wm: 10,
        cv: 0,
        users: { 5: { name: "A" } },
        forms: { tags_form: ["t1"], lists_form: null },
      }),
    );

    const next = applyForms(state, {
      forms: { tags_form: ["t2"], lists_form: ["l"] },
      users: { 6: { name: "B" } },
      config_version: 7,
    });

    expect(next.forms).toEqual({ tags_form: ["t2"], lists_form: ["l"] });
    expect(next.users).toEqual({ 5: { name: "A" }, 6: { name: "B" } });
    expect(next.configVersion).toBe(7);
    expect(next.order).toEqual(state.order);
    expect(next.rows).toBe(state.rows);
    expect(next.versions).toBe(state.versions);
    expect(next.boundary).toBe(state.boundary);
    expect(next.watermark).toBe(state.watermark);
  });
});

// --- markChanges ---------------------------------------------------------

describe("markChanges", () => {
  it("patches a snapshot id with a newer version without going stale", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1, { version: 1 })]),
    );
    const res = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: true, sort_key: [99, 1] }],
    });
    expect(res.idsToPatch).toEqual([1]);
    expect(res.state.stale).toBe(false);
  });

  it("does not patch a same or older version", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1, { version: 2 })]),
    );
    const res = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: true, sort_key: [99, 1] }],
    });
    expect(res.idsToPatch).toEqual([]);
    expect(res.state.stale).toBe(false);
  });

  it("goes stale when a new id enters the view above the boundary", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)]));
    const res = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 2, version: 1, in_view: true, sort_key: [98, 2] }],
    });
    expect(res.idsToPatch).toEqual([]);
    expect(res.state.stale).toBe(true);
  });

  it("goes stale when a snapshot id leaves the view", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)]));
    const res = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: false, sort_key: null }],
    });
    expect(res.state.stale).toBe(true);
  });

  it("goes stale on a sort-key change but not on a same sort key (note edit)", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1, { time: 99 })]),
    );

    const changed = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: true, sort_key: [50, 1] }],
    });
    expect(changed.state.stale).toBe(true);

    const same = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: true, sort_key: [99, 1] }],
    });
    expect(same.state.stale).toBe(false);
  });

  it("ignores an id that is neither in view nor in the snapshot", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)]));
    const res = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 2, version: 1, in_view: false, sort_key: null }],
    });
    expect(res.idsToPatch).toEqual([]);
    expect(res.state.stale).toBe(false);
  });

  it("handles a reset with no patches", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)]));
    const res = markChanges(state, { reset: true, watermark: 20 });
    expect(res.state.stale).toBe(true);
    expect(res.state.watermark).toBe(20);
    expect(res.idsToPatch).toEqual([]);
    expect(res.configChanged).toBe(false);
    expect(res.state.rewound).toBe(false);
  });

  it("marks a reset with a lower watermark as rewound (sticky)", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)], { wm: 10 }));
    let res = markChanges(state, { reset: true, watermark: 3 });
    expect(res.state.stale).toBe(true);
    expect(res.state.rewound).toBe(true);
    expect(res.state.watermark).toBe(3);
    // A later normal reset does not clear it; only a fresh snapshot does.
    res = markChanges(res.state, { reset: true, watermark: 4 });
    expect(res.state.rewound).toBe(true);
  });

  it("keeps stale sticky after a later empty poll", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)]));
    state = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: false, sort_key: null }],
    }).state;
    expect(state.stale).toBe(true);

    state = markChanges(state, {
      watermark: 11,
      config_version: 0,
      changes: [],
    }).state;
    expect(state.stale).toBe(true);
  });

  it("reports a config_version change without going stale", () => {
    let state = appendPage(emptyState("relevant"), page([row(1)], { cv: 0 }));
    const res = markChanges(state, {
      watermark: 11,
      config_version: 1,
      changes: [],
    });
    expect(res.configChanged).toBe(true);
    expect(res.state.stale).toBe(false);
    expect(res.state.configVersion).toBe(1);
  });
});

// --- reconcile -----------------------------------------------------------

describe("reconcile", () => {
  it("drops ids the server no longer lists", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1), row(2), row(3)]),
    );
    const items = [
      [3, 1, sortKeyOf("relevant", row(3))],
      [1, 1, sortKeyOf("relevant", row(1))],
    ];
    const res = reconcile(state, ids(items, { next: null, wm: 11 }));
    expect(res.state.order).toEqual([3, 1]);
    expect(res.idsToFetch).toEqual([]);
  });

  it("inserts a new id at its server position and fetches it", () => {
    let state = appendPage(emptyState("relevant"), page([row(1), row(3)]));
    const items = [
      [5, 1, sortKeyOf("relevant", row(5))],
      [3, 1, sortKeyOf("relevant", row(3))],
      [1, 1, sortKeyOf("relevant", row(1))],
    ];
    const res = reconcile(state, ids(items, { next: null, wm: 11 }));
    expect(res.state.order).toEqual([5, 3, 1]);
    expect(res.idsToFetch).toEqual([5]);
  });

  it("reorders to server order", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1), row(2), row(3)]),
    );
    const items = [
      [3, 1, sortKeyOf("relevant", row(3))],
      [1, 1, sortKeyOf("relevant", row(1))],
      [2, 1, sortKeyOf("relevant", row(2))],
    ];
    const res = reconcile(state, ids(items, { next: null, wm: 11 }));
    expect(res.state.order).toEqual([3, 1, 2]);
  });

  it("fetches only missing or newer ids", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1, { version: 1 }), row(2, { version: 2 })]),
    );
    const items = [
      [1, 2, sortKeyOf("relevant", row(1, { version: 2 }))],
      [2, 2, sortKeyOf("relevant", row(2, { version: 2 }))],
      [3, 1, sortKeyOf("relevant", row(3))],
    ];
    const res = reconcile(state, ids(items, { next: null, wm: 11 }));
    expect(res.idsToFetch).toEqual([1, 3]);
  });

  it("keeps dirty ids at their place and never fetches them", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(10), row(20), row(30), row(40), row(50)]),
    );
    // Server reorders, drops 40, and bumps versions of 10 (dirty) and 30.
    const items = [
      [50, 1, sortKeyOf("relevant", row(50))],
      [30, 2, sortKeyOf("relevant", row(30, { version: 2 }))],
      [20, 1, sortKeyOf("relevant", row(20))],
      [10, 3, sortKeyOf("relevant", row(10, { version: 3 }))],
    ];
    const res = reconcile(state, ids(items, { next: null, wm: 11 }), [10, 40]);
    // Dirty 10 (old front) stays front; dirty 40 (dropped) stays after its
    // old previous neighbour 30. Only non-dirty newer 30 is fetched.
    expect(res.state.order).toEqual([10, 50, 30, 40, 20]);
    expect(res.idsToFetch).toEqual([30]);
  });

  it("clears stale and updates boundary/hasMore/watermark", () => {
    let state = appendPage(
      emptyState("relevant"),
      page([row(1)], { next: "c1" }),
    );
    state = { ...state, stale: true };
    const res = reconcile(
      state,
      ids([[1, 1, sortKeyOf("relevant", row(1))]], { next: "c2", wm: 15 }),
    );
    expect(res.state.stale).toBe(false);
    expect(res.state.boundary).toBe("c2");
    expect(res.state.hasMore).toBe(true);
    expect(res.state.watermark).toBe(15);
  });

  it("reconcile + applyRowUpdates equals a fresh load (equivalence)", () => {
    const subset = "relevant";
    const truth = [
      row(5, { version: 2, time: 95 }),
      row(3, { version: 1, time: 97 }),
      row(1, { version: 3, time: 99 }),
    ];
    // A stale snapshot from older data: missing 5, record 1 at an old version.
    let state = appendPage(
      emptyState(subset),
      page(
        [row(1, { version: 1, time: 90 }), row(3, { version: 1, time: 97 })],
        {
          next: null,
          wm: 5,
        },
      ),
    );

    const items = truth.map((r) => [
      r.record_id,
      r.version,
      sortKeyOf(subset, r),
    ]);
    const rec = reconcile(state, ids(items, { next: null, wm: 9, cv: 0 }));
    state = rec.state;
    state = applyRowUpdates(
      state,
      truth.filter((r) => rec.idsToFetch.includes(r.record_id)),
    );

    const fresh = appendPage(
      emptyState(subset),
      page(truth, { next: null, wm: 9, cv: 0 }),
    );

    const pick = (s) => ({
      order: s.order,
      rows: s.rows,
      versions: s.versions,
      sortKeys: s.sortKeys,
      boundary: s.boundary,
      hasMore: s.hasMore,
    });
    expect(pick(state)).toEqual(pick(fresh));
  });

  it("never mutates its inputs", () => {
    const deepFreeze = (obj) => {
      if (obj && typeof obj === "object" && !Object.isFrozen(obj)) {
        Object.freeze(obj);
        Object.keys(obj).forEach((key) => deepFreeze(obj[key]));
      }
      return obj;
    };

    let state = appendPage(emptyState("relevant"), page([row(1), row(2)]));
    deepFreeze(state);

    const p2 = page([row(3)], { next: "c" });
    deepFreeze(p2);
    appendPage(state, p2);

    const updates = [row(1, { version: 2 })];
    deepFreeze(updates);
    applyRowUpdates(state, updates);

    const changes = {
      watermark: 11,
      config_version: 0,
      changes: [{ record_id: 1, version: 2, in_view: true, sort_key: [99, 1] }],
    };
    deepFreeze(changes);
    markChanges(state, changes);

    const idsResponse = ids([
      [1, 1, [99, 1]],
      [2, 1, [98, 2]],
    ]);
    deepFreeze(idsResponse);
    reconcile(state, idsResponse);

    // In strict mode a mutation of a frozen input would have thrown.
    expect(state.order).toEqual([1, 2]);
  });
});

// --- sortKeyOf / sameSortKey ---------------------------------------------

describe("sortKeyOf / sameSortKey", () => {
  it("builds the right key per subset and compares keys correctly", () => {
    expect(sortKeyOf("unlabeled", row(7))).toBe(7);
    expect(sortKeyOf("relevant", row(7, { time: 12.5 }))).toEqual([12.5, 7]);
    expect(sortKeyOf("relevant", row(7, { time: null }))).toEqual([null, 7]);

    expect(sameSortKey([1.5, 2], [1.5, 2])).toBe(true);
    expect(sameSortKey([1.5, 2], [1.5, 3])).toBe(false);
    expect(sameSortKey(3, 3)).toBe(true);
    expect(sameSortKey(null, 3)).toBe(false);
  });
});

describe("visibleRows", () => {
  it("returns only the loaded rows in display order", () => {
    let state = appendPage(emptyState("relevant"), page([row(1), row(2)]));
    expect(visibleRows(state)).toEqual([row(1), row(2)]);

    // A new id in `order` without a row is skipped until fetched.
    state = reconcile(
      state,
      ids([
        [1, 1, sortKeyOf("relevant", row(1))],
        [3, 1, sortKeyOf("relevant", row(3))],
        [2, 1, sortKeyOf("relevant", row(2))],
      ]),
    ).state;
    expect(visibleRows(state)).toEqual([row(1), row(2)]);
  });
});

describe("snapshotParams", () => {
  it("sends 'end' before anything is loaded, else the boundary", () => {
    expect(snapshotParams(emptyState("relevant"))).toEqual({ boundary: "end" });
    const state = appendPage(
      emptyState("relevant"),
      page([row(1)], { next: "c1" }),
    );
    expect(snapshotParams(state)).toEqual({ boundary: "c1" });
  });
});
