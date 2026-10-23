# Spec 0001: Collection as a versioned snapshot

Status: agreed with the user on 2026-10-20. Author: Claude (requirements); Amp adds implementation notes **below** the line at the end and does not rewrite this spec. Any objection goes back to the user, not into a silent change.

Ships in four parts. Each part is merged and verified before the next starts.

| Part | What | Depends on |
|---|---|---|
| 1 | Cursor fix (interim) + outdated test | — |
| 2 | Slim rows; stop whole-list reloads | 1 |
| 3 | Change tracking in the database (versions, change log, migration ledger) | 2 |
| 4 | Live staleness detection, banner, cheap Refresh | 3 |

---

## Goal

The Collection page (Relevant / Not relevant / Full history / Unlabeled, with filters) works like this:

- It shows a **snapshot**. Records never appear, disappear or move by themselves.
- **Changes to a visible card show immediately** (your own edits, and other reviewers' edits detected by polling).
- **Membership** means: should this card be in the list, given the current filter, sort order and loaded boundary? When the snapshot's membership or order differs from what a full reload would show, a **banner** appears: "This view is out of date — Refresh". Clicking Refresh brings the view to exactly what a full reload would show, while downloading only what changed.
- A full page load (F5, or opening the Collection) always fetches everything fresh. It never uses a client cache.
- Today's behaviour (each found PDF reloads every loaded page, several MB per minute) and the "invalid cursor" 400 error are gone.

## Non-goals

- Moving to Postgres now. The design must not block it (see Decisions), but only SQLite is implemented.
- Keeping a client cache across page reloads (IndexedDB etc.).
- Upgrading react-query or any other frontend library. The frontend will later be ported to Elm.
- Moving the existing `_ensure_*` / `_fix_*` functions into the new migration ledger. That is a separate cleanup task.
- Showing LLM status changes live in the Collection (`llm_dispatch` / `llm_results` are not tracked in this spec).
- Changing the review (screening) page.

## Decisions (with reasons)

D1. **Snapshot + banner, no automatic removal.** Even when your own action (e.g. a PDF found for a record in a "Full text: false" view) means the card no longer matches, it stays and the banner appears. *Reason:* the list never jumps under the user; one rule covers own and others' changes.

D2. **In-place content updates are immediate; membership goes through the banner.** When a visible card's content changes (your edit, or another reviewer's edit found by polling), the card updates in place at once. *Reason:* otherwise your own edit would seem to undo itself.

D3. **"The view" = everything loaded so far**, from the top down to the boundary cursor (the last loaded row), not just the cards on screen. A new match beyond the boundary does not trigger the banner, because scrolling will reach it.

D4. **Refresh is a reconciliation, not a reload.** The server returns the ordered list of `(record_id, version, sort_key)` for the current filter, up to the boundary. The client drops, adds and reorders, and fetches full rows only for ids it doesn't have or whose version is newer. After Refresh, the view must equal a full reload down to the same boundary.

D5. **Form definitions and users are sent once per response, not once per row.** Records store tag selections as references to option ids (normalised `tags` table with a foreign key; deleting a used option is refused), so there is only ever one current form definition. Sending it per row is pure duplication. It also lets pages loaded at different times show different forms. Rows keep `user_id`; names come from a `users` map in the response.

D6. **Change tracking lives in the database, written by triggers.** It covers every writer: the Flask app, the LLM worker and the Zotero sync (separate processes), and the SQLAlchemy `DataStore` connection that writes `record.attachment`. *Reason:* code in the app layer would miss writers and would have to be remembered by every future write path.

D7. **Versions come from a per-project counter row incremented inside the writing transaction.** In Postgres the row lock is held until commit, so version order equals commit order, and "everything since version V" can never skip a change that commits late. A plain autoincrement sequence would only be safe in SQLite. *Cost:* writes within one project are serialised, which is irrelevant at this scale.

D8. **The fork gets its own migration ledger.** `PRAGMA user_version` belongs to upstream: `_is_valid` requires exactly 4, so the fork must not touch it. The ledger is a table of named, ordered revisions, each applied once inside `BEGIN IMMEDIATE`, because several processes open the same database. Its revision ids are in Alembic style, so a later switch to Alembic/Postgres is mechanical. Tables carry `project_id` from the start, since a Postgres move would merge the per-project databases.

D9. **Triggers are re-asserted on every read-write open, not only by the ledger.** `_fix_tags_schema` drops and rebuilds the `tags` table, and SQLite deletes a table's triggers together with the table. One function `_ensure_change_tracking_triggers()` (`CREATE TRIGGER IF NOT EXISTS`) runs last in `_is_valid`'s read-write path and after `create_tables`.

D10. **`results.last_active` changes are NOT tracked.** The heartbeat writes it every 60 s per open record. Tracking it would create endless versions with no visible effect. All other results columns that affect the card or membership are tracked: `label, time, note, user_id, assigned_at`, plus INSERT and DELETE.

D11. **Read the watermark before the data.** Each list or reconciliation response includes the project version (`watermark`), read **before** the rows. *Reason:* `/labeled` reads results through `sqlite3` and records through the SQLAlchemy `DataStore` (two connections, two snapshots). Reading the watermark first means a concurrent change can only make the data newer than the watermark. That leads to a harmless re-check on the next poll, never a missed change.

D12. **The client logic is pure functions in their own module**, with no React and no network, tested with Jest. *Reason:* it ports almost line by line to Elm, and its tests become the specification for the port.

D13. Own edits made in the Collection return the updated row with its new version, and the client patches that card. They get no other special handling: if the edit changes membership, the next poll raises the banner like any other change.

## Implementer's defaults (user accepted; change only with the user)

- Poll every 15 s. Pause while `document.hidden`; poll once immediately when the tab becomes visible again.
- If a poll reports more than 200 changes, or `since` is older than the oldest retained log entry, the server answers `{"reset": true}` and the client shows the banner. Refresh still works, because reconciliation does not depend on the log.
- Prune change-log entries older than 7 days, at most once per hour per project, from the changes endpoint (opportunistic, no new background job).
- The chip counts (`labeled_stats`) are refetched whenever a poll returns at least one change.
- Refresh does not remove a card with unsaved edits; it is removed by the next Refresh after saving.
- Changing the filter, the sort or the subset starts a new snapshot (fresh load, no reuse of the old one). Navigating away from the Collection and back also loads fresh.
- Cursor errors stay HTTP 400: a malformed cursor or one made for another filter is a client bug, not a state to recover from.

---

## Part 1: Cursor fix (interim) + outdated test

**Facts (verified):** `LabeledRecord.js` has `getNextPageParam: (lastPage) => lastPage.next_cursor ?? false`. In react-query 3.39.3, an invalidated infinite query refetches every loaded page and computes each page's param with `getNextPageParam`. It only stops on `undefined`, so after the list has shrunk it requests `cursor=false`. The server then returns 400 "invalid cursor: Invalid base64-encoded string…" and the whole list shows an error. A cursor stores sort-key **values** `(time, record_id)` plus a filter hash and is never looked up, so a record that left the filter cannot break a cursor.

**Change:**
- `?? false` → `?? undefined`. `hasNextPage` stays correct, because v3 treats `undefined` as "no next page".
- `tests/…/test_llm_mapping.py::test_list_item_cleaning`: list items containing `,` or `;` are **allowed** (user decision, 2026-10-19). Update the test to expect them to be kept. Leave the code unchanged.

**Example:** Given "Relevant + Full text: false" with 3 pages loaded, when a PDF is found and the list is refetched with only 2 pages of matches, then no request carries `cursor=false` and the list shows 2 pages without an error.

---

## Part 2: Slim rows; stop whole-list reloads

**Change (backend, `api_get_labeled`):**
- Response shape: `{ result: [rows], next_cursor, forms: { tags_form, lists_form }, users: { <id>: summary } }`. Rows no longer carry `tags_form`, `lists_form` or the embedded `state.user`; they keep `state.user_id`.
- **Measure before and after** on a real project page (bytes of one `/labeled` response with 50 rows), and put both numbers in the commit message.

**Change (frontend):**
- `LabeledRecord.js` / `RecordCard.js`: take the forms and users from the response, not from the row.
- The card React key becomes `record-card-<project_id>-<record_id>`. The current key embeds JSON of the note, tags and lists, which remounts the card on every change.
- Remove every `invalidateQueries(["fetchLabeledRecord", …])` (in `RecordCard.js` at the "live check found a PDF" effect and in `invalidateAttachment`; in `RecordCardLabeler.js` 3×). Own edits patch the edited row in the cached pages (`setQueryData`). The live PDF check updates only that card's attachment display.
- The Collection query gets `refetchOnWindowFocus: false`, and fresh data on mount (D-default: no snapshot reuse after navigation).
- Until Part 4 ships, membership changes only appear after a reload. This follows from D1 and is accepted.

**Examples:**
- Given a "Full text: false" view with 150 loaded records, when 10 PDFs are found while scrolling, then no `/labeled` request is made, the 10 cards show their PDF, and all 150 stay in place.
- Given a card in the Relevant view, when I change its tags and save, then the card shows the new tags immediately, without a `/labeled` request and without the card remounting (an open abstract stays open).
- Given an admin renames a tag option, when I reload the Collection, then every card shows the new name (there is one form per response).

---

## Part 3: Change tracking in the database

**Schema** (new ledger revision `0001_change_tracking`; all tables carry `project_id TEXT NOT NULL`):

- `fork_migrations(revision TEXT PRIMARY KEY, applied_at REAL NOT NULL)`: the ledger.
- `project_state(project_id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0, config_version INTEGER NOT NULL DEFAULT 0)`: exactly one row per project database, created by the revision.
- `record_versions(project_id, record_id INTEGER, version INTEGER NOT NULL, PRIMARY KEY(project_id, record_id))`: the current version of each record. A record without a row has version 0.
- `change_log(project_id, version INTEGER, record_id INTEGER, source TEXT NOT NULL, changed_at REAL NOT NULL, PRIMARY KEY(project_id, version))`. `source` ∈ `results | tags | lists | attachment`. Index on `(project_id, changed_at)` for pruning.

**Ledger mechanics:** an ordered list of `(revision_id, apply_fn)` in one module, e.g. `asreview/database/fork_migrations.py`. On read-write open, inside `BEGIN IMMEDIATE`: read the applied revisions, apply the missing ones in order, insert their ledger rows, commit. A second process that waits on the lock then sees them applied and does nothing.

**Triggers** (`_ensure_change_tracking_triggers()`, see D9). Each trigger does, per affected row: increment `project_state.version`; upsert `record_versions` for that record to the new version; insert a `change_log` row.
- `results`: AFTER INSERT; AFTER DELETE; AFTER UPDATE OF `label, time, note, user_id, assigned_at` (**not** `last_active`, D10).
- `tags`: AFTER INSERT / UPDATE / DELETE.
- `lists`: AFTER INSERT / UPDATE / DELETE.
- `record`: AFTER UPDATE OF `attachment` (source `attachment`).
- `tag_groups`, `tag_options`, `list_containers`: AFTER INSERT / UPDATE / DELETE → increment `version` and set `config_version = version`. No `record_versions` or `change_log` rows.
- `project_id` is read from the single `project_state` row inside the trigger.

**API additions:**
- Every row returned by `/labeled` carries `version`. Each `/labeled` response carries `watermark` (read before the rows, D11) and `config_version`.

**Examples:**
- Given a fresh project at version 0, when record 7 is labeled relevant, then `project_state.version` = 1, `record_versions[7]` = 1, and `change_log` has `(1, 7, 'results')`.
- Given a reviewer has record 7 open, when 10 heartbeats arrive, then no version changes.
- Given the Zotero sync process stores an attachment key on record 9, then `record_versions[9]` increases and `change_log` has source `attachment` (written via the SQLAlchemy connection, outside Flask).
- Given a legacy project whose `tags` table is rebuilt by `_fix_tags_schema` on open, when it has been opened read-write, then all tracking triggers exist.
- Given two processes open the same un-migrated database at the same time, then `0001_change_tracking` is applied exactly once.

---

## Part 4: Live staleness detection, banner, cheap Refresh

**New endpoints** (same auth decorators as `/labeled`; same `subset`, `filter` and `latest_first` params):

1. `GET /projects/<id>/labeled/changes?since=<V>&boundary=<cursor|end>&…filter params`
   → `{ watermark, config_version, changes: [{record_id, version, in_view, sort_key}] }` or `{ "reset": true, watermark }`.
   `in_view` = whether the record would appear in a full reload of this filter **at or before the boundary**. `sort_key` = `(time, record_id)` for labeled subsets, `record_id` for unlabeled. One entry per record (latest version), from `change_log` rows with `version > since`.
2. `GET /projects/<id>/records?ids=1,2,3` → `{ result: [rows], watermark }`: the rows in the same shape as `/labeled`, any filter ignored. Used for in-place patches and Refresh.
3. `GET /projects/<id>/labeled/ids?boundary=<cursor|end>&…filter params`
   → `{ watermark, config_version, items: [[record_id, version, sort_key], …], next_cursor }`: the full ordered membership from the top down to the boundary. The post-scan filters (pdf, tags, abstract, invalid_tags) apply exactly as in `/labeled`.

**Client model module** (pure, no React, no network; e.g. `src/ProjectComponents/HistoryComponents/collectionModel.js`):
- State: `{ order: [record_id], rows: {id → row}, versions: {id → v}, sortKeys: {id → key}, boundary, watermark, configVersion, stale: bool }`.
- `appendPage(state, response)`; `applyRowUpdates(state, rows)` (patch in place, keep order); `markChanges(state, changes) → {state, idsToPatch}` (ids in the snapshot with a newer version → patch; `stale = true` when an id's `in_view` disagrees with whether it's in `order`, or an in-view id's `sort_key` differs from the snapshot's); `reconcile(state, idsResponse, dirtyIds) → {state, idsToFetch}` (new `order` = server order, except dirty ids kept at their old position; fetch the ids that are missing or have a newer version; `stale = false`).
- The React side only wires these up: the poll timer, fetches, rendering, and the banner.

**Behaviour:**
- Poll `changes` every 15 s (defaults apply). Patch cards for `idsToPatch` via `/records?ids=`. Show the banner while `stale`.
- If `config_version` changed, refetch the forms once (from a `/labeled` first page or a small forms endpoint; implementer's choice) and re-render; that is not membership, so no banner.
- Refresh = `labeled/ids` → `reconcile` → `/records?ids=` for `idsToFetch` → done. No page refetching anywhere.
- Banner text: "This view is out of date." Button: "Refresh". No counts or details (user decision: generic message).

**Examples:**
- Given I'm in "Unlabeled + Full text: false" with 100 records loaded, when the live check finds a PDF for record 42, then card 42 shows its PDF immediately, stays in place, and within one poll the banner appears. When I click Refresh, card 42 is gone, nothing else moves, and the network shows one `labeled/ids` call and no `/records` call.
- Given a colleague labels record 88 relevant, and 88 sorts above my boundary in the Relevant view, then the banner appears. Refresh inserts card 88 at its place and fetches exactly one row.
- Given a colleague edits the note of a card I have loaded in Relevant, then the note updates in place within one poll and no banner appears (membership unchanged).
- Given a colleague labels a record that would sort below my boundary, then nothing visible happens (it'll appear when I scroll).
- Given I'm editing tags on card 15 and it no longer matches, when I click Refresh, then card 15 stays until a Refresh after I save.
- Given the tab was hidden for an hour with 500 changes, when it becomes visible, then the poll returns `reset`, the banner appears, and Refresh results in the same view as F5 to the same boundary.
- Given I press F5, then everything is fetched fresh (no client cache).

---

## Proposed tests

**Part 1:** the updated `test_list_item_cleaning`. Jest test of the extracted `getNextPageParam` (returns `undefined` when `next_cursor` is null).

**Part 2:** pytest on `/labeled`: rows contain no `tags_form`, `lists_form` or `user` object; the response has `forms` and `users`; every `user_id` in rows is a key in `users`. Also a manual measurement (before/after bytes) in the commit message.

**Part 3** (pytest, `tests/test_change_tracking.py`, DB-level, no Flask):
- Ledger: applied once; idempotent on re-open; two concurrent openers (threads with separate connections) → one ledger row.
- One test per tracked write path: `label_record`, `update_result`, `update_note`, `_save_tags`, `replace_lists`, `DataStore.set_attachment`-style write, `delete_result`, `checkout_oldest_dispatched`, `reassign_stale` → the version increases and the change_log gets the correct source.
- `touch_last_active` → no version change.
- Config edits → `config_version` changes; no change_log rows.
- Triggers exist after: `create_tables`; opening a legacy DB that goes through `_fix_tags_schema`.
- Version order equals commit order under two writer threads.

**Part 4:**
- pytest (Flask API): `changes` returns the right `in_view` for a record that left the filter (pdf filter: post-scan path) and for one that entered it above and below the boundary; `reset` past 200 changes and after pruning; `labeled/ids` matches the ids and order of `/labeled` pages concatenated down to the same boundary (**the key equivalence test**); `records?ids` returns the same row shape.
- Jest (pure module, the core of the frontend tests): `markChanges` (newer version → patch; `in_view` mismatch → stale; sort-key change → stale; beyond the boundary → not stale); `reconcile` (drops, inserts, reorders, keeps dirty cards, fetches only missing or newer ids); after `reconcile`, the state equals a fresh load built from the same server data.

## Open risks

- **Flask API tests need the Synergy dataset over the network** (`synergy:van_der_Valk_2021`). They hang offline. A small local fixture dataset would make Part 2/4 tests runnable anywhere. This is a separate task, but worth doing first if DeepSeek runs offline.
- **`in_view` cost:** answering it for the pdf, tag or abstract filters needs the same Python post-scan as `/labeled`, down to the boundary. It's bounded by the boundary and the 200-change cap; measure on the largest project.
- **Bulk writes** (`_replace_results_from_df` deletes all results; project import/restore) produce one log row per record → `reset`. That's correct, but it may make those operations slower. Measure on the largest project.
- **Unlabeled + user filter** means "assigned to". Checkout and reassignment change membership there; they are tracked (`user_id`, `assigned_at`) but deserve an explicit test.
- **No frontend test runner is set up yet** beyond the CRA stub. `react-scripts test` works without extra dependencies; Part 1 is the place to prove that, with the first real Jest test.
- `llm_worker.py` has `if __name__ == "__main__": main()` in the middle of the file. Unrelated; don't touch it here.

---
<!-- Amp: implementation notes below this line. Do not edit above. -->

## Implementation notes (planner)

Written 2026-10-23 against HEAD `416bc150`. Everything above the line is the agreed spec, unchanged. These notes say how to build it. The executor prompts are `llm_prompts/10a.txt` … `10i.txt`, each pasted below `llm_prompts/context.txt`.

### Questions for the user (answer before the task named)

Each item is a planner addition or interpretation that goes beyond the literal spec. The prompts follow the default shown. If you say no, the change needed is noted.

| # | Before | Point | Default in the prompts | If rejected |
|---|---|---|---|---|
| Q1 | 10c | D1 vs today: in the Collection, "Change to …" currently collapses the card away. D1 says cards never leave by themselves, so the Collection passes `transitionType={null}`. The card stays and shows its new label; the banner (Part 4) handles membership. | stays | keep `transitionType="collapse"` in `LabeledRecord.js` |
| Q2 | 10b | D13 is delivered in Part 2 already: `PUT record` and `PUT note` return the updated row in `/labeled` shape (`PUT note` keeps `success: true`). Part 2's "own edits patch the edited row" needs it. | yes | none (needed for Part 2) |
| Q3 | 10d | `project_state` gets two extra columns, `pruned_version` and `last_pruned_at`. The spec's "since older than the oldest retained entry" and "at most once per hour" need them. | added | detect via `MIN(version)` of `change_log` and keep the hourly gate in memory (weaker: per process) |
| Q4 | 10d | `project_state.project_id` = the project folder name (`results.db` lives in `<projects>/<project_id>/`), `"memory"` for in-memory DBs. Triggers always read it from the row, so it is informational until a Postgres merge. | folder name | — |
| Q5 | 10e | `results` UPDATE trigger has `WHEN OLD.x IS NOT NEW.x OR …` so no-op writes don't create versions. | on | drop the WHEN clause |
| Q6 | 10e | `record.attachment` trigger fires only when a Zotero key (8 × `[A-Z0-9]`) appears, disappears or changes. Rewriting the failed-lookup ISO timestamp (the live check does this on every miss) is not tracked; it has no visible effect and would churn versions like `last_active` (D10). | on | drop the `GLOB` line |
| Q7 | 10g | `labeled/changes`: `sort_key` is `null` when `in_view` is false (the client never needs it then; computing it would need an extra lookup). | null | compute it via `get_record_states` |
| Q8 | 10g | `records?ids=` also returns `users`, because a patched row may carry a `user_id` the client has never seen. | added | drop it; client shows no name until reload |
| Q9 | 10g | `config_version` change → client refetches forms with `/labeled?per_page=1` (spec: implementer's choice). No new endpoint. | per_page=1 | add `GET /projects/<id>/collection_forms` |
| Q10 | 10c | "Chip counts are refetched whenever a poll returns ≥ 1 change": the Collection chips show no counts today (`fetchLabeledStats` is used only by `PriorCard.js`). The prompts still invalidate `["fetchLabeledStats", {project_id}]` on changes and own edits, so the default holds wherever counts are shown. | invalidate | — |
| Q11 | 10c | The labeler holds tag/list values in `useState` initialised once. For a patched row (own edit, or a colleague's edit in Part 4) to show without a remount, a sync effect re-initialises them when the props change, unless the card is dirty or a dialog is open. | added | without it, remote edits only show after a reload |
| Q12 | 10b | `test_projects.py::test_label_with_lists` asserts `record["lists_form"] is None` on a `/labeled` row; it moves to `labeled.json["forms"]["lists_form"]`. | updated | — |

### Task order

| Prompt | Part | What | Ships with |
|---|---|---|---|
| 10a | 1 | `getNextPageParam` → `undefined`, moved to `labeledPaging.js` + first Jest test; fix `test_list_item_cleaning` | alone |
| 10b | 2 | backend: slim rows, `forms`/`users` per response, helpers extracted from `api_get_labeled`, PUT routes return the row | **with 10c** (10b alone shows no tags in Collection cards) |
| 10c | 2 | frontend: forms/users from the response, stable key, own edits patch one row, no `fetchLabeledRecord` invalidations | with 10b; measure bytes for the commit message |
| 10d | 3 | `fork_migrations.py` ledger + revision `0001_change_tracking` (tables only) | alone |
| 10e | 3 | `_ensure_change_tracking_triggers()` + one test per write path | alone |
| 10f | 3 | `version` on rows, `watermark` (read first) + `config_version` on `/labeled` | alone |
| 10g | 4 | `labeled/changes`, `labeled/ids`, `records?ids=`, pruning; key equivalence test | alone |
| 10h | 4 | `collectionModel.js` (pure) + Jest | alone |
| 10i | 4 | `useCollection` hook, poll, banner, Refresh; delete `labeledPaging.js` | alone |

User checkpoints: after 10a, after 10b+10c (measure on the real project, amend the commit message), after 10f (Part 3 done), after 10i.

### File map

- Backend: `asreview/webapp/_api/projects.py` (helpers + 3 routes), `asreview/database/database.py` (ledger call, triggers, read/prune helpers), `asreview/database/fork_migrations.py` (new).
- Frontend (`asreview/webapp/src/`): `ProjectComponents/HistoryComponents/{LabeledRecord.js, labeledPaging.js (10a–10h only), collectionModel.js, useCollection.js}`, `ProjectComponents/ReviewComponents/{RecordCard.js, RecordCardLabeler.js}`, `api/ProjectAPI.js`.
- Tests: `tests/test_change_tracking.py` (new), `asreview/webapp/tests/test_api/{test_labeled_pagination.py, test_collection_sync.py (new), test_projects.py, test_llm_mapping.py}`, Jest files next to the modules.
- Untouched on purpose: `api_search_data` keeps per-row `tags_form`/`lists_form` (Review page and prior search use it; `RecordCard` falls back to the row fields), `ReviewPage.js`, `PriorSearch.js`, `llm_worker.py`, `PRAGMA user_version`, the `_ensure_*`/`_fix_*` functions.

### Design summary

**Backend helpers (10b)**, all in `projects.py`, shared by `/labeled` and the Part 4 routes so they can't drift:
`_parse_labeled_query(project) → q`, `_iter_labeled_matches(db, q, cursor, batch_size)` (generator of `(state, (time, rid))` in display order, includes the post-scan filters), `_users_map(project)`, `_build_rows(db, project, states)` (always NA → None, keeps `user_id`, adds `version` from 10f), `_collection_forms`. Part 4 adds `_sort_tuple`, `_parse_boundary`, `_scan_to_boundary`, `_sort_key_json`.

**Ledger (10d).** `REVISIONS = [("0001_change_tracking", fn)]`. Fast path: one SELECT on `fork_migrations`. Slow path: `commit()`, `BEGIN IMMEDIATE`, re-read, apply, insert the ledger rows, commit (rollback on error). Runs last in `create_tables()` and in `_is_valid()`'s read-write branch. A read-only open never migrates; `get_project_state()` then returns zeros.

**Triggers (10e)**, names `trg_ct_*`, created with `IF NOT EXISTS` on every read-write open (after the `_fix_*` functions). Per-record body (verified on SQLite 3.53; `INSERT OR REPLACE`/UPSERT inside a trigger fired by the outer UPSERT in `label_record` raised UNIQUE errors, so UPDATE + INSERT…WHERE NOT EXISTS is used):

```sql
UPDATE project_state SET version = version + 1;
UPDATE record_versions SET version = (SELECT version FROM project_state)
 WHERE project_id = (SELECT project_id FROM project_state) AND record_id = {rid};
INSERT INTO record_versions (project_id, record_id, version)
SELECT project_id, {rid}, version FROM project_state
 WHERE NOT EXISTS (SELECT 1 FROM record_versions
                    WHERE project_id = (SELECT project_id FROM project_state) AND record_id = {rid});
INSERT INTO change_log (project_id, version, record_id, source, changed_at)
SELECT project_id, version, {rid}, '{source}', (julianday('now') - 2440587.5) * 86400.0 FROM project_state;
```

Config tables: `UPDATE project_state SET version = version + 1, config_version = version + 1` (right-hand sides see the old row, so `config_version` = new version).

**`labeled/changes` (10g).** Prune (≤ 1/hour) → read `project_state` (watermark, D11) → `reset` if `since > watermark` or `since < pruned_version` → `get_changes_since(since, watermark)` (one entry per record, its max version; clamped to the watermark) → `reset` if > 200 → only if there are changes, scan the membership down to the boundary with the same helpers as `/labeled`. Empty polls cost two small SELECTs.

**`labeled/ids` (10g).** Same scan; `next_cursor` is null exactly when nothing matches beyond the boundary. If every loaded row left the view but more matches exist, the old boundary is returned so scrolling resumes there.

**Client model (10h).** Plain-object state keyed by `String(record_id)`; inputs never mutated. Watermark = lowest of all loaded pages (every loaded row is at least that new). `applyRowUpdates` never moves a card or changes its `sortKey` (the key at the shown position), so a sort-key change surfaces as staleness on the next poll. `stale` is sticky until `reconcile`. Dirty cards (unsaved labeler state or an open Tags/Lists/Note dialog) keep their position on Refresh, re-inserted after their nearest previous old neighbour.

**Hook (10i).** One async operation at a time (load more, poll, patch, Refresh); a generation counter drops results from an old snapshot (filter/subset change). Poll every 15 s, skipped while `document.hidden`, immediate on `visibilitychange`. `/records` fetched in chunks of 200.

### Risks and measurements

- `_save_tags` deletes then inserts, so one tag save writes several log rows (one version per row). Harmless; one entry per record reaches the client.
- Group-wide writes (`label_record`, `update_result`, `delete_result`) also version the duplicates. Harmless.
- `duplicate_of` and `included` changes are not tracked (not in the spec's list). Membership changes from deduplication show only after a reload.
- Bulk writes (`_replace_results_from_df`, import) → one log row per record → `reset`. 10e measures the trigger overhead on 10 000 rows.
- `in_view` cost with post-scan filters: 10g measures one call in the fixture; measure on the largest real project after 10i.
- Byte measurement for Part 2: DeepSeek measures on the test fixture; the user measures one real 50-row `/labeled` response (devtools) and amends the 10b/10c commit message.
- Known failures before this work: about 12 in `tests/test_database.py`, about 12 in `asreview/webapp/tests/test_api/test_projects.py`, and `test_list_item_cleaning` (fixed in 10a). Every prompt runs a baseline first; "no new failures" is the bar.
- Flask API tests need the network (Synergy dataset). See the spec's open risk.
