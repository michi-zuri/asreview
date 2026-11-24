# Spec 0002: One central database (DRAFT)

Status: **DRAFT, 2026-11-24.** Written by Claude from the conversation of 2026-11-24, to be refined with the user in a spec drilling session. Nothing here is agreed beyond the items under "User decisions". Phase 0 is already written as task `llm_prompts/10g3.txt`. Amp adds implementation notes **below** the line at the end and does not rewrite this spec. Any objection goes back to the user, not into a silent change.

| Phase | What | Depends on |
|---|---|---|
| 0 | One project id per database file (task 10g3) | Spec 0001 Part 3 |
| 1 | One project registry; identity checked on every open | 0 |
| 2 | `project.json` (and the side files) move into the database | 1 |
| 3 | `results.db` moves into the central database (Postgres) | 2 |

---

## Goal

- All project state lives in **one central database**, in exactly one place each. In the end that database is Postgres. There are no per-project SQLite files and no JSON files holding live state.
- A project's identity (its id) is stored once. Everything else refers to it. A mismatch is refused loudly and never "works" silently.
- Standalone project files remain only as an **export/import format**.

## Non-goals

- Building the Postgres backend now (that is Phase 3, specified later).
- Keeping the unauthenticated mode working (see U1).
- Keeping `project.json` as a live store "for compatibility". It remains only as an export/import format and as input to the legacy migrations.

## User decisions (2026-11-24)

U1. **The unauthenticated mode is dropped entirely.** So the central database (today `asreview.<env>.sqlite`, SQLAlchemy, created only in the auth branch of `app.py`) is always present and can hold the project registry.

U2. **Simulation may keep producing standalone export files compatible with upstream ASReview.**

U3. **Keep the per-row `project_id` columns** in `project_state`, `record_versions` and `change_log` (Spec 0001 D8). Phase 3 merges the per-project databases into one, and needs them.

U4. **The project id mismatch is a major problem**, to be fixed with guardrails, not left as "informational".

U5. **Simulation output is not a concern for now.** It is inherited from upstream ASReview and is left as it is until the review part works for us (see O1).

U6. **API keys in exports are a high-priority problem, but they are not fixed now.** The plan is to encrypt the keys securely, or better, not to store an API key at all (see O2, O4).

U7. **Phase 0 seeds `project_state` from the folder name.** This is accepted as it is: folder names equal project ids in practice, and Phase 1 adds the cross-check.

## Where things live today (inventory, 2026-11-24)

From the code and a read-only scan of the 15 real projects in `~/.asreview`.

| Item | Today | Target home (proposal) |
|---|---|---|
| Project id | folder name; `project.json` `id`; `project_state.project_id` in `results.db`; auth DB `projects.project_id` (auth mode only) | Registry row in the central DB (Phase 1). `project_state` copies it and is checked against it. |
| Name, mode, created_at, version fields | `project.json` | Registry / `projects` table (Phase 2a) |
| Review settings (model, feature extractor, …) | `project.json` `review` | Typed settings table (Phase 2b) |
| LLM settings, including the **API key** | `project.json` `llm` | Settings table; key encrypted (Phase 2c) |
| Zotero settings, including the **API key** | `zotero.json` | Settings table; key encrypted (Phase 2c) |
| Highlights | `highlights.json` | Table (Phase 2d) |
| Training progress | `training_progress.json` (written by `_tasks.py`) | Task-manager / central DB (Phase 2d) |
| Tags, lists | DB tables (`tag_groups`, `tag_options`, `list_containers`, `lists`, `tags`). `tags.json` / `lists.json` remain only as legacy migration input. | Unchanged; they move with `results.db` (Phase 3) |
| Labels, notes, rankings, LLM results, change tracking | `results.db` | Central DB (Phase 3) |
| Dataset file | `data/` | Blob storage (open question O5) |
| Feature matrices | `feature_matrices/` | Blob storage or recompute (O5) |
| PDF cache | `llm_pdf_cache/` | Blob storage (O5) |
| Legacy material | `legacy_v2/`, `legacy_v3/`, `recovered.db`, `query.sql` | Kept in the folder until a migration consumes it; not live state |
| Task queue | `queue.sqlite` (task manager) | Central DB or unchanged (O6) |

About 75 places read `project.config[...]` directly. They are the main cost of Phase 2.

In all 15 real projects the folder name equals the `project.json` id. Only one project (99fafe13…) has been opened since change tracking landed, and it is consistent (one `project_state` row, no foreign rows).

## Phases

### Phase 0: one id per database file (task 10g3)

Inside `results.db`, `project_state` is the only authority for the id. Reads take the id from it, never from the folder name. Guard triggers enforce one row and keep every tracked row on its id. Revision `0004_project_id_consistency` refuses an inconsistent legacy file. See `llm_prompts/10g3.txt`. Point 2 of the 10g review was a silent mismatch: a copied or renamed folder made every poll empty. Phase 0 fixes it.

Phase 0 does NOT compare the file's id with the folder, `project.json` or the auth DB. That is Phase 1. One consequence was checked on copies of the real projects (2026-11-24). A project that has never been opened since change tracking landed (14 of 15) gets its `project_state` id from the **folder name at first open**. If that folder was renamed, the wrong id is stored from then on. Phase 1 should seed from the registry / `project.json` id instead of the folder name, and check it.

### Phase 1: one registry; identity checked on open

- The central DB has a project registry: one row per project, keyed by the project id (a uuid). Folder name = registry id.
- Opening a project checks, in one place: folder name == registry id == `project.json` `id` (while it exists) == `project_state.project_id`. **Any mismatch refuses to open**, with an error that names every value. There is no silent "fix".
- One explicit operation changes a project's id: import as a copy. It updates the registry, the folder, `project.json` and `project_state` (one `UPDATE`; the 10g3 cascade re-keys the rows). `rekey_project_id` becomes part of it.
- Projects on disk with no registry row (created in the dropped unauth mode) need a one-time adoption step (O3).

### Phase 2: `project.json` and the side files move into the database

- All access goes through a typed `ProjectRepository` (or similar). Nothing reads `project.config[...]` or the side files directly any more.
- Order: (a) identity and metadata, (b) review settings, (c) LLM and Zotero settings with encrypted keys, (d) highlights and training progress.
- In each step the repository first reads the JSON (or file) and writes the DB, then reads the DB only. A step is done when its file is no longer read or written. The legacy migrations still read old files.
- At the end `project.json` is written only by export and read only by import and the legacy migrations.

### Phase 3: `results.db` moves into the central database

- Postgres. Every row carries `project_id` (D8 / U3). Primary keys gain `project_id` where they don't have it yet (`results`, `record`, `tags`, `lists`, `last_ranking`, `llm_*`, …).
- Change tracking becomes Postgres triggers or application code. The per-project `project_state` row becomes a row in a shared `project_state` table keyed by `project_id`. The Spec 0001 semantics (watermark, `record_versions`, D11) stay unchanged.
- The `_ensure_*` / `_fix_*` schema code and the fork ledger are replaced by one migration tool for the central DB (e.g. Alembic; not a dependency today, only Flask-SQLAlchemy and sqlalchemy-utils are).
- Per-project SQLite files remain only inside export files.

## Export and import

- The export stays a standalone file. It is the only place where `project.json` + `results.db` (+ data) live after Phase 2/3. Import reads it into the central DB.
- **Today's exports leak secrets.** `Project.export` copies the whole folder (ignoring only `tmp` and `*.lock`), so the LLM API key in `project.json` and the Zotero API key in `zotero.json` end up in every `.asreview` file. High priority, but deferred by the user (U6). The plan is to encrypt the keys securely, or better, not to store an API key at all (O2, O4).

## Open questions (for the drilling session)

O1. *(Deferred, U5.)* **Simulation output is NOT upstream-compatible today.** `asreview/simulation/cli.py` calls `Project.create(…, project_mode="simulate")`, which writes `project_file_version = ASMR_PROJECT_VERSION` (-1, the asmr mark), and its `results.db` carries the fork schema and mark. Upstream deliberately refuses asmr projects. So U2 ("compatible with upstream") needs work: a separate upstream-format writer for simulate, or an export option. Which upstream version is the target? Does simulate need anything fork-specific (tags, LLM)?

O2. **Webapp exports.** Are they fork-only (as today), or should there also be an upstream-compatible export? Either way, secrets must be stripped. Should an import ask for the keys again, or are keys per user/server and never per project?

O3. **Adopting existing projects.** In Phase 1, what happens to folders without a registry row (from the unauth mode or other machines)? Options: adopt them automatically (registry row from `project.json`, after the identity check), adopt them through an explicit CLI command, or refuse them.

O4. **Key encryption.** Where does the encryption key live (env var, file next to the central DB, OS keyring)? Or should keys move out of projects altogether (per user or per server)?

O5. **Files and blobs.** Datasets, feature matrices, the PDF cache: Postgres large objects/bytea, a blob directory keyed by project id, or S3-compatible storage? Can feature matrices just be recomputed?

O6. **Task queue** (`queue.sqlite`): move it into the central DB or keep it separate?

O7. **Phase 3 concurrency.** SQLite WAL with a 5 s busy timeout versus Postgres row locks. Which of today's `BEGIN IMMEDIATE` sections need `SELECT … FOR UPDATE`, and which need advisory locks?

O8. **Upstream sync.** Every phase moves the fork further from upstream's storage model. Is that accepted (no upstream-compatibility requirement for the live store, only for the export in U2)?

## Risks

- Phase 2 touches about 75 call sites. Without the repository layer first, it ends up half in JSON and half in the DB, which is the chaos U4 wants to avoid.
- During Phases 1–2 there are still several copies of the id. The Phase 1 open check is what keeps them honest, so it must run on every open path (webapp, CLI, simulate, import).

---
<!-- Amp: implementation notes below this line. Do not edit above. -->
