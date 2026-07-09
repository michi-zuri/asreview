import json
import os
import sqlite3
import time
from pathlib import Path


def uuid7():
    """Generate a UUID v7 (RFC 9562).

    Duplicated here so migrations are self-contained and don't depend on the
    current ``database.py`` implementation which may change over time.
    """
    timestamp_ms = int(time.time() * 1000)
    rand_bytes = os.urandom(12)

    uuid_bytes = bytearray(16)
    ts = timestamp_ms.to_bytes(6, "big")
    uuid_bytes[0:6] = ts
    uuid_bytes[6] = 0x70 | (rand_bytes[0] & 0x0F)
    uuid_bytes[7] = rand_bytes[1]
    uuid_bytes[8] = 0x80 | (rand_bytes[2] & 0x3F)
    uuid_bytes[9] = rand_bytes[3]
    uuid_bytes[10:16] = rand_bytes[4:10]

    h = uuid_bytes.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _migrate(project):
    """Migrate a valid project from version 3 to version 4.

    Changes:
    - ``tag_groups`` table is rewritten: ``group_id`` TEXT PK (UUID v7),
      ``description`` → ``input_helper_text``, ``tag_values`` JSON column
      removed.
    - New ``tag_options`` table stores individual tag option definitions
      (formerly the JSON ``tag_values`` array).
    - ``tags`` table is rewritten: ``tag_id`` TEXT PK (UUID v7),
      ``option_id`` FK replaces ``group_id`` + ``export_name`` +
      ``label_name``.
    - New ``list_containers`` table replaces ``lists.json``.
    - ``lists`` table column ``created`` → ``sorted_at``.
    - ``results.tags`` JSON column is dropped.
    - Database ``PRAGMA user_version`` is set to 4.
    - ``project.json`` ``project_file_version`` is set to 4.

    Parameters
    ----------
    project : Path
        Path to the root of the project (unzipped).
    """
    project = Path(project)

    # Update project.json version
    config_fp = Path(project, "project.json")
    with open(config_fp) as f:
        project_config = json.load(f)
    project_config["project_file_version"] = 4
    with open(config_fp, "w") as f:
        json.dump(project_config, f)

    results_db = Path(project, "results.db")
    conn = sqlite3.connect(str(results_db))
    try:
        cur = conn.cursor()

        # 1. Create new-schema tag_groups table (if old one exists, we'll
        #    migrate and then drop it)
        old_tag_groups_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_groups'"
        ).fetchone()

        cur.execute(
            """CREATE TABLE IF NOT EXISTS tag_groups_new (
                group_id TEXT PRIMARY KEY,
                export_name TEXT NOT NULL,
                label_name TEXT NOT NULL,
                required_for_relevant INTEGER NOT NULL DEFAULT 0,
                required_for_irrelevant INTEGER NOT NULL DEFAULT 0,
                all_required INTEGER NOT NULL DEFAULT 0,
                single INTEGER NOT NULL DEFAULT 0,
                input_helper_text TEXT DEFAULT '',
                sorted_at FLOAT NOT NULL DEFAULT 0
            )"""
        )

        # 2. Create tag_options table
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tag_options (
                option_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                export_name TEXT NOT NULL,
                label_name TEXT NOT NULL,
                free_text_enabled INTEGER NOT NULL DEFAULT 0,
                free_text_required INTEGER NOT NULL DEFAULT 0,
                sorted_at FLOAT NOT NULL DEFAULT 0,
                UNIQUE (option_id, group_id),
                FOREIGN KEY (group_id) REFERENCES tag_groups_new(group_id)
            )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_tag_options_group
            ON tag_options(group_id)"""
        )

        # 3. Create new-schema tags table
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tags_new (
                tag_id TEXT PRIMARY KEY,
                record_id INTEGER NOT NULL,
                option_id TEXT NOT NULL,
                checked INTEGER NOT NULL DEFAULT 0,
                additional_text TEXT,
                UNIQUE(record_id, option_id),
                FOREIGN KEY (record_id) REFERENCES record(record_id),
                FOREIGN KEY (option_id) REFERENCES tag_options(option_id)
            )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_tags_record
            ON tags_new(record_id)"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_tags_record_option
            ON tags_new(record_id, option_id)"""
        )

        # 4. Create list_containers table
        cur.execute(
            """CREATE TABLE IF NOT EXISTS list_containers (
                list_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                required_for_relevant INTEGER NOT NULL DEFAULT 0,
                description TEXT,
                sorted_at FLOAT NOT NULL DEFAULT 0
            )"""
        )

        # 5. Migrate tag group definitions
        old_to_new_group_id = {}  # old int id → new UUID v7 group_id
        tags_data = None

        # Source 1: tags.json (v3 projects)
        tags_json_path = Path(project, "tags.json")
        if tags_json_path.exists():
            with open(tags_json_path) as f:
                tags_data = json.load(f)

        # Source 2: project.json tags field (v2→v3 migration)
        if not tags_data and project_config.get("tags"):
            tags_data = project_config["tags"]

        if tags_data:
            for group in tags_data:
                old_id = group.get("id")
                new_group_id = uuid7()
                if old_id is not None:
                    old_to_new_group_id[old_id] = new_group_id

                cur.execute(
                    """INSERT INTO tag_groups_new
                       (group_id, export_name, label_name,
                        required_for_relevant, required_for_irrelevant,
                        all_required, single, input_helper_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_group_id,
                        group.get("export", ""),
                        group.get("label", ""),
                        1 if group.get("required_relevant") else 0,
                        1 if group.get("required_irrelevant") else 0,
                        1 if group.get("require_all") else 0,
                        1 if group.get("single_select") else 0,
                        group.get("input_helper_text", "") or "",
                    ),
                )

                # Migrate tag option values with sequential sort order
                values = group.get("values", [])
                for i, val in enumerate(values):
                    option_id = uuid7()
                    cur.execute(
                        """INSERT INTO tag_options
                           (option_id, group_id, export_name, label_name,
                            free_text_enabled, free_text_required, sorted_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            option_id,
                            new_group_id,
                            val.get("export", ""),
                            val.get("label", ""),
                            1 if val.get("free_text", False) else 0,
                            0,
                            float(i + 1),
                        ),
                    )

        # 6. Migrate lists from lists.json → list_containers
        lists_json_path = Path(project, "lists.json")
        if lists_json_path.exists():
            with open(lists_json_path) as f:
                lists_data = json.load(f)
            for lst in lists_data:
                list_id = lst.get("id")
                if not list_id:
                    continue
                cur.execute(
                    """INSERT INTO list_containers
                       (list_id, name, required_for_relevant, description)
                       VALUES (?, ?, ?, ?)""",
                    (
                        str(list_id),
                        lst.get("name", ""),
                        1 if lst.get("required_for_relevant") else 0,
                        lst.get("input_helper_text") or None,
                    ),
                )

        # 7. Rename ``created`` → ``sorted_at`` in lists table (if it exists)
        lists_table_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lists'"
        ).fetchone()
        if lists_table_exists:
            list_cols = [
                row[1] for row in cur.execute("PRAGMA table_info(lists)")
            ]
            if "created" in list_cols and "sorted_at" not in list_cols:
                # Rebuild lists table with the new column name
                cur.execute("""
                    CREATE TABLE lists_new
                    (record_id INTEGER NOT NULL,
                    list_id TEXT NOT NULL,
                    item_id TEXT NOT NULL PRIMARY KEY,
                    name TEXT NOT NULL,
                    sorted_at FLOAT NOT NULL,
                    UNIQUE (record_id, list_id, name))
                """)
                cur.execute(
                    """INSERT INTO lists_new
                       (record_id, list_id, item_id, name, sorted_at)
                       SELECT record_id, list_id, item_id, name, created
                       FROM lists"""
                )
                cur.execute("DROP TABLE lists")
                cur.execute("ALTER TABLE lists_new RENAME TO lists")

        # 8. Migrate per-record tag selections from results.tags
        columns = [row[1] for row in cur.execute("PRAGMA table_info(results)")]
        if "tags" in columns:
            # Build option_id mapping: (old_group_id, value_id) → new option_id
            # We need to reconstruct the old value id mapping from the tag
            # options we just inserted, keyed by export_name within each group.
            old_value_to_option = {}
            if tags_data:
                # Build (old_group_id, old_value_id) → option lookup
                option_rows = cur.execute(
                    """SELECT o.option_id, o.group_id, o.export_name
                       FROM tag_options o
                       JOIN tag_groups_new g ON g.group_id = o.group_id"""
                ).fetchall()
                option_by_group_export = {}
                for opt_id, gid, export_name in option_rows:
                    option_by_group_export[(gid, export_name)] = opt_id

                for group in tags_data:
                    old_group_id = group.get("id")
                    new_group_id = old_to_new_group_id.get(old_group_id)
                    if new_group_id is None:
                        continue
                    for val in group.get("values", []):
                        old_val_id = val.get("id")
                        export_name = val.get("export", "")
                        new_opt_id = option_by_group_export.get(
                            (new_group_id, export_name)
                        )
                        if old_val_id is not None and new_opt_id is not None:
                            old_value_to_option[
                                (old_group_id, old_val_id)
                            ] = new_opt_id

            # Now migrate the tags
            rows = cur.execute(
                "SELECT record_id, tags FROM results WHERE tags IS NOT NULL"
            ).fetchall()

            for record_id, tags_json in rows:
                if not tags_json or tags_json == "[]":
                    continue
                try:
                    parsed = json.loads(tags_json)
                except (json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(parsed, list):
                    continue

                for group in parsed:
                    if not isinstance(group, dict):
                        continue
                    old_group_id = group.get("id")
                    for tag_val in group.get("values", []):
                        if not isinstance(tag_val, dict):
                            continue
                        old_val_id = tag_val.get("id")
                        option_id = old_value_to_option.get(
                            (old_group_id, old_val_id)
                        )
                        if option_id is None:
                            continue
                        checked = bool(tag_val.get("checked", False))
                        text = tag_val.get("text")

                        cur.execute(
                            """INSERT OR REPLACE INTO tags_new
                               (tag_id, record_id, option_id, checked,
                                additional_text)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                uuid7(),
                                record_id,
                                option_id,
                                1 if checked else 0,
                                text if checked and text else None,
                            ),
                        )

            # Drop the old tags column
            try:
                cur.execute("ALTER TABLE results DROP COLUMN tags")
            except sqlite3.OperationalError:
                _rebuild_results_without_tags(cur, conn)

        # 9. Swap old tables for new ones
        if old_tag_groups_exists:
            cur.execute("DROP TABLE tag_groups")
        cur.execute("ALTER TABLE tag_groups_new RENAME TO tag_groups")

        # Check if old tags table exists and swap
        old_tags_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tags'"
        ).fetchone()
        if old_tags_exists:
            tag_cols = [
                row[1] for row in cur.execute("PRAGMA table_info(tags)")
            ]
            # Only swap if the old table doesn't already have the v4 schema
            if "option_id" not in tag_cols:
                cur.execute("DROP TABLE tags")
                cur.execute("ALTER TABLE tags_new RENAME TO tags")
            else:
                # tags_new might not have been created if there was nothing
                # to migrate — just drop it
                cur.execute("DROP TABLE tags_new")
        else:
            cur.execute("ALTER TABLE tags_new RENAME TO tags")

        conn.commit()

        # 10. Update database version
        cur.execute("PRAGMA user_version = 4")
        conn.commit()

        # 11. Re-create decision_changes triggers
        _ensure_triggers(cur)
        conn.commit()

        # 12. Remove legacy files
        if tags_json_path.exists():
            tags_json_path.unlink()
        if lists_json_path.exists():
            lists_json_path.unlink()

    finally:
        conn.close()


def _rebuild_results_without_tags(cur, conn):
    """Rebuild the results table without the tags column.

    Used as a fallback when ALTER TABLE DROP COLUMN is not supported.
    """
    columns = [
        row[1]
        for row in cur.execute("PRAGMA table_info(results)")
        if row[1] != "tags"
    ]
    col_list = ", ".join(columns)

    cur.execute(
        f"""CREATE TABLE results_new (
            {col_list}
        )"""
    )
    cur.execute(
        f"INSERT INTO results_new SELECT {col_list} FROM results"
    )
    cur.execute("DROP TABLE results")
    cur.execute("ALTER TABLE results_new RENAME TO results")
    conn.commit()


def _ensure_triggers(cur):
    """Ensure the decision_changes triggers exist."""
    cur.execute("DROP TRIGGER IF EXISTS trg_results_delete")
    cur.execute("DROP TRIGGER IF EXISTS trg_results_label_update")
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_results_delete
        AFTER DELETE ON results
        FOR EACH ROW
        BEGIN
            INSERT INTO decision_changes (record_id, label, time, user_id)
            VALUES (OLD.record_id, OLD.label, OLD.time, OLD.user_id);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_results_label_update
        AFTER UPDATE OF label ON results
        FOR EACH ROW
        WHEN OLD.label IS NOT NEW.label
        BEGIN
            INSERT INTO decision_changes (record_id, label, time, user_id)
            VALUES (OLD.record_id, OLD.label, OLD.time, OLD.user_id);
        END
    """)


def _validate(project):
    """Validate a migrated v4 project.

    Parameters
    ----------
    project : Path
        Path to the migrated project folder.

    Raises
    ------
    ValueError
        If the migrated project is not valid.
    """
    project = Path(project)
    config_fp = Path(project, "project.json")
    if not config_fp.exists():
        raise ValueError("Migrated project is missing project.json.")

    with open(config_fp) as f:
        config = json.load(f)

    if config.get("project_file_version") != 4:
        raise ValueError(
            f"Expected project file version 4, "
            f"got {config.get('project_file_version')}."
        )

    results_db_fp = Path(project, "results.db")
    if not results_db_fp.exists():
        raise ValueError("Migrated project is missing results.db.")

    from asreview.database.database import Database

    with Database(results_db_fp) as db:
        db._is_valid(expected_version=4)
