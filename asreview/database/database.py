import json
import os
import sqlite3
import time
from functools import cached_property

import pandas as pd

from asreview.data.record import Record
from asreview.database.store import DataStore
from asreview.database.store import _build_conn_uri


def uuid7():
    """Generate a UUID v7 (RFC 9562).

    48-bit Unix ms timestamp + version (4 bits) + variant (2 bits) +
    74 random bits. Returns the standard 36-character hex string with hyphens.

    Python 3.14 adds ``uuid.uuid7()``; this backport can be replaced once
    that is available.
    """
    timestamp_ms = int(time.time() * 1000)
    rand_bytes = os.urandom(12)

    # timestamp (48 bits) as 6 bytes, big-endian
    ts = timestamp_ms.to_bytes(6, "big")

    # version nibble: 7
    ts = bytes([ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]])

    # Build UUID bytes: 6-byte timestamp + 10-byte random
    uuid_bytes = bytearray(16)

    # bytes 0-5: timestamp (first 48 bits)
    uuid_bytes[0:6] = ts

    # bytes 6-7: version (4 bits → 0x7xxx) + random
    uuid_bytes[6] = 0x70 | (rand_bytes[0] & 0x0F)
    uuid_bytes[7] = rand_bytes[1]

    # bytes 8-9: variant (2 bits → 10xx) + random
    uuid_bytes[8] = 0x80 | (rand_bytes[2] & 0x3F)
    uuid_bytes[9] = rand_bytes[3]

    # bytes 10-15: remaining random
    uuid_bytes[10:16] = rand_bytes[4:10]

    # Format: 00000000-0000-7000-8000-000000000000
    h = uuid_bytes.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

__all__ = ["Database"]

CURRENT_DATABASE_VERSION = 4

MODEL_COLUMNS = [
    "classifier",
    "querier",
    "balancer",
    "feature_extractor",
    "training_set",
]

REQUIRED_TABLES = [
    "results",
    "last_ranking",
    "decision_changes",
]

RESULTS_TABLE_COLUMNS_PANDAS_DTYPES = {
    "record_id": "Int64",
    "label": "Int64",
    "classifier": "object",
    "querier": "object",
    "balancer": "object",
    "feature_extractor": "object",
    "training_set": "Int64",
    "time": "Float64",
    "note": "object",
    "user_id": "Int64",
}

RANKING_TABLE_COLUMNS_PANDAS_DTYPES = {
    "record_id": "Int64",
    "ranking": "Int64",
    "classifier": "object",
    "querier": "object",
    "balancer": "object",
    "feature_extractor": "object",
    "training_set": "Int64",
    "time": "Float64",
}


def open_db(fp, read_only=False):
    """Open a database.

    Parameters
    ----------
    fp : path-like
        File path to the database
    read_only : bool, optional
        Whether to create a new database if one doesn't exist yet and whether the opened
        database will be in read only mode or not.

    Returns
    -------
    Database
        ASReview database.

    Raises
    ------
    FileNotFoundError
        If `read_only` and there is no file at `fp`.
    ValueError
        If `read_only` and there is no valid database at `fp`.
    """
    if not fp.is_file():
        if read_only:
            raise FileNotFoundError(
                f"File path {fp} is not a file and 'read_only' is 'True'"
            )
        fp.parent.mkdir(parents=True, exist_ok=True)

    db = Database(fp, read_only=read_only)
    try:
        db._is_valid()
    except ValueError as e:
        if read_only:
            raise ValueError(
                f"There is no valid database at {fp} and the database is opened in"
                " read-only mode"
            ) from e
        db.create_tables()
    return db


class Database:
    """Database containing the input data and results.

    Database contains two parts: the input and the results. For more information on the
    input, see `asreview.database.store.py`. For more information on the results, see
    `asreview.database.sqlstate.py`.

    Attributes
    ----------
    user_version: str
        Return the version number of the database.
    """

    def __init__(self, fp=":memory:", record_cls=Record, read_only=False):
        """Initialize the Database.

        Parameters
        ----------
        fp : str | Path
            Path of the database file. Use `":memory:"` for an in-memory database.
        record_cls : type[asreview.data.record.Base], optional
            Type to use for the input records, see `DataStore` for more information.
        read_only : bool, optional
            Whether to open the database in read only mode. If the database is opened in
            read only mode and an attempt to write to the database is made, an
            `sqlite3.OperationalError` will be raised.
        """
        if fp == ":memory:" and read_only:
            raise ValueError("Can't open an in-memory database in read only mode")

        self.fp = fp
        self.record_cls = record_cls
        self.read_only = read_only
        self._in_memory = fp == ":memory:"
        self._closed = False
        self._conn_uri = _build_conn_uri(fp, read_only)

        self.input = DataStore(
            conn_uri=self._conn_uri, record_cls=record_cls, read_only=read_only
        )

        if self._in_memory:
            # Eagerly open the sqlite3 connection. For named in-memory databases,
            # the database is destroyed when the last connection to it closes.
            # This connection acts as an anchor that keeps the database alive
            # for the lifetime of this object.
            self._conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @cached_property
    def _conn(self):
        """Get a connection to the SQLite database.

        Returns
        -------
        sqlite3.Connection
            Connection to the SQLite database.
        """
        conn = sqlite3.connect(self._conn_uri, uri=True)
        if not self.read_only:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def close(self):
        """Close the database and release all resources.

        For in-memory databases this will destroy the database. Safe to call multiple
        times.
        """
        if self._closed:
            return
        self._closed = True
        self.input.engine.dispose()
        if "_conn" in self.__dict__:
            self._conn.close()
            del self.__dict__["_conn"]

    @property
    def user_version(self):
        """Version number of the state."""
        cur = self._conn.cursor()
        version = cur.execute("PRAGMA user_version")

        return int(version.fetchone()[0])

    @user_version.setter
    def user_version(self, version):
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA user_version = {version}")
        self._conn.commit()
        cur.close()

    def create_tables(self):
        self.user_version = CURRENT_DATABASE_VERSION
        self.input.create_tables()

        cur = self._conn.cursor()

        cur.execute(
            """CREATE TABLE results
                            (record_id INTEGER PRIMARY KEY,
                            label INTEGER,
                            classifier TEXT,
                            querier TEXT,
                            balancer TEXT,
                            feature_extractor TEXT,
                            training_set INTEGER,
                            time FLOAT,
                            note TEXT,
                            user_id INTEGER,
                            assigned_at REAL,
                            last_active REAL)"""
        )

        cur.execute(
            """CREATE TABLE last_ranking
                            (record_id INTEGER PRIMARY KEY,
                            ranking INT,
                            classifier TEXT,
                            querier TEXT,
                            balancer TEXT,
                            feature_extractor TEXT,
                            training_set INTEGER,
                            time FLOAT)"""
        )

        cur.execute(
            """CREATE TABLE decision_changes
                            (record_id INTEGER PRIMARY KEY,
                            label INTEGER,
                            time FLOAT,
                            user_id INTEGER)"""
        )

        self._conn.commit()

        self._set_results_changes_triggers()
        self._ensure_results_indexes()
        self._ensure_lists_table()
        self._ensure_list_containers_table()
        self._ensure_tag_groups_table()
        self._ensure_tag_options_table()
        self._ensure_tags_table()
        self._ensure_llm_dispatch_table()
        self._ensure_llm_results_table()

    def _ensure_lists_table(self):
        """Create the ``lists`` table that stores per-record list items.

        A record can have several user-defined lists (configured in
        ``list_containers``), and each list can contain multiple free-text
        items. Because of this clear many-to-one relationship the items live
        in their own table instead of a JSON column on ``results``. Both
        ``list_id`` and ``item_id`` are ``uuid7`` strings; ``sorted_at`` is a
        unix timestamp used to order the items within a list. ``item_id`` is
        the primary key so the items can later be referenced by foreign keys.

        Idempotent (``CREATE TABLE IF NOT EXISTS``), so it is safe to call on
        every read-write open. The table is only ever created, never migrated:
        the schema is applied when the table does not exist yet and skipped
        otherwise.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS lists (
                record_id INTEGER NOT NULL,
                list_id TEXT NOT NULL,
                item_id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                sorted_at FLOAT NOT NULL,
                UNIQUE (record_id, list_id, name),
                FOREIGN KEY (record_id) REFERENCES record(record_id),
                FOREIGN KEY (list_id) REFERENCES list_containers(list_id)
            )"""
        )
        self._conn.commit()

    def _ensure_list_containers_table(self):
        """Create the ``list_containers`` table that stores list definitions.

        Replaces the ``lists.json`` file. Each row defines one list with its
        display name, behaviour flags, and optional description. Lists are
        ordered by ``sorted_at`` (oldest first).

        Idempotent (``CREATE TABLE IF NOT EXISTS``), so it is safe to call on
        every read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS list_containers (
                list_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                required_for_relevant INTEGER NOT NULL DEFAULT 0,
                description TEXT,
                sorted_at FLOAT NOT NULL DEFAULT 0
            )"""
        )
        self._conn.commit()

    def _ensure_tag_groups_table(self):
        """Create the ``tag_groups`` table that stores tag group definitions.

        Replaces the ``tags.json`` file. Each row defines one tag group with
        its behaviour flags. Tag options are stored in the ``tag_options``
        table. Groups are ordered by ``sorted_at`` (oldest first).

        Idempotent (``CREATE TABLE IF NOT EXISTS``), so it is safe to call on
        every read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tag_groups (
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
        self._conn.commit()

    def _ensure_tag_options_table(self):
        """Create the ``tag_options`` table that stores tag option definitions.

        Each row is one selectable tag value within a tag group. Replaces the
        JSON ``tag_values`` column that was previously stored on
        ``tag_groups``.  Options are ordered by ``sorted_at`` (oldest first).

        Idempotent (``CREATE TABLE IF NOT EXISTS``), so it is safe to call on
        every read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tag_options (
                option_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                export_name TEXT NOT NULL,
                label_name TEXT NOT NULL,
                free_text_enabled INTEGER NOT NULL DEFAULT 0,
                free_text_required INTEGER NOT NULL DEFAULT 0,
                sorted_at FLOAT NOT NULL DEFAULT 0,
                UNIQUE (export_name, group_id),
                FOREIGN KEY (group_id) REFERENCES tag_groups(group_id)
            )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_tag_options_group
            ON tag_options(group_id)"""
        )
        self._conn.commit()

    def _ensure_tags_table(self):
        """Create the ``tags`` table that stores per-record tag selections.

        Each row represents one tag value applied (or not) to one record.
        Rows are keyed by the unique combination of record and option,
        allowing efficient upserts when a label is changed.

        Idempotent (``CREATE TABLE IF NOT EXISTS``), so it is safe to call on
        every read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tags (
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
            ON tags(record_id)"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_tags_option
            ON tags(option_id)"""
        )
        self._conn.commit()

    def _ensure_llm_dispatch_table(self):
        """Create the ``llm_dispatch`` table (LLM screening queue).

        One row per record that has been (or should be) sent to the LLM.
        ``record_id`` is the primary key: a record has at most one dispatch
        row, updated in place when it is re-queued. ``dispatched_at`` is a
        unix timestamp (float, UTC instant) that defines the canonical
        serving order. ``status`` is one of ``queued``, ``in_flight``,
        ``ready``, ``failed``, ``missing_pdf``.

        Idempotent (``CREATE TABLE IF NOT EXISTS``); safe on every
        read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS llm_dispatch (
                record_id INTEGER PRIMARY KEY,
                dispatched_at REAL,
                status TEXT NOT NULL,
                prompt_hash TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                FOREIGN KEY (record_id) REFERENCES record(record_id)
            )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_llm_dispatch_status_time
               ON llm_dispatch(status, dispatched_at)"""
        )
        self._conn.commit()

    def _ensure_llm_results_table(self):
        """Create the ``llm_results`` table (cached LLM screening output).

        Primary key ``(record_id, prompt_hash)`` so a prompt change
        naturally invalidates the cache without deleting old rows.
        ``payload_json`` is the raw LLM JSON; ``created_at`` is a unix
        timestamp (float, UTC instant).

        Idempotent (``CREATE TABLE IF NOT EXISTS``); safe on every
        read-write open.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS llm_results (
                record_id INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                model TEXT,
                payload_json TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                created_at REAL,
                PRIMARY KEY (record_id, prompt_hash),
                FOREIGN KEY (record_id) REFERENCES record(record_id)
            )"""
        )
        self._conn.commit()

    def _ensure_results_indexes(self):
        """Create indexes that speed up collection (labeled history) loading.

        Idempotent (uses ``CREATE INDEX IF NOT EXISTS``), so it is safe to call
        on every read-write open and acts as a lightweight migration for
        existing projects. The index matches the ordering used by the
        keyset-paginated :meth:`get_results_page` query
        (``(time IS NULL), time DESC, record_id DESC``) restricted to labeled
        records, allowing SQLite to serve a page without scanning and sorting
        the whole table.
        """
        cur = self._conn.cursor()
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_results_collection_desc
            ON results ((time IS NULL), time DESC, record_id DESC)
            WHERE label IS NOT NULL"""
        )
        self._conn.commit()

    def _is_valid(self, expected_version=None):
        if expected_version is None:
            expected_version = CURRENT_DATABASE_VERSION
        if self.user_version != expected_version:
            raise ValueError(
                f"Database version {self.user_version} is not supported. "
                "See migration guide."
            )
        cur = self._conn.cursor()
        column_names = cur.execute("PRAGMA table_info(results)").fetchall()
        table_names = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        table_names = [tup[0] for tup in table_names]
        missing_tables = [
            table
            for table in REQUIRED_TABLES + [self.record_table_name]
            if table not in table_names
        ]
        if missing_tables:
            raise ValueError(
                f"The SQL file should contain tables named "
                f"'{' '.join(missing_tables)}'."
            )

        column_names = [tup[1] for tup in column_names]
        missing_columns = [
            col
            for col in RESULTS_TABLE_COLUMNS_PANDAS_DTYPES.keys()
            if col not in column_names
        ]
        if missing_columns:
            raise ValueError(
                f"The results table does not contain the columns "
                f"{' '.join(missing_columns)}."
            )

        if not self.read_only:
            self._fix_decision_changes_schema(cur)
            self._fix_record_schema(cur)
            self._fix_results_schema(cur)
            self._fix_tag_options_schema(cur)
            self._fix_list_containers_schema(cur)
            self._fix_tag_groups_schema(cur)
            self._fix_tags_schema(cur)
            self._ensure_results_indexes()
            self._ensure_lists_table()
            self._ensure_list_containers_table()
            self._ensure_tag_groups_table()
            self._ensure_tag_options_table()
            self._ensure_tags_table()
            self._ensure_llm_dispatch_table()
            self._ensure_llm_results_table()

    def _fix_record_schema(self, cur):
        """Add columns introduced after the initial schema to the record table."""
        columns = [
            row[1]
            for row in cur.execute(f"PRAGMA table_info({self.record_table_name})")
        ]

        if "original_id" not in columns:
            cur.execute(
                f"ALTER TABLE {self.record_table_name} ADD COLUMN original_id TEXT"
            )
            self._conn.commit()

        if "attachment" not in columns:
            cur.execute(
                f"ALTER TABLE {self.record_table_name} ADD COLUMN attachment TEXT"
            )
            self._conn.commit()

    def _fix_results_schema(self, cur):
        """Add columns introduced after the initial results schema."""
        columns = [
            row[1] for row in cur.execute("PRAGMA table_info(results)")
        ]
        if "assigned_at" not in columns:
            cur.execute("ALTER TABLE results ADD COLUMN assigned_at REAL")
            self._conn.commit()
        if "last_active" not in columns:
            cur.execute("ALTER TABLE results ADD COLUMN last_active REAL")
            self._conn.commit()

    def _fix_tag_options_schema(self, cur):
        """Add ``sorted_at`` column to ``tag_options`` if it is missing.

        Projects created or migrated before this column was introduced will
        have a ``tag_options`` table without ``sorted_at``.  The column is
        added with a default of 0 so existing options keep their UUID v7
        creation order as a fallback.
        """
        exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_options'"
        ).fetchone()
        if not exists:
            return
        columns = [row[1] for row in cur.execute("PRAGMA table_info(tag_options)")]
        if "sorted_at" not in columns:
            cur.execute(
                "ALTER TABLE tag_options ADD COLUMN sorted_at FLOAT NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def _fix_list_containers_schema(self, cur):
        """Add ``sorted_at`` column to ``list_containers`` if it is missing."""
        exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='list_containers'"
        ).fetchone()
        if not exists:
            return
        columns = [row[1] for row in cur.execute("PRAGMA table_info(list_containers)")]
        if "sorted_at" not in columns:
            cur.execute(
                "ALTER TABLE list_containers ADD COLUMN sorted_at FLOAT NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def _fix_tag_groups_schema(self, cur):
        """Add ``sorted_at`` column to ``tag_groups`` if it is missing."""
        exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_groups'"
        ).fetchone()
        if not exists:
            return
        columns = [row[1] for row in cur.execute("PRAGMA table_info(tag_groups)")]
        if "sorted_at" not in columns:
            cur.execute(
                "ALTER TABLE tag_groups ADD COLUMN sorted_at FLOAT NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def _fix_tags_schema(self, cur):
        """Ensure the ``tags`` table has the v4 schema (``option_id`` column).

        Some projects migrated from v3 may have an old-format ``tags`` table
        that lacks ``option_id`` because the v3→v4 migration step 9 skipped
        the swap when the old table had no ``group_id`` column.  When this is
        detected the old table is dropped and recreated with the correct
        schema (per-record tag selections were originally stored in
        ``results.tags`` JSON so the migration step 8 should already have
        extracted them into the new table).
        """
        exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tags'"
        ).fetchone()
        if not exists:
            return
        columns = [row[1] for row in cur.execute("PRAGMA table_info(tags)")]
        if "option_id" not in columns:
            cur.execute("DROP TABLE tags")
            self._conn.commit()

    def _fix_decision_changes_schema(self, cur):
        """Fix decision_changes schema for projects migrated from old v2 format.

        Old v2 projects had (record_id, new_label, time) instead of
        (record_id, label, time, user_id). Projects already migrated to v3
        may still carry the old schema.
        """
        columns = [row[1] for row in cur.execute("PRAGMA table_info(decision_changes)")]

        if "new_label" in columns and "label" not in columns:
            cur.execute("ALTER TABLE decision_changes RENAME COLUMN new_label TO label")

        if "user_id" not in columns:
            cur.execute("ALTER TABLE decision_changes ADD COLUMN user_id INTEGER")

        self._conn.commit()

    def _set_results_changes_triggers(self):
        con = self._conn
        cur = con.cursor()
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
        con.commit()

    @property
    def record_table_name(self):
        return self.input.record_cls.__tablename__

    @property
    def exist_new_labeled_records(self):
        """Return True if there are new labeled records.

        Return True if there are any record labels added since the last time
        the model ranking was added to the state. Also returns True if no
        model was trained yet, but priors have been added.
        """
        labeled = self.get_results_table("label")
        last_training_set = self.get_last_ranking_table()["training_set"]

        if last_training_set.empty or pd.isna(last_training_set.max()):
            return len(labeled) > 0
        else:
            return len(labeled) > last_training_set.max()

    def _replace_results_from_df(self, results):
        # Drop the tags column if present -- tags are now stored in the
        # separate ``tags`` table, not as a results table column.
        results = results.drop(columns=["tags"], errors="ignore")

        if not set(results.columns) == set(RESULTS_TABLE_COLUMNS_PANDAS_DTYPES):
            raise ValueError(
                f"Columns of the results dataframe should be "
                f"{list(RESULTS_TABLE_COLUMNS_PANDAS_DTYPES.keys())}."
            )

        cur = self._conn.cursor()
        cur.execute("delete from results")
        self._conn.commit()
        cur.close()

        results.to_sql("results", self._conn, if_exists="append", index=False)

    def _replace_last_ranking_from_df(self, last_ranking):
        if not set(last_ranking.columns) == set(RANKING_TABLE_COLUMNS_PANDAS_DTYPES):
            raise ValueError(
                f"Columns of the last ranking dataframe should be "
                f"{list(RANKING_TABLE_COLUMNS_PANDAS_DTYPES.keys())}."
            )

        last_ranking.to_sql(
            "last_ranking", self._conn, if_exists="replace", index=False
        )

    def add_last_ranking(
        self,
        ranked_record_ids,
        classifier,
        querier,
        balancer,
        feature_extractor,
        training_set=None,
    ):
        """Save the ranking of the last iteration of the model.

        Save the ranking of the last iteration of the model, in the ranking
        order, so the record on row 0 is ranked first by the model.

        Parameters
        ----------
        ranked_record_ids: list, numpy.ndarray
            A list of records ids in the order that they were ranked.
        classifier: str
            Name of the classifier of the model.
        querier: str
            Name of the query strategy of the model.
        balancer: str
            Name of the balance strategy of the model.
        feature_extractor: str
            Name of the feature extraction method of the model.
        training_set: int
            Number of labeled records available at the time of training.
        """

        pd.DataFrame(
            {
                "record_id": ranked_record_ids,
                "ranking": range(len(ranked_record_ids)),
                "classifier": classifier,
                "querier": querier,
                "balancer": balancer,
                "feature_extractor": feature_extractor,
                "training_set": training_set,
                "time": time.time(),
            }
        ).to_sql("last_ranking", self._conn, if_exists="replace", index=False)

    def get_last_ranking_table(self):
        """Get the ranking from the state.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns 'record_id', 'ranking', 'classifier',
            'querier', 'balancer', 'feature_extractor',
            'training_set' and 'time'. It has one row for each record in the
            dataset, and is ordered by ranking.
        """
        return pd.read_sql_query(
            "SELECT * FROM last_ranking",
            self._conn,
            dtype=RANKING_TABLE_COLUMNS_PANDAS_DTYPES,
        )

    def label_record(self, record_id, label, tags=None, user_id=None):
        labeling_time = time.time()
        con = self._conn
        cur = con.cursor()
        model_string = ", ".join(MODEL_COLUMNS)
        target_result_string = ", ".join(
            f"target_result.{col}" for col in MODEL_COLUMNS
        )
        upsert_columns = ["label", "time", "user_id"] + MODEL_COLUMNS
        upsert_string = ", ".join(f"{col} = excluded.{col}" for col in upsert_columns)

        cur.execute(
            f"""
            WITH target_group AS (
                SELECT record_id
                FROM {self.record_table_name}
                WHERE group_id = (
                    SELECT group_id
                    FROM {self.record_table_name}
                    WHERE record_id=:record_id
                )
            ), target_result AS (
                SELECT {model_string}
                FROM results
                WHERE record_id = :record_id
            )
            INSERT INTO results(record_id, label, time, user_id, {model_string})
            SELECT target_group.record_id, :label, :time, :user_id, {target_result_string}
            FROM target_group
            LEFT JOIN target_result ON 1
            ON CONFLICT(record_id) DO UPDATE
                SET {upsert_string};
            """,
            {
                "record_id": record_id,
                "label": label,
                "time": labeling_time,
                "user_id": user_id,
            },
        )
        con.commit()

        if tags is not None:
            self._save_tags(record_id, tags)

    def query_top_ranked(self, user_id=None):
        model_string = ", ".join(MODEL_COLUMNS)
        top_record_string = ", ".join(f"top_record.{col}" for col in MODEL_COLUMNS)
        upsert_columns = ["user_id"] + MODEL_COLUMNS
        upsert_string = ", ".join(f"{col} = excluded.{col}" for col in upsert_columns)

        con = self._conn
        cur = con.cursor()
        result = cur.execute(
            f"""WITH top_record AS (
                SELECT last_ranking.*
                FROM last_ranking
                LEFT JOIN results USING (record_id)
                WHERE results.record_id IS NULL OR (results.label IS NULL AND results.user_id IS NULL)
                ORDER BY ranking
                LIMIT 1
            ), group_records AS (
                SELECT record.record_id
                FROM record
                WHERE group_id = (
                    SELECT group_id
                    FROM record
                    WHERE record.record_id = (SELECT record_id FROM top_record)
                )
            )
            INSERT INTO results (record_id, user_id, {model_string})
            SELECT group_records.record_id, :user_id, {top_record_string}
            FROM group_records
            CROSS JOIN top_record ON TRUE

            ON CONFLICT(record_id) DO UPDATE
                SET {upsert_string}
            RETURNING record_id
            ;""",
            {"user_id": user_id},
        ).fetchone()
        con.commit()
        # Check if any record was updated, if not, the query did not return a top ranked record
        # and we should not return the newly pending record.
        # cur.rowcount does not work here, because UPSERTS always return -1 (unknown)
        if result is None:
            raise ValueError("Failed to query top ranked record")
        return self.get_pending(user_id=user_id)

    def top_up_dispatch(self, buffer_size, prompt_hash):
        """Refill the LLM dispatch queue up to ``buffer_size``.

        Inserts the next needed records into ``llm_dispatch`` from the pool,
        in ranking order, skipping records that are prior-irrelevant
        (``record.included == 0``) and records already in the queue. A
        candidate that already has a cached ``llm_results`` row for
        ``prompt_hash`` is inserted with status ``ready`` (cache hit);
        otherwise status ``queued``. Inserts rows only; performs no LLM work.

        Parameters
        ----------
        buffer_size : int
            Desired number of not-yet-checked-out dispatch rows.
        prompt_hash : str
            Hash of the current assembled system prompt.

        Returns
        -------
        int
            Number of rows inserted.
        """
        con = self._conn
        cur = con.cursor()

        # Active buffer = dispatch rows not yet checked out (checkout creates
        # a results row for the record).
        active = cur.execute(
            """SELECT COUNT(*) FROM llm_dispatch
               WHERE status IN ('queued', 'in_flight', 'ready')
                 AND record_id NOT IN (
                     SELECT record_id FROM results WHERE label IS NOT NULL
                 )"""
        ).fetchone()[0]

        deficit = buffer_size - active
        if deficit <= 0:
            return 0

        candidates = cur.execute(
            f"""SELECT lr.record_id
                FROM last_ranking lr
                JOIN {self.record_table_name} rec ON rec.record_id = lr.record_id
                LEFT JOIN results r ON r.record_id = lr.record_id
                WHERE (r.record_id IS NULL OR r.label IS NULL)
                  AND (rec.included IS NULL OR rec.included != 0)
                  AND lr.record_id IN (
                      SELECT group_id FROM {self.record_table_name}
                  )
                  AND lr.record_id NOT IN (SELECT record_id FROM llm_dispatch)
                ORDER BY lr.ranking
                LIMIT ?""",
            (deficit,),
        ).fetchall()

        base = time.time()
        inserted = 0
        for i, (record_id,) in enumerate(candidates):
            has_result = cur.execute(
                "SELECT 1 FROM llm_results "
                "WHERE record_id = ? AND prompt_hash = ? LIMIT 1",
                (record_id, prompt_hash),
            ).fetchone()
            status = "ready" if has_result else "queued"
            cur.execute(
                """INSERT INTO llm_dispatch
                   (record_id, dispatched_at, status, prompt_hash,
                    attempts, last_error)
                   VALUES (?, ?, ?, ?, 0, NULL)""",
                (record_id, base + i * 1e-6, status, prompt_hash),
            )
            inserted += 1

        con.commit()
        return inserted

    def requeue_for_prompt_change(self, new_prompt_hash):
        """Re-queue all not-yet-labeled dispatched records under a new prompt.

        Targets records that have an llm_dispatch row whose prompt_hash differs
        from new_prompt_hash AND whose results row (if any) is not yet labeled
        (label IS NULL) or has no results row. Already-labeled records are
        skipped. Rewrites their llm_dispatch rows to status 'queued' with the
        new prompt_hash and a fresh, contiguous, monotonically increasing block
        of dispatched_at values that PRESERVES the previous relative order.
        Resets attempts=0 and last_error=NULL. Returns the number of records
        re-queued.
        """
        con = self._conn
        cur = con.cursor()
        rows = cur.execute(
            "SELECT d.record_id FROM llm_dispatch d "
            "LEFT JOIN results r ON r.record_id = d.record_id "
            "WHERE d.prompt_hash != ? "
            "AND (r.record_id IS NULL OR r.label IS NULL) "
            "ORDER BY d.dispatched_at ASC",
            (new_prompt_hash,),
        ).fetchall()
        base = time.time()
        for i, (record_id,) in enumerate(rows):
            cur.execute(
                "UPDATE llm_dispatch SET status='queued', prompt_hash=?, "
                "dispatched_at=?, attempts=0, last_error=NULL "
                "WHERE record_id=?",
                (new_prompt_hash, base + i * 1e-6, record_id),
            )
        con.commit()
        return len(rows)

    def requeue_record(self, record_id, prompt_hash):
        """Re-queue a single record under a new prompt hash.

        Resets status to 'queued', updates prompt_hash, zeroes attempts and
        clears last_error. The dispatched_at timestamp is preserved so the
        record keeps its relative ordering.
        """
        con = self._conn
        cur = con.cursor()
        cur.execute(
            "UPDATE llm_dispatch SET status='queued', prompt_hash=?, "
            "attempts=0, last_error=NULL WHERE record_id=?",
            (prompt_hash, record_id),
        )
        con.commit()

    def force_requeue(self, record_id, prompt_hash):
        """Force one record back into llm_dispatch (bypass cache-skip).

        Upserts an llm_dispatch row for record_id: status 'queued',
        prompt_hash, fresh dispatched_at, attempts 0, last_error NULL.
        Returns True.
        """
        now = time.time()
        con = self._conn
        cur = con.cursor()
        cur.execute(
            """INSERT INTO llm_dispatch
               (record_id, dispatched_at, status, prompt_hash,
                attempts, last_error)
               VALUES (?, ?, 'queued', ?, 0, NULL)
               ON CONFLICT(record_id) DO UPDATE SET
                   status = 'queued',
                   prompt_hash = excluded.prompt_hash,
                   dispatched_at = excluded.dispatched_at,
                   attempts = 0,
                   last_error = NULL""",
            (record_id, now, prompt_hash),
        )
        con.commit()
        return True

    def claim_next_queued_dispatch(self):
        """Atomically claim the oldest queued dispatch row.

        Sets its status to 'in_flight' and returns its record_id, or None if
        no queued rows exist. Concurrency-safe: SQLite serializes writers, so
        two workers claim different rows.
        """
        con = self._conn
        cur = con.cursor()
        row = cur.execute(
            """UPDATE llm_dispatch
               SET status = 'in_flight'
               WHERE record_id = (
                   SELECT record_id FROM llm_dispatch
                   WHERE status = 'queued'
                   ORDER BY dispatched_at ASC
                   LIMIT 1
               )
               RETURNING record_id"""
        ).fetchone()
        con.commit()
        return row[0] if row else None

    def store_llm_result(self, record_id, prompt_hash, model, payload_json,
                         input_tokens=None, output_tokens=None):
        """Upsert an llm_results row and mark its dispatch row 'ready'.

        Keyed by (record_id, prompt_hash); re-processing overwrites. Also
        clears last_error on the dispatch row.
        """
        now = time.time()
        con = self._conn
        cur = con.cursor()
        cur.execute(
            """INSERT INTO llm_results
               (record_id, prompt_hash, model, payload_json,
                input_tokens, output_tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(record_id, prompt_hash) DO UPDATE SET
                   model = excluded.model,
                   payload_json = excluded.payload_json,
                   input_tokens = excluded.input_tokens,
                   output_tokens = excluded.output_tokens,
                   created_at = excluded.created_at""",
            (record_id, prompt_hash, model, payload_json,
             input_tokens, output_tokens, now),
        )
        cur.execute(
            "UPDATE llm_dispatch SET status = 'ready', last_error = NULL "
            "WHERE record_id = ?",
            (record_id,),
        )
        con.commit()

    def mark_dispatch_failed(self, record_id, error):
        """Set a dispatch row to 'failed' and record last_error."""
        con = self._conn
        cur = con.cursor()
        cur.execute(
            "UPDATE llm_dispatch SET status = 'failed', last_error = ? "
            "WHERE record_id = ?",
            (str(error) if error is not None else None, record_id),
        )
        con.commit()

    def mark_dispatch_missing_pdf(self, record_id):
        """Set a dispatch row to 'missing_pdf'."""
        con = self._conn
        cur = con.cursor()
        cur.execute(
            "UPDATE llm_dispatch SET status = 'missing_pdf' WHERE record_id = ?",
            (record_id,),
        )
        con.commit()

    def delete_llm_results(self, record_id):
        """Delete all cached LLM results for a record.

        Called when the PDF the results were derived from is deleted, so that
        stale pre-fill data does not linger in the UI.
        """
        con = self._conn
        cur = con.cursor()
        cur.execute(
            "DELETE FROM llm_results WHERE record_id = ?",
            (record_id,),
        )
        con.commit()

    def increment_dispatch_attempts(self, record_id):
        """Increment attempts on a dispatch row; return the new count."""
        con = self._conn
        cur = con.cursor()
        row = cur.execute(
            "UPDATE llm_dispatch SET attempts = attempts + 1 "
            "WHERE record_id = ? RETURNING attempts",
            (record_id,),
        ).fetchone()
        con.commit()
        return row[0] if row else None

    def get_llm_meta(self, record_id, prompt_hash):
        """Return combined dispatch+result metadata for a record, or None.

        Keys: status, dispatched_at, attempts, last_error (from llm_dispatch);
        has_result (bool, whether an llm_results row exists for prompt_hash)
        and, when present, model, input_tokens, output_tokens, created_at.
        """
        cur = self._conn.cursor()
        d = cur.execute(
            "SELECT status, dispatched_at, attempts, last_error "
            "FROM llm_dispatch WHERE record_id = ?", (record_id,)
        ).fetchone()
        if d is None:
            return None
        meta = {"status": d[0], "dispatched_at": d[1], "attempts": d[2],
                "last_error": d[3], "has_result": False}
        r = cur.execute(
            "SELECT model, input_tokens, output_tokens, created_at "
            "FROM llm_results WHERE record_id = ? AND prompt_hash = ?",
            (record_id, prompt_hash),
        ).fetchone()
        if r is not None:
            meta.update(has_result=True, model=r[0], input_tokens=r[1],
                        output_tokens=r[2], created_at=r[3])
        return meta

    def get_llm_payload(self, record_id, prompt_hash):
        """Return stored payload_json for (record_id, prompt_hash) or None."""
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT payload_json FROM llm_results "
            "WHERE record_id = ? AND prompt_hash = ?",
            (record_id, prompt_hash),
        ).fetchone()
        return row[0] if row else None

    def get_result_status(self, record_id):
        """Return {'user_id', 'label'} for a record's results row, or None."""
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT user_id, label FROM results WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        return None if row is None else {"user_id": row[0], "label": row[1]}

    def checkout_oldest_dispatched(self, user_id, stale_timeout=None):
        """Check out the oldest ready (or queued) dispatched record.

        Selects a dispatch row with status in (ready, queued, in_flight) whose
        corresponding results row either does not exist or is still unlabeled
        (label IS NULL).  "ready" records are served first so that LLM-screened
        records reach the user ahead of unscreened ones.  Within a status tier
        the oldest dispatched_at wins.

        When *stale_timeout* is given, records that are currently assigned to a
        different user and whose last_active is fresher than the timeout are
        skipped — this prevents the cascade from immediately bouncing a record
        back that was just reassigned.

        Creates or reassigns ``results`` rows for the whole group to
        ``user_id``.

        Returns
        -------
        pandas.DataFrame
            The user's pending row(s) (same shape as get_pending). Empty when
            nothing was available to check out.
        """
        now = time.time()
        model_string = ", ".join(MODEL_COLUMNS)
        top_cols = ", ".join(f"top_record.{c}" for c in MODEL_COLUMNS)

        # Build the staleness guard: when stale_timeout is given, skip records
        # that were recently assigned to another user so we don't bounce the
        # same record straight back.
        if stale_timeout is not None:
            cutoff = now - stale_timeout
            candidate_filter = """
                AND (
                    results.record_id IS NULL
                    OR (results.label IS NULL AND results.user_id = :user_id)
                    OR (results.label IS NULL AND results.last_active < :cutoff)
                )
            """
            params = {"user_id": user_id, "now": now, "cutoff": cutoff}
        else:
            candidate_filter = """
                AND (results.record_id IS NULL OR results.label IS NULL)
            """
            params = {"user_id": user_id, "now": now}

        con = self._conn
        cur = con.cursor()
        result = cur.execute(
            f"""
            WITH top_record AS (
                SELECT last_ranking.*
                FROM llm_dispatch
                JOIN last_ranking USING (record_id)
                LEFT JOIN results ON results.record_id = llm_dispatch.record_id
                WHERE llm_dispatch.status IN ('queued', 'in_flight', 'ready')
                  {candidate_filter}
                ORDER BY CASE WHEN llm_dispatch.status = 'ready' THEN 0 ELSE 1 END,
                         llm_dispatch.dispatched_at ASC
                LIMIT 1
            ),
            group_records AS (
                SELECT record.record_id
                FROM {self.record_table_name} AS record
                WHERE group_id = (
                    SELECT group_id
                    FROM {self.record_table_name}
                    WHERE record_id = (SELECT record_id FROM top_record)
                )
            )
            INSERT INTO results
                (record_id, user_id, assigned_at, last_active, {model_string})
            SELECT group_records.record_id, :user_id, :now, :now, {top_cols}
            FROM group_records
            CROSS JOIN top_record ON TRUE
            ON CONFLICT(record_id) DO UPDATE SET
                user_id = excluded.user_id,
                assigned_at = excluded.assigned_at,
                last_active = excluded.last_active
            RETURNING record_id
            """,
            params,
        ).fetchone()
        con.commit()

        if result is None:
            return self.get_pending(user_id=user_id).iloc[0:0]
        return self.get_pending(user_id=user_id)

    def reassign_stale(self, user_id, older_than):
        """Reassign the oldest stale checkout to ``user_id``.

        A checkout is stale when it is pending (label IS NULL), owned by a
        DIFFERENT user, and its ``last_active`` is older than ``older_than``
        seconds. The oldest such checkout (smallest last_active) has its whole
        group transferred to ``user_id`` with assigned_at and last_active
        reset to now.

        Parameters
        ----------
        user_id : int
            The requesting user who should receive the reassigned checkout.
        older_than : float
            Staleness threshold in seconds.

        Returns
        -------
        pandas.DataFrame
            The user's pending row(s) (same shape as get_pending). Empty when
            nothing stale was available.
        """
        now = time.time()
        cutoff = now - older_than

        con = self._conn
        cur = con.cursor()
        result = cur.execute(
            f"""
            WITH stale AS (
                SELECT record_id
                FROM results
                WHERE label IS NULL
                  AND user_id IS NOT NULL
                  AND user_id != :user_id
                  AND last_active IS NOT NULL
                  AND last_active < :cutoff
                ORDER BY last_active ASC
                LIMIT 1
            ),
            group_records AS (
                SELECT record.record_id
                FROM {self.record_table_name} AS record
                WHERE group_id = (
                    SELECT group_id
                    FROM {self.record_table_name}
                    WHERE record_id = (SELECT record_id FROM stale)
                )
            )
            UPDATE results
            SET user_id = :user_id, assigned_at = :now, last_active = :now
            WHERE record_id IN (SELECT record_id FROM group_records)
            RETURNING record_id
            """,
            {"user_id": user_id, "cutoff": cutoff, "now": now},
        ).fetchone()
        con.commit()

        if result is None:
            return self.get_pending(user_id=user_id).iloc[0:0]
        return self.get_pending(user_id=user_id)

    def touch_last_active(self, record_id, user_id):
        """Bump last_active for a user's pending checkout of a record.

        Returns True if the record is still checked out to this user (a
        pending results row with label IS NULL and matching user_id existed
        and was updated); otherwise False.
        """
        now = time.time()
        con = self._conn
        cur = con.cursor()
        row = cur.execute(
            "UPDATE results SET last_active = ? "
            "WHERE record_id = ? AND user_id = ? AND label IS NULL "
            "RETURNING record_id",
            (now, record_id, user_id),
        ).fetchone()
        con.commit()
        return row is not None

    def update_result(self, record_id, label=None, tags=None, user_id=None):
        if label is None and tags is None:
            raise ValueError("At least one of 'label' or 'tags' must be provided.")

        fields = []
        values = {"record_id": record_id}
        if label is not None:
            fields.append("label = :label")
            values["label"] = label
            if user_id is not None:
                # We only update the user_id if the label changes.
                fields.append("user_id = :user_id")
                values["user_id"] = user_id

        if fields:
            set_string = ", ".join(fields)
            con = self._conn
            cur = con.cursor()
            cur.execute(
                f"""
                WITH target_group AS (
                    SELECT record_id
                    FROM {self.record_table_name}
                    WHERE group_id = (
                        SELECT group_id
                        FROM {self.record_table_name}
                        WHERE record_id=:record_id
                    )
                )
                UPDATE results
                SET {set_string}
                WHERE record_id IN (SELECT record_id FROM target_group)
                """,
                values,
            )
            con.commit()

        if tags is not None:
            self._save_tags(record_id, tags)

    def update_note(self, record_id, note=None):
        """Change the note of an already labeled or pending record.

        Parameters
        ----------
        record_id: int
            Id of the record whose label should be changed.
        note: str
            Note to add to the record.
        """

        cur = self._conn.cursor()
        cur.execute(
            f"""
            WITH target_group AS (
                SELECT record_id
                FROM {self.record_table_name}
                WHERE group_id = (
                    SELECT group_id
                    FROM {self.record_table_name}
                    WHERE record_id=:record_id
                )
            )
            UPDATE results SET note = :note WHERE record_id IN (
                SELECT record_id FROM target_group
            )""",
            {"note": note, "record_id": record_id},
        )

        if cur.rowcount == 0:
            raise ValueError(f"Record with id {record_id} not found.")

        self._conn.commit()

    def get_lists(self, record_id):
        """Get the list items stored for a record, ordered within each list.

        Returns
        -------
        list[dict]
            One dict per item with keys ``list_id``, ``item_id``, ``name`` and
            ``sorted_at``, ordered by ``sorted_at`` (oldest first). Empty list
            when the record has no items (or the ``lists`` table does not exist
            yet in a read-only legacy project).
        """
        cur = self._conn.cursor()
        try:
            rows = cur.execute(
                """SELECT list_id, item_id, name, sorted_at
                FROM lists WHERE record_id = ? ORDER BY list_id, sorted_at, rowid""",
                (record_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Legacy project opened read-only before the lists table existed.
            return []
        return [
            {
                "list_id": row[0],
                "item_id": row[1],
                "name": row[2],
                "sorted_at": row[3],
            }
            for row in rows
        ]

    def get_lists_for_records(self, record_ids):
        """Get list items for many records at once.

        Returns
        -------
        dict[int, list[dict]]
            Mapping of ``record_id`` to its list items (see :meth:`get_lists`),
            ordered by ``sorted_at`` within each list. Records without items are
            absent from the mapping.
        """
        record_ids = [int(r) for r in record_ids]
        if not record_ids:
            return {}
        cur = self._conn.cursor()
        placeholders = ",".join("?" * len(record_ids))
        try:
            rows = cur.execute(
                f"""SELECT record_id, list_id, item_id, name, sorted_at
                FROM lists WHERE record_id IN ({placeholders})
                ORDER BY record_id, list_id, sorted_at, rowid""",
                record_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        result = {}
        for row in rows:
            result.setdefault(int(row[0]), []).append(
                {
                    "list_id": row[1],
                    "item_id": row[2],
                    "name": row[3],
                    "sorted_at": row[4],
                }
            )
        return result

    def replace_lists(self, record_id, items):
        """Replace all list items for a record.

        Each item belongs to exactly one record. Items are keyed by
        ``item_id`` (the table's primary key) so they can later be referenced by
        foreign keys.

        Parameters
        ----------
        record_id : int
            Record whose list items should be replaced.
        items : list[dict]
            Items with keys ``list_id``, ``item_id``, ``name`` and optionally
            ``sorted_at``. An empty list clears the record's items.
        """
        self._ensure_lists_table()
        con = self._conn
        cur = con.cursor()

        cur.execute("DELETE FROM lists WHERE record_id = ?", (int(record_id),))

        rows = []
        for item in items or []:
            name = item["name"]
            sorted_at = item.get("sorted_at")
            if sorted_at is None:
                sorted_at = time.time()
            rows.append(
                (
                    int(record_id),
                    str(item["list_id"]),
                    str(item["item_id"]),
                    name,
                    float(sorted_at),
                )
            )
        if rows:
            cur.executemany(
                """INSERT INTO lists
                (record_id, list_id, item_id, name, sorted_at)
                VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
        con.commit()

    def delete_list_container(self, list_id):
        """Delete a list container definition.

        A list container can only be deleted when no records have items
        referencing it.  If any ``lists`` rows still reference the container
        the method raises ``ValueError`` with a descriptive message.

        Parameters
        ----------
        list_id : str
            The list container ID to delete.
        """
        self._ensure_list_containers_table()
        self._ensure_lists_table()
        cur = self._conn.cursor()
        refs = cur.execute(
            "SELECT COUNT(*) FROM lists WHERE list_id = ?", (list_id,)
        ).fetchone()[0]
        if refs > 0:
            raise ValueError(
                "This list cannot be deleted because it still has items "
                "assigned to records. Remove all items from this list first."
            )
        cur.execute(
            "DELETE FROM list_containers WHERE list_id = ?", (list_id,)
        )
        if cur.rowcount == 0:
            raise ValueError(f"List with id '{list_id}' not found.")
        self._conn.commit()

    def delete_tag_option(self, option_id, group_id):
        """Delete a single tag option from a group.

        The option can only be deleted when no tag selection rows in the
        ``tags`` table reference it (enforced by the FK constraint).  If any
        record still references the option the method raises ``ValueError``.

        Parameters
        ----------
        option_id : str
            The option to delete.
        group_id : str
            The group the option belongs to.
        """
        self._ensure_tag_options_table()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM tag_options WHERE option_id = ? AND group_id = ?",
                (option_id, group_id),
            )
        except sqlite3.IntegrityError:
            raise ValueError(
                "This tag cannot be deleted because one or more records "
                "have been labeled with this tag. Remove the tag from all "
                "records first."
            )
        if cur.rowcount == 0:
            raise ValueError(f"Tag option '{option_id}' not found.")
        self._conn.commit()

    def delete_tag_group(self, group_id):
        """Delete a tag group and all its options.

        A tag group can only be deleted when none of its options are
        referenced by the ``tags`` table — the FK from ``tag_options`` to
        ``tag_groups`` blocks the delete if any options still exist, and the
        FK from ``tags`` to ``tag_options`` prevents deleting options that
        are in use.  Both conditions must be satisfied before calling this
        method.

        Parameters
        ----------
        group_id : str
            The tag group ID to delete.
        """
        self._ensure_tag_groups_table()
        self._ensure_tag_options_table()
        cur = self._conn.cursor()
        opt_count = cur.execute(
            "SELECT COUNT(*) FROM tag_options WHERE group_id = ?", (group_id,)
        ).fetchone()[0]
        if opt_count > 0:
            raise ValueError(
                "This tag group cannot be deleted because it still contains "
                "options. Delete all options first."
            )
        try:
            cur.execute(
                "DELETE FROM tag_groups WHERE group_id = ?", (group_id,)
            )
        except sqlite3.IntegrityError:
            raise ValueError(
                "This tag group cannot be deleted because it is still in use."
            )
        if cur.rowcount == 0:
            raise ValueError(f"Tag group with id '{group_id}' not found.")
        self._conn.commit()

    def _save_tags(self, record_id, tags, cur=None):
        """Store tag selections for a record in the ``tags`` table.

        Deletes existing tag rows for the given record within the affected
        groups, then inserts one row per checked tag value. Only checked tags
        are persisted; unchecked tags are omitted (their absence means
        unchecked).

        The ``tags`` parameter uses the same nested format as the API:

            [{"id": group_id, "values": [{"id": option_id, "checked": bool,
                                          "text": str}, ...]}, ...]

        Each value ``id`` maps directly to a ``tag_options.option_id``.
        """
        if not tags:
            return

        own_cur = cur is None
        if own_cur:
            cur = self._conn.cursor()

        # Collect all option_ids to delete existing rows in one pass
        option_ids = []
        rows = []
        for group in tags:
            if not isinstance(group, dict):
                continue
            for tag_val in group.get("values", []):
                if not isinstance(tag_val, dict):
                    continue
                option_id = tag_val.get("id")
                if option_id is None:
                    continue
                option_ids.append(str(option_id))
                checked = bool(tag_val.get("checked", False))
                text = tag_val.get("text")
                rows.append(
                    (
                        uuid7(),
                        int(record_id),
                        str(option_id),
                        1 if checked else 0,
                        text if checked and text else None,
                    )
                )

        if option_ids:
            placeholders = ",".join("?" * len(option_ids))
            cur.execute(
                f"DELETE FROM tags WHERE record_id = ? AND option_id IN ({placeholders})",
                [int(record_id)] + option_ids,
            )

        if rows:
            cur.executemany(
                """INSERT OR REPLACE INTO tags
                   (tag_id, record_id, option_id, checked, additional_text)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

        if own_cur:
            self._conn.commit()
            cur.close()

    def _load_tags(self, record_ids):
        """Load per-record tag selections and reconstruct the nested format.

        Returns a ``dict[int, list[dict]]`` mapping ``record_id`` to a list of
        tag group objects in the same format the API expects:

            [{"id": group_id, "export": ..., "label": ...,
              "values": [{"id": option_id, "export": ..., "label": ...,
                          "checked": bool, "text": str | None}, ...]}, ...]

        Records without any tag rows are absent from the result.
        """
        result = {}
        if not record_ids:
            return result

        record_ids = [int(r) for r in record_ids]
        cur = self._conn.cursor()

        try:
            # Phase 1: build template from tag_groups + tag_options
            option_rows = cur.execute(
                """SELECT g.group_id, g.export_name, g.label_name,
                          g.single, g.required_for_relevant,
                          g.required_for_irrelevant, g.all_required,
                          g.input_helper_text,
                          o.option_id, o.export_name, o.label_name,
                          o.free_text_enabled, o.free_text_required
                   FROM tag_groups g
                   JOIN tag_options o ON o.group_id = g.group_id
                   ORDER BY g.sorted_at, g.group_id, o.sorted_at, o.option_id"""
            ).fetchall()
        except sqlite3.OperationalError:
            # tables do not exist (read-only legacy project)
            return result

        # Build template: group_id → group_dict with all values unchecked
        groups_by_id = {}
        for row in option_rows:
            gid = row[0]
            if gid not in groups_by_id:
                groups_by_id[gid] = {
                    "id": gid,
                    "export": row[1],
                    "label": row[2],
                    "single_select": bool(row[3]),
                    "required_relevant": bool(row[4]),
                    "required_irrelevant": bool(row[5]),
                    "require_all": bool(row[6]),
                    "input_helper_text": row[7] or "",
                    "values": [],
                }
            groups_by_id[gid]["values"].append(
                {
                    "id": row[8],
                    "export": row[9],
                    "label": row[10],
                    "free_text": bool(row[11]),
                    "free_text_required": bool(row[12]),
                    "checked": False,
                    "text": None,
                }
            )

        if not groups_by_id:
            return result

        # Phase 2: load checked tags for requested records
        placeholders = ",".join("?" * len(record_ids))
        try:
            checked_rows = cur.execute(
                f"""SELECT t.record_id, t.option_id, t.checked,
                           t.additional_text
                    FROM tags t
                    WHERE t.record_id IN ({placeholders})""",
                record_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            checked_rows = []

        # Phase 3: apply checkmarks
        # Build option_id → group_id map for fast lookup
        option_to_group = {}
        for gid, gdata in groups_by_id.items():
            for v in gdata["values"]:
                option_to_group[v["id"]] = gid

        record_checkmarks = {}
        for row in checked_rows:
            rid = int(row[0])
            option_id = row[1]
            if rid not in record_checkmarks:
                record_checkmarks[rid] = {}
            record_checkmarks[rid][option_id] = {
                "checked": bool(row[2]),
                "text": row[3],
            }

        # Build per-record results
        for rid, checkmarks in record_checkmarks.items():
            # Deep-copy the template groups that have checkmarks
            seen_groups = set()
            for option_id, chk in checkmarks.items():
                gid = option_to_group.get(option_id)
                if gid is None:
                    continue
                if rid not in result:
                    result[rid] = []
                if gid not in seen_groups:
                    # Copy the group template
                    template = groups_by_id[gid]
                    group_copy = {
                        k: v for k, v in template.items() if k != "values"
                    }
                    group_copy["values"] = [dict(v) for v in template["values"]]
                    result[rid].append(group_copy)
                    seen_groups.add(gid)

                # Apply checkmark to the matching value
                for gdata in result[rid]:
                    if gdata["id"] == gid:
                        for v in gdata["values"]:
                            if v["id"] == option_id:
                                v["checked"] = chk["checked"]
                                v["text"] = chk["text"]
                                break
                        break

        return result

    def delete_result(self, record_id):
        con = self._conn
        cur = con.cursor()
        cur.execute(
            f"""
            WITH target_group AS (
                SELECT record_id
                FROM {self.record_table_name}
                WHERE group_id = (
                    SELECT group_id
                    FROM {self.record_table_name}
                    WHERE record_id=:record_id
                )
            )
            DELETE FROM results
            WHERE record_id IN (SELECT record_id FROM target_group)
            """,
            {"record_id": record_id},
        )
        con.commit()

    def get_results_record(self, record_id):
        """Get the data of a specific query from the results table.

        Parameters
        ----------
        record_id: int
            Record id of which you want the data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the data from the results table with the given
            record_id and columns.
        """

        result = pd.read_sql_query(
            f"SELECT * FROM results WHERE record_id={record_id}",
            self._conn,
            dtype=RESULTS_TABLE_COLUMNS_PANDAS_DTYPES,
        )
        if not result.empty:
            tags_by_record = self._load_tags(result["record_id"].tolist())
            result["tags"] = result["record_id"].map(
                lambda rid: tags_by_record.get(int(rid), [])
            )
        else:
            result["tags"] = pd.Series(dtype="object")
        return result

    def get_results_table(self, columns=None, priors=True, pending=False, groups=False):
        """Get a subset from the results table.

        Can be used to get any column subset from the results table.
        Most other get functions use this one, except some that use a direct
        SQL query for efficiency.

        Parameters
        ----------
        columns: list, str
            List of columns names of the results table, or a string containing
            one column name.
        priors: bool
            Whether to keep the records containing the prior knowledge.
        pending: bool
            Whether to keep the records which are pending a labeling decision.
        groups: bool
            Return all the records of a group of records. Be default only returns the
            base record of each group.

        Returns
        -------
        pd.DataFrame:
            Dataframe containing the data of the specified columns of the
            results table.
        """
        if isinstance(columns, str):
            columns = [columns]

        # Remove "tags" from SQL columns list since it is no longer a results
        # table column -- it is reconstructed from the ``tags`` table below.
        want_tags = columns is None or "tags" in columns
        sql_columns = (
            None
            if columns is None
            else [c for c in columns if c != "tags"]
        )

        if (not priors) or (not pending) or (not groups):
            sql_where = []
            if not priors:
                sql_where.append("querier is not NULL")
            if not pending:
                sql_where.append("label is not NULL")
            if not groups:
                sql_where.append(
                    f"record_id IN ( SELECT group_id FROM {self.record_table_name})"
                )
            sql_where_str = "WHERE " + " AND ".join(sql_where)
        else:
            sql_where_str = ""

        if sql_columns is None:
            col_dtype = RESULTS_TABLE_COLUMNS_PANDAS_DTYPES
        else:
            col_dtype = {
                k: v
                for k, v in RESULTS_TABLE_COLUMNS_PANDAS_DTYPES.items()
                if sql_columns and k in sql_columns
            }

        query_string = "*" if sql_columns is None else ",".join(sql_columns)
        df_results = pd.read_sql_query(
            f"SELECT {query_string} FROM results {sql_where_str} ORDER BY rowid",
            self._conn,
            dtype=col_dtype,
        )

        if want_tags:
            if not df_results.empty:
                tags_by_record = self._load_tags(df_results["record_id"].tolist())
                tags_series = df_results["record_id"].map(
                    lambda rid: tags_by_record.get(int(rid), [])
                )
            else:
                tags_series = pd.Series(dtype="object")
            # Preserve the original column order by inserting tags at the
            # position it was requested (or appending if columns is None).
            if columns is not None and "tags" in columns:
                tag_index = columns.index("tags")
                # Adjust for columns already present in the dataframe (which
                # don't include "tags" since it was stripped from the SQL).
                df_columns = df_results.columns.tolist()
                insert_at = tag_index
                if insert_at > len(df_columns):
                    insert_at = len(df_columns)
                df_results.insert(insert_at, "tags", tags_series)
            else:
                df_results["tags"] = tags_series
        return df_results

    def get_results_page(
        self,
        *,
        label=None,
        priors=None,
        has_note=None,
        include_users=None,
        exclude_users=None,
        cursor=None,
        limit=50,
        latest_first=True,
    ):
        """Get an ordered page of labeled results using keyset pagination.

        Instead of loading the whole results table and slicing in pandas, this
        builds a SQL query that pushes the cheap filters into the WHERE clause,
        orders by ``(time, record_id)`` and returns at most ``limit`` rows after
        the given ``cursor``. Only the returned rows have their ``tags`` JSON
        parsed.

        Parameters
        ----------
        label : int | None
            Keep only rows with this label (1 or 0). ``None`` keeps all labeled
            rows.
        priors : bool | None
            ``True`` keeps only priors (``querier IS NULL``), ``False`` excludes
            priors, ``None`` keeps both.
        has_note : bool | None
            ``True`` keeps only rows with a note, ``False`` only rows without a
            note, ``None`` keeps both.
        include_users : Iterable[int] | None
            Keep only rows decided by one of these users.
        exclude_users : Iterable[int] | None
            Exclude rows decided by these users (rows without a user are kept).
        cursor : tuple[float | None, int] | None
            ``(time, record_id)`` of the last row of the previous page. ``None``
            starts from the beginning.
        limit : int
            Maximum number of rows to return.
        latest_first : bool
            Order by descending time when ``True`` (most recent first).

        Returns
        -------
        pd.DataFrame
            Up to ``limit`` rows of the results table, ordered, with ``tags``
            parsed.
        """
        where = ["label IS NOT NULL"]
        params = {}

        # Keep only the base record of each group. A base record is one whose
        # ``duplicate_of`` is NULL, which is equivalent to
        # ``record_id IN (SELECT group_id FROM record)`` but, as a correlated
        # EXISTS, lets SQLite drive the query with the collection index
        # (ordering + cursor range) instead of the group subquery.
        where.append(
            f"EXISTS (SELECT 1 FROM {self.record_table_name} AS rec "
            "WHERE rec.record_id = results.record_id "
            "AND rec.duplicate_of IS NULL)"
        )

        if label is not None:
            where.append("label = :label")
            params["label"] = int(label)

        if priors is True:
            where.append("querier IS NULL")
        elif priors is False:
            where.append("querier IS NOT NULL")

        if has_note is True:
            where.append("note IS NOT NULL")
        elif has_note is False:
            where.append("note IS NULL")

        include_users = list(include_users) if include_users else []
        if include_users:
            keys = [f":iu{i}" for i in range(len(include_users))]
            where.append(f"user_id IN ({', '.join(keys)})")
            for k, v in zip(keys, include_users):
                params[k[1:]] = int(v)

        exclude_users = list(exclude_users) if exclude_users else []
        if exclude_users:
            keys = [f":eu{i}" for i in range(len(exclude_users))]
            # Mirror pandas ``~isin`` which keeps rows with a NULL user_id.
            where.append(f"(user_id IS NULL OR user_id NOT IN ({', '.join(keys)}))")
            for k, v in zip(keys, exclude_users):
                params[k[1:]] = int(v)

        if cursor is not None:
            cursor_time, cursor_id = cursor
            params["cid"] = int(cursor_id)
            if latest_first:
                if cursor_time is None:
                    where.append("(time IS NULL AND record_id < :cid)")
                else:
                    params["ct"] = float(cursor_time)
                    where.append(
                        "(time IS NULL OR time < :ct "
                        "OR (time = :ct AND record_id < :cid))"
                    )
            else:
                if cursor_time is None:
                    where.append("(time IS NULL AND record_id > :cid)")
                else:
                    params["ct"] = float(cursor_time)
                    # Null-time rows sort last, so they remain "after" a
                    # non-null cursor and must be included here too.
                    where.append(
                        "(time IS NULL OR time > :ct "
                        "OR (time = :ct AND record_id > :cid))"
                    )

        if latest_first:
            order_by = "(time IS NULL) ASC, time DESC, record_id DESC"
        else:
            order_by = "(time IS NULL) ASC, time ASC, record_id ASC"

        params["limit"] = int(limit)

        df_results = pd.read_sql_query(
            f"""SELECT * FROM results
            WHERE {" AND ".join(where)}
            ORDER BY {order_by}
            LIMIT :limit""",
            self._conn,
            params=params,
            dtype=RESULTS_TABLE_COLUMNS_PANDAS_DTYPES,
        )
        if not df_results.empty:
            tags_by_record = self._load_tags(df_results["record_id"].tolist())
            df_results["tags"] = df_results["record_id"].map(
                lambda rid: tags_by_record.get(int(rid), [])
            )
        else:
            df_results["tags"] = pd.Series(dtype="object")
        return df_results

    def get_priors(self):
        """Get the record ids of the priors.

        Returns
        -------
        pd.DataFrame:
            The result records of the priors in the order they were added. If multiple
            records are in the same group, only the base record of the group is
            returned.
        """
        df_results = pd.read_sql_query(
            f"""
            SELECT * FROM results
            WHERE results.querier is NULL
            AND results.label is not NULL
            AND record_id IN (
                SELECT group_id FROM {self.record_table_name}
            )
            ORDER BY rowid
            """,
            self._conn,
            dtype=RESULTS_TABLE_COLUMNS_PANDAS_DTYPES,
        )
        if not df_results.empty:
            tags_by_record = self._load_tags(df_results["record_id"].tolist())
            df_results["tags"] = df_results["record_id"].map(
                lambda rid: tags_by_record.get(int(rid), [])
            )
        else:
            df_results["tags"] = pd.Series(dtype="object")
        return df_results

    def get_pool(self):
        """Get the unlabeled, not-pending records in ranking order.

        Returns
        -------
        pd.Series
            Series containing the record_ids of the unlabeled, not pending
            records, in the order of the last available ranking. If the state does not
            yet contain a last ranking, the return value will be an empty dataframe. If
            multiple records are in the same group, only the base record of the group is
            returned.
        """

        return pd.read_sql_query(
            f"""SELECT record_id, last_ranking.ranking
                FROM last_ranking
                LEFT JOIN results
                USING (record_id)
                WHERE results.record_id is null AND last_ranking.record_id IN (
                    SELECT group_id FROM {self.record_table_name}
                )
                ORDER BY ranking
                """,
            self._conn,
        )["record_id"]

    def get_unlabeled(self, groups=False):
        """Get the unlabeled record ids in ranking order.

        Records that have no label or no entry in the results table are considered
        unlabeled.

        Parameters
        ----------
        groups : bool
            If True, return all records in each unlabeled group. If False,
            return only group representatives (record_id == group_id).

        Returns
        -------
        pd.Series
            Series of record_ids of unlabeled records ordered by ranking.
        """
        if groups:
            sql_group_filter = ""
        else:
            sql_group_filter = (
                f"AND record_id IN (SELECT group_id FROM {self.record_table_name})"
            )

        return pd.read_sql_query(
            f"""SELECT record_id, last_ranking.ranking
            FROM last_ranking
            JOIN {self.record_table_name} USING (record_id)
            LEFT JOIN results USING (record_id)
            WHERE (results.record_id IS NULL OR results.label IS NULL)
            {sql_group_filter}
            ORDER BY ranking
            """,
            self._conn,
        )["record_id"]

    def get_pending(self, user_id=None):
        """Get pending records from the results table.

        Parameters
        ----------
        user_id: int
            User id of the user who labeled the records.

        Returns
        -------
        pd.DataFrame
            DataFrame with pending results records.
        """
        query = f"""SELECT * FROM results WHERE label is null AND record_id IN (
            SELECT group_id FROM {self.record_table_name}
        )"""
        params = None
        if user_id is not None:
            query += " AND user_id=?"
            params = (user_id,)
        query += " ORDER BY rowid"

        return pd.read_sql_query(
            query,
            self._conn,
            params=params,
            dtype=RESULTS_TABLE_COLUMNS_PANDAS_DTYPES,
        )

    def get_decision_changes(self):
        """Get the record ids for any decision changes.

        Get the record ids of the records whose labels have been changed after the
        original labeling action.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns 'record_id', 'label', 'time', and 'user_id' for each
            record of which the labeling decision was changed.
        """

        return pd.read_sql_query("SELECT * FROM decision_changes", self._conn)
