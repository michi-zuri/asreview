from datetime import datetime, timedelta, timezone
from pathlib import Path

import asreview as asr
from asreview.data.record import Record
from asreview.webapp._api import zotero_sync
from asreview.webapp._api.zotero import ZoteroConfig
from asreview.webapp._api.zotero import ZoteroLookupError


def _fake_fetch(monkeypatch, results):
    """Monkeypatch zotero.fetch_pdf_attachment_key.

    `results` maps an item key to the attachment key to return (or None).
    Returns the list of item keys the stub was called with.
    """
    fetched = []

    def stub(config, item_key):
        fetched.append(item_key)
        return results.get(item_key)

    monkeypatch.setattr(
        "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
    )
    return fetched


def test_records_due_for_check_order_and_filtering():
    """Mixed rows: only due records, never-checked first, oldest timestamp
    first."""
    now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        (1, "K1", None, None),
        (2, "K2", "ABCD1234", None),
        (3, "K3", (now - timedelta(hours=23, minutes=59)).isoformat(), None),
        (4, "K4", (now - timedelta(hours=30)).isoformat(), None),
        (5, "K5", (now - timedelta(hours=48)).isoformat(), None),
        (6, None, None, None),
        (7, "K7", None, 1),
        (8, "K8", float("nan"), float("nan")),
    ]

    ids = zotero_sync.records_due_for_check(rows, 86400, now=now)

    assert ids == [1, 8, 5, 4]


def test_sync_db_attachments_24h_window(tmp_path, monkeypatch):
    """Due records are looked up once; a second pass is a no-op for 24 h."""
    now = datetime.now(timezone.utc)
    with asr.Database(Path(tmp_path, "t.db")) as db:
        db.create_tables()
        db.input.add_records([
            Record(0, "d", original_id="ITEMA001"),
            Record(
                1, "d", original_id="ITEMB002",
                attachment=(now - timedelta(hours=25)).isoformat(),
            ),
            Record(
                2, "d", original_id="ITEMC003",
                attachment=(now - timedelta(hours=1)).isoformat(),
            ),
        ])

        fetched = _fake_fetch(monkeypatch, {"ITEMA001": "ABCD1234"})
        sleeps = []

        config = ZoteroConfig("1", None, "K", 300)
        n = zotero_sync.sync_db_attachments(
            db, config, delay=1.0, sleep=sleeps.append
        )

        assert n == 2
        assert fetched == ["ITEMA001", "ITEMB002"]
        assert sleeps == [1.0, 1.0]

        assert db.input.get_records(0).attachment == "ABCD1234"

        rec1 = db.input.get_records(1)
        assert datetime.fromisoformat(rec1.attachment) > (
            now - timedelta(minutes=1)
        )

        assert db.input.get_records(2).attachment == (
            now - timedelta(hours=1)
        ).isoformat()

        # Nothing is due any more: the second pass performs no lookups.
        n = zotero_sync.sync_db_attachments(
            db, config, delay=1.0, sleep=sleeps.append
        )
        assert n == 0
        assert fetched == ["ITEMA001", "ITEMB002"]


def test_sync_project_attachments_unconfigured(tmp_path, monkeypatch):
    """No zotero.json: returns 0 without any Zotero request."""
    fetched = _fake_fetch(monkeypatch, {})

    n = zotero_sync.sync_project_attachments(tmp_path)

    assert n == 0
    assert fetched == []


def test_sync_db_attachments_stops_on_lookup_error(tmp_path, monkeypatch):
    """A ZoteroLookupError stops the pass after one fetch; nothing written."""
    with asr.Database(Path(tmp_path, "t.db")) as db:
        db.create_tables()
        db.input.add_records([
            Record(0, "d", original_id="ITEMA001"),
            Record(1, "d", original_id="ITEMB002"),
        ])

        fetched = []

        def stub(config, item_key):
            fetched.append(item_key)
            raise ZoteroLookupError("network down")

        monkeypatch.setattr(
            "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
        )

        n = zotero_sync.sync_db_attachments(
            db, ZoteroConfig("1", None, "K", 300),
            delay=1.0, sleep=lambda s: None,
        )

        assert n == 0
        assert fetched == ["ITEMA001"]
        assert db.input.get_records(0).attachment is None
        assert db.input.get_records(1).attachment is None


def _dispatch_rows(db):
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT record_id, status FROM llm_dispatch ORDER BY record_id"
    ).fetchall()


def test_sync_clears_missing_pdf_rows(tmp_path, monkeypatch):
    """A found PDF key clears the record's 'missing_pdf' dispatch row."""
    with asr.Database(Path(tmp_path, "t.db")) as db:
        db.create_tables()
        db.input.add_records([
            Record(0, "d", original_id="ITEMA001", attachment="ABCD1234"),
            Record(1, "d", original_id="ITEMB002"),
            Record(2, "d", original_id="ITEMC003"),
        ])
        for rid in (0, 1, 2):
            db.force_requeue(rid, "H")
            db.mark_dispatch_missing_pdf(rid)

        _fake_fetch(monkeypatch, {"ITEMB002": "WXYZ9876", "ITEMC003": None})

        n = zotero_sync.sync_db_attachments(
            db, ZoteroConfig("1", None, "K", 300),
            delay=1.0, sleep=lambda s: None,
        )

        assert n == 2
        # Records 0 (key already cached) and 1 (key found now) have NO
        # dispatch row; record 2 (still no PDF) keeps 'missing_pdf'. Nothing
        # was queued.
        assert _dispatch_rows(db) == [(2, "missing_pdf")]


def test_sync_sweep_runs_after_lookup_error(tmp_path, monkeypatch):
    """The missing_pdf sweep runs even when the pass breaks on a lookup error."""
    with asr.Database(Path(tmp_path, "t.db")) as db:
        db.create_tables()
        db.input.add_records([
            Record(0, "d", original_id="ITEMA001", attachment="ABCD1234"),
            Record(1, "d", original_id="ITEMB002"),
        ])
        for rid in (0, 1):
            db.force_requeue(rid, "H")
            db.mark_dispatch_missing_pdf(rid)

        fetched = []

        def stub(config, item_key):
            fetched.append(item_key)
            raise ZoteroLookupError("network down")

        monkeypatch.setattr(
            "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
        )

        n = zotero_sync.sync_db_attachments(
            db, ZoteroConfig("1", None, "K", 300),
            delay=1.0, sleep=lambda s: None,
        )

        assert n == 0
        assert fetched == ["ITEMB002"]
        # The sweep needs no network: record 0 already had a key, so its
        # missing_pdf row is deleted despite the aborted pass.
        assert _dispatch_rows(db) == [(1, "missing_pdf")]
