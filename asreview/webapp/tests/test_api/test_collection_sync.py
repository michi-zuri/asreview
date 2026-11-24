"""Spec 0001 Part 4a: the Collection sync endpoints.

Covers ``labeled/changes``, ``labeled/ids`` and ``records?ids=``, with the
key equivalence test that pins ``labeled/ids`` to the exact ids and order of
the ``/labeled`` pages concatenated down to the same boundary.
"""

import sqlite3

import pytest

import asreview as asr
import asreview.webapp.tests.utils.api_utils as au
from asreview.database.database import Database
from asreview.webapp._api.projects import _encode_cursor
from asreview.webapp._api.projects import _labeled_filter_signature


def _labeled(client, project, **params):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/labeled", query_string=params
    )


def _changes(client, project, **params):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/labeled/changes",
        query_string=params,
    )


def _ids(client, project, **params):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/labeled/ids",
        query_string=params,
    )


def _records(client, project, **params):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/records", query_string=params
    )


PER_PAGE = 2
PAGES = 3
FILTERS = ([], ["has_note"], ["pdf=false"], ["pdf"])
SUBSETS = ("relevant", "irrelevant", "all", "unlabeled")


def _setup_equivalence(client, project):
    """Label 40 records and give the matrix of the equivalence test real work.

    Records labeled 0..39 alternate relevant / irrelevant. Notes on labeled
    records with i % 4 in (0, 1) (10 relevant, 10 irrelevant) and on 8 pending
    (checked-out, unlabeled) records spread over the id range. A Zotero
    attachment key on labeled records with i % 5 < 3 (12 of 20 per label) and
    on every second unlabeled record. With PER_PAGE * PAGES = 6, every
    (subset, filter) case has more than 6 matches, so every boundary is a real
    cursor.
    """
    res = au.search_project_data(client, project, query="The&n_max=80")
    ids = [r["record_id"] for r in res.json["result"]][:40]
    assert len(ids) == 40
    for i, rid in enumerate(ids):
        au.label_project_record(client, project, rid, label=1 if i % 2 == 0 else 0)
    with asr.Project(project.project_path) as p:
        db = p.db
        for i, rid in enumerate(ids):
            if i % 4 in (0, 1):
                db.update_note(rid, "hello")
        all_ids = db.input["record_id"].tolist()
        # Rank every 50th record first, so the pending records are spread
        # over the id range (not all at one end).
        first = all_ids[::50]
        ranked = first + [r for r in all_ids if r not in first]
        db.add_last_ranking(ranked, "nb", "max", "balanced", "tfidf", 0)
        for _ in range(8):
            pending = db.query_top_ranked(user_id=1)
        for rid in pending["record_id"].tolist():
            db.update_note(int(rid), "pending note")
        labeled = set(ids)
        unlabeled = [r for r in all_ids if r not in labeled]
        with_pdf = [r for i, r in enumerate(ids) if i % 5 < 3]
        for rid in with_pdf + unlabeled[::2]:
            db.input.set_attachment(rid, "ABCD1234")
    return ids


def _page_three(client, project, subset, latest_first, filters):
    """Page /labeled up to PAGES pages; return (ids, boundary, last next_cursor)."""
    params = {"subset": subset, "latest_first": latest_first, "per_page": PER_PAGE}
    if filters:
        params["filter"] = filters
    ids, cursor, last_next = [], None, None
    for _ in range(PAGES):
        if cursor:
            params["cursor"] = cursor
        page = _labeled(client, project, **params).json
        ids.extend(item["record_id"] for item in page["result"])
        last_next = page["next_cursor"]
        if not last_next:
            break
        cursor = last_next
    return ids, (last_next or "end"), last_next


def test_ids_equal_concatenated_pages(client, project):
    _setup_equivalence(client, project)
    unfiltered = {}
    for subset in SUBSETS:
        for latest_first in (1, 0):
            for filters in FILTERS:
                case = (subset, latest_first, tuple(filters))
                paged, boundary, last_next = _page_three(
                    client, project, subset, latest_first, filters
                )
                # The case is real: a cursor boundary after full pages ...
                assert last_next is not None, case
                assert len(paged) == PER_PAGE * PAGES, case
                # ... and the filter removes a row inside the window.
                if not filters:
                    unfiltered[(subset, latest_first)] = paged
                else:
                    assert paged != unfiltered[(subset, latest_first)], case
                params = {"subset": subset, "latest_first": latest_first}
                if filters:
                    params["filter"] = filters
                r = _ids(client, project, boundary=boundary, **params)
                assert r.status_code == 200, case
                assert [i[0] for i in r.json["items"]] == paged, case
                assert r.json["next_cursor"] is not None, case

    # boundary="end" returns every match, in order, down to exhaustion.
    params = {"subset": "all", "latest_first": 1, "per_page": 7}
    all_ids = []
    cursor = None
    while True:
        if cursor:
            params["cursor"] = cursor
        page = _labeled(client, project, **params).json
        all_ids.extend(item["record_id"] for item in page["result"])
        cursor = page["next_cursor"]
        if not cursor:
            break
    r = _ids(client, project, subset="all", latest_first=1, boundary="end")
    assert [i[0] for i in r.json["items"]] == all_ids
    assert r.json["next_cursor"] is None

    # Every loaded row leaving the filter: items == [] and next_cursor is the
    # boundary that was sent.
    params = {
        "subset": "unlabeled",
        "latest_first": 1,
        "per_page": 5,
        "filter": "pdf=false",
    }
    page1 = _labeled(client, project, **params).json
    params["cursor"] = page1["next_cursor"]
    page2 = _labeled(client, project, **params).json
    assert page2["next_cursor"] is not None
    loaded = [item["record_id"] for item in page1["result"] + page2["result"]]
    boundary = page2["next_cursor"]
    with asr.Project(project.project_path) as p:
        for rid in loaded:
            p.db.input.set_attachment(rid, "ABCD1234")
    r = _ids(
        client,
        project,
        subset="unlabeled",
        latest_first=1,
        filter="pdf=false",
        boundary=boundary,
    )
    assert r.json["items"] == []
    assert r.json["next_cursor"] == boundary


def test_changes_in_view_left_filter(client, project):
    res = au.search_project_data(client, project, query="The&n_max=30")
    ids = [r["record_id"] for r in res.json["result"]][:12]
    for rid in ids[:2]:
        au.label_project_record(client, project, rid, label=1)

    params = {"subset": "unlabeled", "filter": "pdf=false", "per_page": 5}
    page1 = _labeled(client, project, **params).json
    params["cursor"] = page1["next_cursor"]
    page2 = _labeled(client, project, **params).json
    assert page2["next_cursor"] is not None
    boundary = page2["next_cursor"]
    old_wm = page1["watermark"]

    target = page1["result"][0]["record_id"]
    with asr.Project(project.project_path) as p:
        p.db.input.set_attachment(target, "ABCD1234")

    r = _changes(
        client,
        project,
        subset="unlabeled",
        filter="pdf=false",
        since=old_wm,
        boundary=boundary,
    )
    assert r.status_code == 200
    body = r.json
    assert "reset" not in body
    changed = {c["record_id"]: c for c in body["changes"]}
    assert target in changed
    assert changed[target]["in_view"] is False
    assert changed[target]["sort_key"] is None


def test_changes_entered_above_and_below_boundary(client, project):
    res = au.search_project_data(client, project, query="The&n_max=30")
    ids = [r["record_id"] for r in res.json["result"]][:11]
    for rid in ids[:10]:
        au.label_project_record(client, project, rid, label=1)

    # latest_first (newest first): 1 page loaded.
    params = {"subset": "relevant", "latest_first": 1, "per_page": 5}
    page1 = _labeled(client, project, **params).json
    assert page1["next_cursor"] is not None
    boundary = page1["next_cursor"]
    old_wm = page1["watermark"]

    new_rid = ids[10]
    au.label_project_record(client, project, new_rid, label=1)

    r = _changes(
        client,
        project,
        subset="relevant",
        latest_first=1,
        since=old_wm,
        boundary=boundary,
    )
    body = r.json
    assert "reset" not in body
    changed = {c["record_id"]: c for c in body["changes"]}
    assert new_rid in changed
    assert changed[new_rid]["in_view"] is True
    assert changed[new_rid]["sort_key"][0] is not None

    # oldest first: the same new record now sorts beyond the boundary.
    params = {"subset": "relevant", "latest_first": 0, "per_page": 5}
    page1_old = _labeled(client, project, **params).json
    assert page1_old["next_cursor"] is not None
    boundary_old = page1_old["next_cursor"]
    r2 = _changes(
        client,
        project,
        subset="relevant",
        latest_first=0,
        since=old_wm,
        boundary=boundary_old,
    )
    changed2 = {c["record_id"]: c for c in r2.json["changes"]}
    assert changed2[new_rid]["in_view"] is False
    assert changed2[new_rid]["sort_key"] is None


def test_changes_empty_poll(client, project):
    res = au.search_project_data(client, project, query="The&n_max=10")
    rid = res.json["result"][0]["record_id"]
    au.label_project_record(client, project, rid, label=1)
    wm = _labeled(client, project).json["watermark"]
    r = _changes(client, project, since=wm, boundary="end")
    assert r.status_code == 200
    assert "reset" not in r.json
    assert r.json["changes"] == []


def test_changes_reset_over_200(client, project):
    with asr.Project(project.project_path) as p:
        ids = p.db.input["record_id"].tolist()[:201]
        assert len(ids) == 201
        for rid in ids:
            p.db.label_record(rid, 1)
    r = _changes(client, project, since=0, boundary="end")
    assert r.status_code == 200
    assert r.json.get("reset") is True
    assert "changes" not in r.json


def test_changes_unaffected_by_prune(client, project):
    res = au.search_project_data(client, project, query="The&n_max=10")
    ids = [r["record_id"] for r in res.json["result"]][:2]
    for rid in ids:
        au.label_project_record(client, project, rid, label=1)

    with asr.Project(project.project_path) as p:
        now = (
            p.db._conn.execute("SELECT MAX(changed_at) FROM change_log").fetchone()[0]
            + 8 * 86400
        )
        p.db.prune_change_log(now=now)

    r = _changes(client, project, since=0, boundary="end")
    assert r.status_code == 200
    assert "reset" not in r.json
    got = {c["record_id"] for c in r.json["changes"]}
    assert got == set(ids)
    assert all(c["version"] > 0 for c in r.json["changes"])


def test_changes_note_edit_in_view(client, project):
    res = au.search_project_data(client, project, query="The&n_max=10")
    rid = res.json["result"][0]["record_id"]
    au.label_project_record(client, project, rid, label=1)

    page = _labeled(client, project, subset="relevant").json
    old_wm = page["watermark"]
    row = next(r for r in page["result"] if r["record_id"] == rid)

    client.put(
        f"/api/projects/{au.get_project_id(project)}/record/{rid}/note",
        data={"note": "hello"},
    )

    r = _changes(client, project, subset="relevant", since=old_wm, boundary="end")
    changed = {c["record_id"]: c for c in r.json["changes"]}
    assert changed[rid]["in_view"] is True
    # A note edit does not change `time`, so the sort key is unchanged.
    assert changed[rid]["sort_key"] == [row["state"]["time"], rid]


def test_records_same_shape(client, project):
    res = au.search_project_data(client, project, query="The&n_max=10")
    ids = [r["record_id"] for r in res.json["result"]][:2]
    au.label_project_record(client, project, ids[0], label=1)
    au.label_project_record(client, project, ids[1], label=0)

    labeled = _labeled(client, project).json
    labeled_by_id = {r["record_id"]: r for r in labeled["result"]}
    r = _records(client, project, ids=f"{ids[0]},{ids[1]}")
    assert r.status_code == 200
    assert len(r.json["result"]) == 2
    for row in r.json["result"]:
        assert row == labeled_by_id[row["record_id"]]

    unlabeled = _labeled(client, project, subset="unlabeled", per_page=50).json
    unlabeled_ids = [row["record_id"] for row in unlabeled["result"]][:2]
    unlabeled_by_id = {row["record_id"]: row for row in unlabeled["result"]}
    r2 = _records(client, project, ids=",".join(map(str, unlabeled_ids)))
    assert r2.status_code == 200
    assert len(r2.json["result"]) == 2
    for row in r2.json["result"]:
        assert row == unlabeled_by_id[row["record_id"]]

    # Too many ids, or a non-integer, is a 400.
    r3 = _records(client, project, ids=",".join(str(i) for i in range(501)))
    assert r3.status_code == 400
    r4 = _records(client, project, ids="a,b")
    assert r4.status_code == 400


def test_unlabeled_user_filter_checkout(client, project):
    with asr.Project(project.project_path) as p:
        db = p.db
        ids = db.input["record_id"].tolist()
        db.add_last_ranking(ids, "nb", "max", "balanced", "tfidf", 0)
        old_wm = db.get_project_state()["version"]
        pending = db.query_top_ranked(user_id=1)
        assigned_rid = int(pending["record_id"].iloc[0])

    r = _changes(
        client,
        project,
        subset="unlabeled",
        filter="user_1",
        since=old_wm,
        boundary="end",
    )
    assert r.status_code == 200
    changed = {c["record_id"]: c for c in r.json["changes"]}
    assert assigned_rid in changed
    assert changed[assigned_rid]["in_view"] is True


def test_bad_params(client, project):
    # since missing / non-int / negative → 400.
    assert _changes(client, project, boundary="end").status_code == 400
    assert _changes(client, project, since="abc", boundary="end").status_code == 400
    assert _changes(client, project, since=-1, boundary="end").status_code == 400
    # boundary garbage → 400.
    assert _changes(client, project, since=0, boundary="garbage").status_code == 400
    # a boundary minted for another filter → 400.
    sig = _labeled_filter_signature("relevant", [], True)
    cursor = _encode_cursor(123.0, 5, sig)
    r = _ids(client, project, subset="unlabeled", boundary=cursor)
    assert r.status_code == 400


def test_changes_prune_busy_does_not_fail_poll(client, project, monkeypatch):
    def busy_prune(self, *args, **kwargs):
        err = sqlite3.OperationalError("database is locked")
        err.sqlite_errorcode = sqlite3.SQLITE_BUSY
        raise err

    monkeypatch.setattr(Database, "prune_change_log", busy_prune)

    res = au.search_project_data(client, project, query="The&n_max=10")
    rid = res.json["result"][0]["record_id"]
    au.label_project_record(client, project, rid, label=1)

    r = _changes(client, project, since=0, boundary="end", subset="relevant")
    assert r.status_code == 200
    assert "reset" not in r.json
    assert rid in {c["record_id"] for c in r.json["changes"]}


def test_changes_prune_other_error_propagates(client, project, monkeypatch):
    def corrupt_prune(self, *args, **kwargs):
        err = sqlite3.OperationalError("database disk image is malformed")
        err.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
        raise err

    monkeypatch.setattr(Database, "prune_change_log", corrupt_prune)

    res = au.search_project_data(client, project, query="The&n_max=10")
    rid = res.json["result"][0]["record_id"]
    au.label_project_record(client, project, rid, label=1)

    with pytest.raises(sqlite3.OperationalError):
        _changes(client, project, since=0, boundary="end", subset="relevant")


def test_changes_prune_open_transaction_returns_409(client, project, monkeypatch):
    def mid_transaction_prune(self, *args, **kwargs):
        raise RuntimeError("prune_change_log() must not run inside an open transaction")

    monkeypatch.setattr(Database, "prune_change_log", mid_transaction_prune)

    res = au.search_project_data(client, project, query="The&n_max=10")
    rid = res.json["result"][0]["record_id"]
    au.label_project_record(client, project, rid, label=1)

    r = _changes(client, project, since=0, boundary="end", subset="relevant")
    assert r.status_code == 409
    assert "janitor" in r.json["message"]
