import io
import json
from pathlib import Path

import asreview as asr
import asreview.webapp.tests.utils.api_utils as au


def _dispatch_count(project):
    with asr.Project(project.project_path).db as db:
        return db._conn.execute(
            "SELECT COUNT(*) FROM llm_dispatch").fetchone()[0]


def _setup_review(client, project):
    """Label one relevant and one irrelevant record; add a last ranking."""
    au.label_random_project_data_record(client, project, 1)
    au.label_random_project_data_record(client, project, 0)
    asr_project = asr.Project(project.project_path)
    record_ids = asr_project.db.input["record_id"].to_list()
    with asr_project.db as db:
        db.add_last_ranking(record_ids, "nb", "max", "double", "tfidf", 2)


def _put_llm_settings(client, project, api_key):
    return client.put(
        f"/api/projects/{au.get_project_id(project)}/llm_settings",
        json={"api_key": api_key},
    )


def _get_record(client, project):
    return client.get(f"/api/projects/{au.get_project_id(project)}/get_record")


def test_no_key_plain_order_no_queue(client, project):
    """Without an API key: plain ASReview order, llm null, queue stays empty."""
    _setup_review(client, project)

    r = _get_record(client, project)
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "review"
    assert body["result"] is not None
    assert body["result"]["llm"] is None
    assert _dispatch_count(project) == 0


def test_key_opens_gate(client, project):
    """Setting an API key queues records and serves llm state."""
    _setup_review(client, project)

    res = _put_llm_settings(client, project, "sk-test")
    assert res.status_code == 200

    r = _get_record(client, project)
    assert r.status_code == 200
    body = r.get_json()
    assert _dispatch_count(project) > 0
    assert body["result"]["llm"] is not None


def test_whitespace_key_behaves_like_no_key(client, project):
    """A whitespace-only API key is treated as no key."""
    _setup_review(client, project)

    res = _put_llm_settings(client, project, "   ")
    assert res.status_code == 200

    r = _get_record(client, project)
    assert r.status_code == 200
    assert r.get_json()["result"]["llm"] is None
    assert _dispatch_count(project) == 0


def test_reprocess_409_without_key(client, project):
    """Without a key, reprocess is rejected and queues nothing."""
    asr_project = asr.Project(project.project_path)
    rid = int(asr_project.db.input["record_id"].to_list()[0])

    r = client.post(
        f"/api/projects/{au.get_project_id(project)}/record/{rid}/reprocess"
    )
    assert r.status_code == 409
    assert _dispatch_count(project) == 0


def test_upload_pdf_queues_only_with_key(client, project, monkeypatch):
    """A PDF upload stores the attachment always, but queues only with a key."""
    Path(project.project_path, "zotero.json").write_text(
        json.dumps({"group_id": "1", "api_key": "K"})
    )

    asr_project = asr.Project(project.project_path)
    with asr_project.db as db:
        store = db.input
        rids = store["record_id"].to_list()[:2]
        rid, rid2 = int(rids[0]), int(rids[1])
        with store.Session() as s, s.begin():
            s.get(store.record_cls, rid).original_id = "ITEMKEY1"
            s.get(store.record_cls, rid2).original_id = "ITEMKEY2"

    monkeypatch.setattr(
        "asreview.webapp._api.projects.upload_pdf_to_zotero",
        lambda config, original_id, tmp_path: "ABCD1234",
    )

    pid = au.get_project_id(project)

    # No key: the upload stores the attachment but queues nothing.
    r = client.post(
        f"/api/projects/{pid}/record/{rid}/upload_pdf",
        data={"file": (io.BytesIO(b"%PDF-1.4"), "a.pdf")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    with asr_project.db as db:
        assert db.input.get_records(rid).attachment == "ABCD1234"
    assert _dispatch_count(project) == 0

    # With a key the same upload re-queues the record.
    res = _put_llm_settings(client, project, "sk-test")
    assert res.status_code == 200
    r = client.post(
        f"/api/projects/{pid}/record/{rid2}/upload_pdf",
        data={"file": (io.BytesIO(b"%PDF-1.4"), "b.pdf")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert _dispatch_count(project) == 1


def test_key_removed_ignores_existing_queue(client, project):
    """Removing the key stops serving llm state and top-ups; rows remain."""
    _setup_review(client, project)

    res = _put_llm_settings(client, project, "sk-test")
    assert res.status_code == 200
    r = _get_record(client, project)
    assert r.status_code == 200
    assert r.get_json()["result"]["llm"] is not None
    queued = _dispatch_count(project)
    assert queued > 0

    res = _put_llm_settings(client, project, "")
    assert res.status_code == 200
    r = _get_record(client, project)
    assert r.status_code == 200
    assert r.get_json()["result"]["llm"] is None
    assert _dispatch_count(project) == queued
