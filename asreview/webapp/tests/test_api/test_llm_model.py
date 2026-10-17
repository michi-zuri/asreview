import json
import zipfile
from pathlib import Path

import asreview as asr
import asreview.webapp.tests.utils.api_utils as au
from asreview.webapp._api.projects import AVAILABLE_LLM_MODELS
from asreview.webapp._api.projects import DEFAULT_LLM_MODEL


def _get_settings(client, project):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/llm_settings")


def _put_settings(client, project, body):
    return client.put(
        f"/api/projects/{au.get_project_id(project)}/llm_settings",
        json=body,
    )


def _project_config(project):
    return asr.Project(project.project_path).config


def test_default_model_and_catalog(client, project):
    """GET returns the default model + catalog; create wrote the default."""
    r = _get_settings(client, project)
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"] == DEFAULT_LLM_MODEL
    assert body["models"] == AVAILABLE_LLM_MODELS

    assert _project_config(project)["llm"]["model"] == DEFAULT_LLM_MODEL


def test_put_model_persists(client, project):
    """PUT with a known model id persists it."""
    r = _put_settings(client, project, {"model": "claude-sonnet-4-6"})
    assert r.status_code == 200

    assert _get_settings(client, project).get_json()["model"] == \
        "claude-sonnet-4-6"
    assert _project_config(project)["llm"]["model"] == "claude-sonnet-4-6"


def test_put_unknown_model_400(client, project):
    """PUT with an unknown model id is rejected; config unchanged."""
    r = _put_settings(client, project, {"model": "claude-nope"})
    assert r.status_code == 400

    assert _project_config(project)["llm"]["model"] == DEFAULT_LLM_MODEL


def test_model_change_invalidates_unlabeled_only(client, project):
    """Changing the model deletes results and re-queues unlabeled records.

    The invalidation runs even without an API key (pure DB bookkeeping).
    """
    asr_project = asr.Project(project.project_path)
    rids = [int(r) for r in asr_project.db.input["record_id"].to_list()[:2]]
    rid1, rid2 = rids
    with asr_project.db as db:
        db.force_requeue(rid1, "H")
        db.force_requeue(rid2, "H")
        db.store_llm_result(rid1, "H", DEFAULT_LLM_MODEL, '{"labels":[]}', 1, 2)
        db.store_llm_result(rid2, "H", DEFAULT_LLM_MODEL, '{"labels":[]}', 1, 2)
        db.label_record(rid2, 1, user_id=1)

    r = _put_settings(client, project, {"model": "claude-haiku-4-5-20251001"})
    assert r.status_code == 200

    with asr_project.db as db:
        cur = db._conn.cursor()
        # rid1 (no final decision): results deleted, dispatch re-queued.
        assert cur.execute(
            "SELECT COUNT(*) FROM llm_results WHERE record_id = ?",
            (rid1,),
        ).fetchone()[0] == 0
        assert cur.execute(
            "SELECT status, prompt_hash, attempts, last_error "
            "FROM llm_dispatch WHERE record_id = ?", (rid1,),
        ).fetchone() == ("queued", "H", 0, None)
        # rid2 (labeled): result and 'ready' status kept.
        assert cur.execute(
            "SELECT COUNT(*) FROM llm_results WHERE record_id = ?",
            (rid2,),
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT status FROM llm_dispatch WHERE record_id = ?", (rid2,),
        ).fetchone()[0] == "ready"


def _export_project_file(client, project, tmp_path, name="export.asreview"):
    r = au.export_project(client, project)
    assert r.status_code == 200
    path = Path(tmp_path, name)
    path.write_bytes(r.data)
    return path


def test_import_preserves_model(client, project, tmp_path):
    """Export/import round-trip keeps the configured model."""
    assert _put_settings(
        client, project, {"model": "claude-sonnet-4-6"}).status_code == 200

    r = au.import_project(
        client, _export_project_file(client, project, tmp_path))
    assert r.status_code == 200
    assert r.json["data"]["llm"]["model"] == "claude-sonnet-4-6"


def test_import_defaults_model_when_missing(client, project, tmp_path):
    """Imported projects without a stored model get the default."""
    src = _export_project_file(client, project, tmp_path)
    # Rewrite the export without the llm config section.
    stripped = Path(tmp_path, "stripped.asreview")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(stripped, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "project.json":
                cfg = json.loads(data)
                cfg.pop("llm", None)
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(item, data)

    r = au.import_project(client, stripped)
    assert r.status_code == 200
    assert r.json["data"]["llm"]["model"] == DEFAULT_LLM_MODEL
