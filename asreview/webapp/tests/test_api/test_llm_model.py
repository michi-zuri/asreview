import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import asreview as asr
import asreview.webapp._api.projects as projects_mod
import asreview.webapp.tests.utils.api_utils as au
from asreview.webapp._api.projects import DEFAULT_LLM_MODEL
from asreview.webapp._api.projects import FALLBACK_LLM_MODELS


def _get_settings(client, project):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/llm_settings")


def _put_settings(client, project, body):
    return client.put(
        f"/api/projects/{au.get_project_id(project)}/llm_settings",
        json=body,
    )


def _post_models(client, project, body):
    return client.post(
        f"/api/projects/{au.get_project_id(project)}/llm_models",
        json=body,
    )


def _project_config(project):
    return asr.Project(project.project_path).config


def _set_api_key(client, project, api_key="sk-ant-test"):
    r = _put_settings(client, project, {"api_key": api_key})
    assert r.status_code == 200


def _catalog_pages(default, retired=()):
    """Map listing kind -> page dicts; None is the active+deprecated one."""
    return {None: default, "retired": retired}


class FakeModelsApi:
    """Stand-in for anthropic.Anthropic that serves canned Models pages.

    ``pages`` maps the listing kind (None for the default active+deprecated
    listing, "retired" for the retired one) to a list of page dicts, each
    with "data" (model dicts), "has_more", "first_id" and "last_id".
    ``calls`` records every models.list() kwargs dict; ``init_calls``
    records the Anthropic() constructor kwargs.
    """

    def __init__(self, pages, fail_kinds=()):
        self.pages = pages
        self.fail_kinds = set(fail_kinds)
        self.calls = []
        self.init_calls = []

    def __call__(self, **kwargs):
        self.init_calls.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(list=self._list))

    def _list(self, **kwargs):
        self.calls.append(kwargs)
        kind = (kwargs.get("extra_query") or {}).get("lifecycle[]")
        if kind in self.fail_kinds:
            raise OSError(f"{kind} listing down")
        page_list = self.pages[kind]
        after_id = kwargs.get("after_id")
        if not page_list:
            return SimpleNamespace(data=[], has_more=False, first_id=None,
                                   last_id=None)
        if after_id is None:
            page = page_list[0]
        else:
            page = next(p for p in page_list if p.get("first_id") == after_id)
        return SimpleNamespace(
            data=[SimpleNamespace(**m) for m in page["data"]],
            has_more=page.get("has_more", False),
            first_id=page.get("first_id"),
            last_id=page.get("last_id"),
        )


def test_default_model_and_catalog(client, project):
    """GET returns the default model; without a key the fallback catalog."""
    r = _get_settings(client, project)
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"] == DEFAULT_LLM_MODEL
    assert body["models"] == FALLBACK_LLM_MODELS

    assert _project_config(project)["llm"]["model"] == DEFAULT_LLM_MODEL


def test_put_model_persists(client, project):
    """PUT with a known model id persists it."""
    r = _put_settings(client, project, {"model": "claude-sonnet-4-6"})
    assert r.status_code == 200

    assert _get_settings(client, project).get_json()["model"] == \
        "claude-sonnet-4-6"
    assert _project_config(project)["llm"]["model"] == "claude-sonnet-4-6"


def test_catalog_fetched_live_from_models_api(client, project, monkeypatch):
    """With a key, GET serves the live Models API listing (incl. retired)."""
    _set_api_key(client, project)
    fake = FakeModelsApi(_catalog_pages(
        default=[{
            "data": [
                {"id": "claude-opus-4-6", "display_name": "Claude Opus 4.6",
                 "lifecycle": "active"},
                {"id": "claude-sonnet-4-5-20250929",
                 "display_name": "Claude Sonnet 4.5", "lifecycle": "active"},
                {"id": "claude-3-7-sonnet-latest",
                 "display_name": "Claude 3.7 Sonnet",
                 "lifecycle": "deprecated"},
            ],
            "has_more": False,
        }],
        retired=[{
            "data": [
                {"id": "claude-opus-4-1", "display_name": "Claude Opus 4.1",
                 "lifecycle": "retired"},
            ],
            "has_more": False,
        }],
    ))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _get_settings(client, project)
    assert r.status_code == 200
    body = r.get_json()
    assert body["models"] == [
        {"id": "claude-opus-4-6", "label": "Claude Opus 4.6",
         "lifecycle": "active"},
        {"id": "claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5",
         "lifecycle": "active"},
        {"id": "claude-3-7-sonnet-latest",
         "label": "Claude 3.7 Sonnet (deprecated)",
         "lifecycle": "deprecated"},
        {"id": "claude-opus-4-1", "label": "Claude Opus 4.1 (retired)",
         "lifecycle": "retired"},
    ]
    assert body["api_key"] is True  # still masked
    # Both listings requested via the repeated "lifecycle[]" query param
    # (a plain "lifecycle" param is rejected with a 400), full page size.
    kinds = [(c.get("extra_query") or {}).get("lifecycle[]")
             for c in fake.calls]
    assert kinds == [None, "retired"]
    assert all(c["limit"] == 1000 for c in fake.calls)


def test_catalog_paginates_when_has_more(client, project, monkeypatch):
    """The fetch follows after_id cursors until has_more is false."""
    _set_api_key(client, project)
    fake = FakeModelsApi(_catalog_pages(default=[
        {"data": [{"id": "m1", "display_name": "M1", "lifecycle": "active"}],
         "has_more": True, "last_id": "m1"},
        {"data": [{"id": "m2", "display_name": "M2", "lifecycle": "active"}],
         "has_more": False, "first_id": "m1"},
    ]))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _get_settings(client, project)
    assert r.status_code == 200
    assert [m["id"] for m in r.get_json()["models"]] == ["m1", "m2"]
    default_calls = [c for c in fake.calls
                     if (c.get("extra_query") or {}).get("lifecycle[]") is None]
    assert default_calls[0].get("after_id") is None
    assert default_calls[1].get("after_id") == "m1"


def test_catalog_falls_back_when_models_api_unreachable(client, project,
                                                        monkeypatch):
    """GET serves the fallback catalog when the Models API is down."""
    _set_api_key(client, project)

    def boom(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", boom)

    r = _get_settings(client, project)
    assert r.status_code == 200
    assert r.get_json()["models"] == FALLBACK_LLM_MODELS


def test_put_validation_uses_live_catalog(client, project, monkeypatch):
    """PUT validates the model id against the live Models API listing."""
    _set_api_key(client, project)
    fake = FakeModelsApi(_catalog_pages(default=[{
        "data": [{"id": "claude-sonnet-4-5-20250929",
                  "display_name": "Claude Sonnet 4.5", "lifecycle": "active"}],
        "has_more": False,
    }]))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _put_settings(client, project, {"model": "claude-nope"})
    assert r.status_code == 400
    assert _project_config(project)["llm"]["model"] == DEFAULT_LLM_MODEL

    assert _put_settings(
        client, project, {"model": "claude-sonnet-4-5-20250929"}
    ).status_code == 200
    assert _project_config(project)["llm"]["model"] == "claude-sonnet-4-5-20250929"


def test_put_unknown_model_accepted_without_key(client, project):
    """Without a key there is no live listing to validate against."""
    r = _put_settings(client, project, {"model": "claude-nope"})
    assert r.status_code == 200
    assert _project_config(project)["llm"]["model"] == "claude-nope"


def test_models_client_pins_anthropic_endpoint(monkeypatch):
    """The Models client ignores the dev-shell DeepSeek env routing."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-deepseek")
    captured = {}

    class SpyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", SpyClient)
    projects_mod._anthropic_models_client("sk-ant-test")
    assert captured == {
        "api_key": "sk-ant-test",
        "base_url": "https://api.anthropic.com",
        "timeout": projects_mod.LLM_MODELS_REQUEST_TIMEOUT,
    }


def test_retired_listing_failure_keeps_main_listing(client, project,
                                                    monkeypatch):
    """A failing retired listing must not hide the active+deprecated one."""
    _set_api_key(client, project)
    fake = FakeModelsApi(
        _catalog_pages(default=[{
            "data": [{"id": "m1", "display_name": "M1", "lifecycle": "active"}],
            "has_more": False,
        }]),
        fail_kinds=("retired",),
    )
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _get_settings(client, project)
    assert r.status_code == 200
    assert r.get_json()["models"] == [
        {"id": "m1", "label": "M1", "lifecycle": "active"},
    ]


def test_post_llm_models_uses_stored_key(client, project, monkeypatch):
    """POST /llm_models without a body key lists with the stored key."""
    _set_api_key(client, project)
    fake = FakeModelsApi(_catalog_pages(default=[{
        "data": [{"id": "m1", "display_name": "M1", "lifecycle": "active"}],
        "has_more": False,
    }]))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _post_models(client, project, {})
    assert r.status_code == 200
    assert r.get_json()["models"] == [
        {"id": "m1", "label": "M1", "lifecycle": "active"},
    ]
    assert fake.init_calls[0]["api_key"] == "sk-ant-test"


def test_post_llm_models_uses_typed_key(client, project, monkeypatch):
    """POST /llm_models lists with the body's key, which is not stored."""
    fake = FakeModelsApi(_catalog_pages(default=[{
        "data": [{"id": "m1", "display_name": "M1", "lifecycle": "active"}],
        "has_more": False,
    }]))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    r = _post_models(client, project, {"api_key": "sk-typed"})
    assert r.status_code == 200
    assert [m["id"] for m in r.get_json()["models"]] == ["m1"]
    assert fake.init_calls[0]["api_key"] == "sk-typed"
    assert not _project_config(project)["llm"].get("api_key")


def test_post_llm_models_blank_key_falls_back(client, project):
    """A blank body key means "no key": the fallback catalog is served."""
    r = _post_models(client, project, {"api_key": "  "})
    assert r.status_code == 200
    assert r.get_json()["models"] == FALLBACK_LLM_MODELS


def test_post_llm_models_without_any_key_falls_back(client, project):
    """No stored key and no body key: fallback catalog, no API call."""
    r = _post_models(client, project, {})
    assert r.status_code == 200
    assert r.get_json()["models"] == FALLBACK_LLM_MODELS


def test_put_validates_against_entered_key(client, project, monkeypatch):
    """PUT validates the model against the api_key entered in the same call."""
    fake = FakeModelsApi(_catalog_pages(default=[{
        "data": [{"id": "claude-sonnet-4-5-20250929",
                  "display_name": "Claude Sonnet 4.5", "lifecycle": "active"}],
        "has_more": False,
    }]))
    monkeypatch.setattr(projects_mod.anthropic, "Anthropic", fake)

    # No stored key: model + key arrive in the same PUT.
    r = _put_settings(client, project,
                      {"model": "claude-nope", "api_key": "sk-typed"})
    assert r.status_code == 400
    assert not _project_config(project)["llm"].get("api_key")

    assert _put_settings(
        client, project,
        {"model": "claude-sonnet-4-5-20250929", "api_key": "sk-typed"},
    ).status_code == 200
    cfg = _project_config(project)["llm"]
    assert cfg["model"] == "claude-sonnet-4-5-20250929"
    assert cfg["api_key"] == "sk-typed"
    assert fake.init_calls[0]["api_key"] == "sk-typed"


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
