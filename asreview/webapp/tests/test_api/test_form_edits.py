import json

import asreview as asr
import asreview.webapp.tests.utils.api_utils as au


def cfg(project):
    """Return (config_version, version) from the project's change-tracking state."""
    with asr.Project(project.project_path) as asr_project:
        state = asr_project.db.get_project_state()
    return state["config_version"], state["version"]


def test_tag_group_unchanged_save_no_bump(client, project):
    """PUTting exactly what GET returned does not bump config_version."""
    project_id = au.get_project_id(project)
    group = {
        "export": "g1",
        "label": "Group 1",
        "values": [
            {"export": "opt1", "label": "Option 1"},
            {"export": "opt2", "label": "Option 2"},
        ],
    }
    r = client.post(
        f"/api/projects/{project_id}/tags",
        data={"group": json.dumps(group)},
    )
    assert r.status_code == 200
    group_id = r.json[0]["id"]

    r = client.get(f"/api/projects/{project_id}/tags")
    got = r.json[0]

    before = cfg(project)
    r = client.put(
        f"/api/projects/{project_id}/tags/{group_id}",
        data={"group": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project) == before


def test_tag_group_label_change_bumps(client, project):
    """A new group label bumps config_version and is served back."""
    project_id = au.get_project_id(project)
    group = {"export": "g1", "label": "Group 1", "values": []}
    r = client.post(
        f"/api/projects/{project_id}/tags",
        data={"group": json.dumps(group)},
    )
    group_id = r.json[0]["id"]

    got = client.get(f"/api/projects/{project_id}/tags").json[0]
    before = cfg(project)

    got["label"] = "New label"
    r = client.put(
        f"/api/projects/{project_id}/tags/{group_id}",
        data={"group": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project)[0] > before[0]
    assert (
        client.get(f"/api/projects/{project_id}/tags").json[0]["label"] == "New label"
    )


def test_tag_option_change_bumps(client, project):
    """Changing one option's label bumps config_version; the other is untouched."""
    project_id = au.get_project_id(project)
    group = {
        "export": "g1",
        "label": "Group 1",
        "values": [
            {"export": "opt1", "label": "Option 1"},
            {"export": "opt2", "label": "Option 2"},
        ],
    }
    r = client.post(
        f"/api/projects/{project_id}/tags",
        data={"group": json.dumps(group)},
    )
    group_id = r.json[0]["id"]

    got = client.get(f"/api/projects/{project_id}/tags").json[0]
    before = cfg(project)

    got["values"][0]["label"] = "Changed option"
    r = client.put(
        f"/api/projects/{project_id}/tags/{group_id}",
        data={"group": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project)[0] > before[0]

    values = {
        v["id"]: v["label"]
        for v in client.get(f"/api/projects/{project_id}/tags").json[0]["values"]
    }
    assert values[got["values"][0]["id"]] == "Changed option"
    assert values[got["values"][1]["id"]] == "Option 2"


def test_tag_group_missing_sorted_at_keeps_stored(client, project):
    """Removing sorted_at from the PUT keeps the stored values; no bump."""
    project_id = au.get_project_id(project)
    group = {
        "export": "g1",
        "label": "Group 1",
        "values": [{"export": "opt1", "label": "Option 1", "sorted_at": 10.0}],
    }
    r = client.post(
        f"/api/projects/{project_id}/tags",
        data={"group": json.dumps(group)},
    )
    group_id = r.json[0]["id"]

    got = client.get(f"/api/projects/{project_id}/tags").json[0]
    original_group_sorted = got["sorted_at"]
    original_option_sorted = got["values"][0]["sorted_at"]

    got.pop("sorted_at", None)
    for value in got["values"]:
        value.pop("sorted_at", None)

    before = cfg(project)
    r = client.put(
        f"/api/projects/{project_id}/tags/{group_id}",
        data={"group": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project) == before

    updated = client.get(f"/api/projects/{project_id}/tags").json[0]
    assert updated["sorted_at"] == original_group_sorted
    assert updated["values"][0]["sorted_at"] == original_option_sorted


def test_tag_group_unknown_404(client, project):
    project_id = au.get_project_id(project)
    r = client.put(
        f"/api/projects/{project_id}/tags/nonexistent",
        data={"group": json.dumps({"export": "x", "label": "X", "values": []})},
    )
    assert r.status_code == 404


def test_list_unchanged_save_no_bump(client, project):
    """PUTting exactly what GET returned does not bump config_version."""
    project_id = au.get_project_id(project)
    r = client.post(
        f"/api/projects/{project_id}/lists",
        data={"list": json.dumps({"name": "Outcomes", "required_for_relevant": True})},
    )
    assert r.status_code == 200
    list_id = r.json[0]["id"]

    got = client.get(f"/api/projects/{project_id}/lists").json[0]

    before = cfg(project)
    r = client.put(
        f"/api/projects/{project_id}/lists/{list_id}",
        data={"list": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project) == before


def test_list_rename_bumps(client, project):
    """A new list name bumps config_version."""
    project_id = au.get_project_id(project)
    r = client.post(
        f"/api/projects/{project_id}/lists",
        data={"list": json.dumps({"name": "Outcomes", "required_for_relevant": True})},
    )
    list_id = r.json[0]["id"]

    got = client.get(f"/api/projects/{project_id}/lists").json[0]
    before = cfg(project)

    got["name"] = "Populations"
    r = client.put(
        f"/api/projects/{project_id}/lists/{list_id}",
        data={"list": json.dumps(got)},
    )
    assert r.status_code == 200
    assert cfg(project)[0] > before[0]


def test_list_unknown_404(client, project):
    project_id = au.get_project_id(project)
    r = client.put(
        f"/api/projects/{project_id}/lists/nonexistent",
        data={"list": json.dumps({"name": "X"})},
    )
    assert r.status_code == 404
