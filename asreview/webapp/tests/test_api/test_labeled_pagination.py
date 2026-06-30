import pandas as pd

import asreview.webapp.tests.utils.api_utils as au
from asreview.webapp._api.projects import (
    _decode_cursor,
    _encode_cursor,
    _flatten_tags,
    _labeled_filter_signature,
    _record_tags_invalid,
    _tag_is_checked,
)


def _labeled(client, project, **params):
    return client.get(
        f"/api/projects/{au.get_project_id(project)}/labeled", query_string=params
    )


def test_cursor_pagination_completeness(client, project):
    # label several distinct records
    res = au.search_project_data(client, project, query="The&n_max=10")
    ids = [r["record_id"] for r in res.json["result"]][:5]
    assert len(ids) == 5
    for rid in ids:
        au.label_project_record(client, project, rid, label=1)

    # full page
    full = _labeled(client, project).json
    assert len(full["result"]) == 5
    assert full["next_cursor"] is None  # nothing more to load

    # paginate with per_page=2 via cursor
    seen = []
    cursor = None
    pages = 0
    while True:
        params = {"per_page": 2}
        if cursor:
            params["cursor"] = cursor
        page = _labeled(client, project, **params).json
        pages += 1
        for r in page["result"]:
            assert r["record_id"] not in seen, "duplicate across pages"
            seen.append(r["record_id"])
        cursor = page["next_cursor"]
        if not cursor:
            break
        assert pages < 20, "pagination did not terminate"

    assert sorted(seen) == sorted(ids)
    assert pages == 3  # 2 + 2 + 1


def test_subset_filter_via_keyset(client, project):
    res = au.search_project_data(client, project, query="The&n_max=10")
    ids = [r["record_id"] for r in res.json["result"]][:4]
    au.label_project_record(client, project, ids[0], label=1)
    au.label_project_record(client, project, ids[1], label=1)
    au.label_project_record(client, project, ids[2], label=0)
    au.label_project_record(client, project, ids[3], label=0)

    rel = _labeled(client, project, subset="relevant").json
    assert {r["record_id"] for r in rel["result"]} == {ids[0], ids[1]}
    irr = _labeled(client, project, subset="irrelevant").json
    assert {r["record_id"] for r in irr["result"]} == {ids[2], ids[3]}


def test_invalid_cursor_returns_400(client, project):
    r = _labeled(client, project, cursor="not-a-valid-cursor")
    assert r.status_code == 400


def test_cursor_filter_mismatch_rejected(client, project):
    # a cursor minted for one filter set must not be accepted for another
    sig = _labeled_filter_signature("all", [], True)
    cur = _encode_cursor(123.0, 5, sig)
    r = _labeled(client, project, cursor=cur, subset="relevant")
    assert r.status_code == 400


def test_tag_helpers():
    saved = [
        {
            "id": 0,
            "export": "grp",
            "single_select": True,
            "values": [
                {"id": 0, "export": "a", "checked": True},
                {"id": 1, "export": "b", "checked": True},
            ],
        },
        {
            "id": 1,
            "export": "multi",
            "values": [
                {"id": 0, "export": "x", "checked": False},
                {"id": 1, "export": "y", "checked": True},
            ],
        },
    ]
    assert _tag_is_checked(saved, "grp", "a") is True
    assert _tag_is_checked(saved, "multi", "x") is False
    assert _tag_is_checked(saved, "multi", "y") is True
    assert _tag_is_checked(saved, "nope", "a") is False
    assert _tag_is_checked(None, "grp", "a") is False

    tags_form = [
        {"id": 0, "export": "grp", "single_select": True, "values": []},
        {"id": 1, "export": "multi", "values": []},
    ]
    # grp is single_select with 2 checked -> invalid
    assert _record_tags_invalid(saved, tags_form) is True
    # if grp had only one checked -> valid
    saved_valid = [
        {
            "id": 0,
            "export": "grp",
            "single_select": True,
            "values": [
                {"id": 0, "export": "a", "checked": True},
                {"id": 1, "export": "b", "checked": False},
            ],
        }
    ]
    assert _record_tags_invalid(saved_valid, tags_form) is False
    assert _record_tags_invalid([], tags_form) is False

    # A required group with no checked value is invalid (missing selection),
    # even when no group is single_select.
    required_form = [
        {"id": 0, "export": "grp", "required": True, "values": []},
    ]
    saved_missing = [
        {
            "id": 0,
            "export": "grp",
            "values": [
                {"id": 0, "export": "a", "checked": False},
                {"id": 1, "export": "b", "checked": False},
            ],
        }
    ]
    assert _record_tags_invalid(saved_missing, required_form) is True
    # The group not being present at all also counts as missing.
    assert _record_tags_invalid([], required_form) is True
    # A single checked value satisfies the requirement.
    saved_required_ok = [
        {
            "id": 0,
            "export": "grp",
            "values": [
                {"id": 0, "export": "a", "checked": True},
                {"id": 1, "export": "b", "checked": False},
            ],
        }
    ]
    assert _record_tags_invalid(saved_required_ok, required_form) is False


def test_flatten_tags_free_text_only_when_checked():
    tags_config = [
        {
            "export": "grp",
            "values": [{"export": "a"}, {"export": "b"}],
        }
    ]
    results = pd.DataFrame(
        {
            "tags": [
                # checked with text -> text exported
                [
                    {
                        "export": "grp",
                        "values": [
                            {"export": "a", "checked": True, "text": "keep me"},
                            {"export": "b", "checked": False},
                        ],
                    }
                ],
                # deselected but a stale note remains in storage -> text dropped
                [
                    {
                        "export": "grp",
                        "values": [
                            {"export": "a", "checked": False, "text": "drop me"},
                            {"export": "b", "checked": True, "text": "kept b"},
                        ],
                    }
                ],
            ]
        }
    )

    out = _flatten_tags(results, tags_config)

    # Selection columns always present.
    assert out.loc[0, "tag_grp_a"] == 1
    assert out.loc[1, "tag_grp_a"] == 0

    # Row 0: a is checked with text -> exported; b has no text.
    assert out.loc[0, "tag_grp_a_text"] == "keep me"

    # Row 1: a deselected -> its stale text is not exported (NA), b checked -> kept.
    assert pd.isna(out.loc[1, "tag_grp_a_text"])
    assert out.loc[1, "tag_grp_b_text"] == "kept b"


def test_cursor_roundtrip():
    sig = _labeled_filter_signature("relevant", ["has_note"], True)
    assert _decode_cursor(None, sig) is None
    enc = _encode_cursor(1712345678.123456, 42, sig)
    assert _decode_cursor(enc, sig) == (1712345678.123456, 42)
    enc_null = _encode_cursor(None, 7, sig)
    assert _decode_cursor(enc_null, sig) == (None, 7)
