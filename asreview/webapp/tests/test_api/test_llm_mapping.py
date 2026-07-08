import pytest

from asreview.webapp._api.llm_mapping import map_llm_payload_to_asreview

TAGS = [
    {
        "id": 10,
        "export": "study_type",
        "single_select": True,
        "values": [
            {"id": 100, "export": "case_report", "free_text": False},
            {"id": 101, "export": "case_series", "free_text": False},
            {"id": 102, "export": "other", "free_text": True},
        ],
    },
    {
        "id": 20,
        "export": "inclusion_criteria",
        "single_select": False,
        "values": [
            {"id": 200, "export": "infectious_agent", "free_text": False},
            {"id": 201, "export": "mutation", "free_text": False},
        ],
    },
]

LISTS = [
    {"id": "L1", "name": "case_identifiers"},
    {"id": "L2", "name": "issues"},
]


def test_happy_path():
    """Labels with valid picks + list items produce exact output."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": [{"checked": "case_report"}],
            },
            {
                "group": "inclusion_criteria",
                "values": [
                    {"checked": "infectious_agent"},
                    {"checked": "mutation"},
                ],
            },
        ],
        "lists": [
            {"list": "case_identifiers", "items": ["Patient 1", "Case A"]},
        ],
    }

    result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert result["tags"] == [
        {
            "id": 10,
            "values": [
                {"id": 100, "checked": True, "text": None},
            ],
        },
        {
            "id": 20,
            "values": [
                {"id": 200, "checked": True, "text": None},
                {"id": 201, "checked": True, "text": None},
            ],
        },
    ]
    assert result["lists"] == [
        {"list_id": "L1", "name": "Patient 1", "sorted_at": 0.0},
        {"list_id": "L1", "name": "Case A", "sorted_at": 1.0},
    ]


def test_free_text():
    """Free-text option carries text; non-free_text option ignores bogus text."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": [
                    {"checked": "other", "text": "HIV"},
                ],
            },
            {
                "group": "inclusion_criteria",
                "values": [
                    {"checked": "infectious_agent", "text": "should be ignored"},
                ],
            },
        ],
    }

    result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    # study_type group: "other" (free_text=True) with text
    st = next(g for g in result["tags"] if g["id"] == 10)
    other = next(v for v in st["values"] if v["id"] == 102)
    assert other["text"] == "HIV"

    # inclusion_criteria group: "infectious_agent" (free_text=False) — text must be None
    ic = next(g for g in result["tags"] if g["id"] == 20)
    agent = next(v for v in ic["values"] if v["id"] == 200)
    assert agent["text"] is None


def test_unknown_group_export(caplog):
    """Unknown group export is dropped + warning logged."""
    payload = {
        "labels": [
            {"group": "nonexistent", "values": [{"checked": "case_report"}]},
            {"group": "study_type", "values": [{"checked": "case_report"}]},
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["tags"]) == 1
    assert result["tags"][0]["id"] == 10
    assert "nonexistent" in caplog.text


def test_unknown_option_export(caplog):
    """Unknown option export within a known group -> dropped + warning."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": [
                    {"checked": "unknown_option"},
                    {"checked": "case_report"},
                ],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["tags"]) == 1
    values = result["tags"][0]["values"]
    assert len(values) == 1
    assert values[0]["id"] == 100
    assert "unknown_option" in caplog.text


def test_single_select_too_many(caplog):
    """single_select group with two checked options keeps only the first."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": [
                    {"checked": "case_report"},
                    {"checked": "case_series"},
                ],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["tags"]) == 1
    values = result["tags"][0]["values"]
    assert len(values) == 1
    assert values[0]["id"] == 100  # first one kept
    assert "single_select" in caplog.text.lower() or "keeping only the first" in caplog.text.lower()


def test_duplicate_option_deduped(caplog):
    """Duplicate option in same group -> deduped to one with warning."""
    payload = {
        "labels": [
            {
                "group": "inclusion_criteria",
                "values": [
                    {"checked": "infectious_agent"},
                    {"checked": "infectious_agent"},
                    {"checked": "mutation"},
                ],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    values = result["tags"][0]["values"]
    assert len(values) == 2
    ids = [v["id"] for v in values]
    assert ids.count(200) == 1
    assert "duplicate" in caplog.text.lower()


def test_group_with_zero_valid_values_omitted(caplog):
    """Group where all values are invalid -> omitted from output entirely."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": [
                    {"checked": "bogus1"},
                    {"checked": "bogus2"},
                ],
            },
            {
                "group": "inclusion_criteria",
                "values": [{"checked": "infectious_agent"}],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["tags"]) == 1
    assert result["tags"][0]["id"] == 20


def test_unknown_list_name(caplog):
    """Unknown list name -> dropped + warning."""
    payload = {
        "lists": [
            {"list": "unknown_list", "items": ["A"]},
            {"list": "case_identifiers", "items": ["B"]},
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["lists"]) == 1
    assert result["lists"][0]["name"] == "B"
    assert "unknown_list" in caplog.text


def test_list_item_dedupe():
    """Duplicate names within a list -> deduped, keeping first."""
    payload = {
        "lists": [
            {"list": "case_identifiers", "items": ["A", "A", "B"]},
        ],
    }

    result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    names = [item["name"] for item in result["lists"]]
    assert names == ["A", "B"]
    # sorted_at should be 0.0 and 1.0
    assert result["lists"][0]["sorted_at"] == 0.0
    assert result["lists"][1]["sorted_at"] == 1.0


def test_list_item_cleaning(caplog):
    """Empty strings, whitespace, commas, semicolons all dropped with warnings."""
    payload = {
        "lists": [
            {
                "list": "case_identifiers",
                "items": [
                    "",
                    "   ",
                    "bad,comma",
                    "bad;semicolon",
                    "good_item",
                    123,  # non-string coerced
                ],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    names = [item["name"] for item in result["lists"]]
    assert "good_item" in names
    assert "123" in names  # str(123).strip() = "123"
    assert "bad,comma" not in names
    assert "bad;semicolon" not in names
    assert "" not in names
    assert "   " not in names
    assert "comma" in caplog.text or "',' or ';'" in caplog.text


def test_payload_none():
    """payload=None returns empty structures, no crash."""
    result = map_llm_payload_to_asreview(None, TAGS, LISTS)
    assert result == {"tags": [], "lists": []}


def test_payload_string():
    """payload='nonsense' returns empty structures, no crash."""
    result = map_llm_payload_to_asreview("nonsense", TAGS, LISTS)
    assert result == {"tags": [], "lists": []}


def test_payload_empty_dict():
    """payload={} returns empty structures, no crash."""
    result = map_llm_payload_to_asreview({}, TAGS, LISTS)
    assert result == {"tags": [], "lists": []}


def test_payload_malformed_labels_and_lists(caplog):
    """Non-list 'labels' and 'lists' produce empty tags/lists with warnings."""
    payload = {"labels": "x", "lists": 5}

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert result == {"tags": [], "lists": []}
    assert "'labels' is not a list" in caplog.text
    assert "'lists' is not a list" in caplog.text


def test_group_entry_not_dict(caplog):
    """A group entry that is a list (not a dict) is skipped."""
    payload = {
        "labels": [
            ["not a dict"],
            {"group": "study_type", "values": [{"checked": "case_report"}]},
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    assert len(result["tags"]) == 1
    assert "non-dict label group" in caplog.text


def test_value_not_dict(caplog):
    """A value entry that is a string (not a dict) is skipped."""
    payload = {
        "labels": [
            {
                "group": "study_type",
                "values": ["not a dict", {"checked": "case_report"}],
            },
        ],
    }

    with caplog.at_level("WARNING"):
        result = map_llm_payload_to_asreview(payload, TAGS, LISTS)

    values = result["tags"][0]["values"]
    assert len(values) == 1
    assert values[0]["id"] == 100
    assert "non-dict value" in caplog.text


def test_tags_form_none():
    """tags_form=None treated as empty list."""
    payload = {
        "labels": [
            {"group": "study_type", "values": [{"checked": "case_report"}]},
        ],
    }
    result = map_llm_payload_to_asreview(payload, None, LISTS)
    assert result["tags"] == []


def test_lists_form_none():
    """lists_form=None treated as empty list."""
    payload = {
        "lists": [
            {"list": "case_identifiers", "items": ["A"]},
        ],
    }
    result = map_llm_payload_to_asreview(payload, TAGS, None)
    assert result["lists"] == []


# --- build_prefill_state tests ---


def test_build_prefill_state_tags_pass_through():
    """Tags from mapper pass through unchanged."""
    from asreview.webapp._api.llm_mapping import build_prefill_state

    payload = {"labels": [], "lists": []}
    tags_form = [
        {
            "id": "g1", "export": "quality", "label": "Quality",
            "single_select": True, "values": [
                {"id": "o1", "export": "high", "label": "High",
                 "free_text": False},
            ],
        },
    ]
    lists_form = []
    counter = [0]

    def fake_uuid():
        counter[0] += 1
        return f"item-{counter[0]}"

    result = build_prefill_state(payload, tags_form, lists_form,
                                 uuid_fn=fake_uuid)
    assert result == {"tags": [], "lists": []}


def test_build_prefill_state_lists_gain_item_id():
    """Each list item gets a deterministic string item_id."""
    from asreview.webapp._api.llm_mapping import build_prefill_state

    payload = {
        "labels": [],
        "lists": [
            {"list": "issues", "items": ["bug", "feature"]},
        ],
    }
    tags_form = []
    lists_form = [
        {"id": "l1", "name": "issues", "input_helper_text": ""},
    ]
    counter = [0]

    def fake_uuid():
        counter[0] += 1
        return f"item-{counter[0]}"

    result = build_prefill_state(payload, tags_form, lists_form,
                                 uuid_fn=fake_uuid)
    assert result["tags"] == []
    assert result["lists"] == [
        {"list_id": "l1", "name": "bug", "sorted_at": 0.0,
         "item_id": "item-1"},
        {"list_id": "l1", "name": "feature", "sorted_at": 1.0,
         "item_id": "item-2"},
    ]


def test_build_prefill_state_tags_passthrough_with_lists():
    """Tags from mapper are included alongside lists."""
    from asreview.webapp._api.llm_mapping import build_prefill_state

    payload = {
        "labels": [
            {"group": "quality", "values": [{"checked": "high"}]},
        ],
        "lists": [],
    }
    tags_form = [
        {
            "id": "g1", "export": "quality", "label": "Quality",
            "single_select": False, "values": [
                {"id": "o1", "export": "high", "label": "High",
                 "free_text": False},
            ],
        },
    ]
    lists_form = []

    result = build_prefill_state(payload, tags_form, lists_form)
    assert result["tags"] == [
        {"id": "g1", "values": [
            {"id": "o1", "checked": True, "text": None},
        ]},
    ]
    assert result["lists"] == []
