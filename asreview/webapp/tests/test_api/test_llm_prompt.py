import hashlib

from asreview.webapp._api.llm_prompt import build_system_prompt

TAGS = [
    {
        "id": 0,
        "export": "exclusion_reasons",
        "label": "Exclusion reasons",
        "required_relevant": False,
        "required_irrelevant": True,
        "require_all": False,
        "single_select": False,
        "input_helper_text": "",
        "sorted_at": 0,
        "values": [
            {"id": 0, "export": "no_pdf", "label": "no pdf", "free_text": False},
            {"id": 1, "export": "other", "label": "other", "free_text": True},
        ],
    },
    {
        "id": 1,
        "export": "inclusion_criteria",
        "label": "Inclusion criteria",
        "required_relevant": True,
        "required_irrelevant": False,
        "require_all": True,
        "single_select": False,
        "input_helper_text": "",
        "sorted_at": 1,
        "values": [
            {
                "id": 0,
                "export": "infectious_agent",
                "label": "mentions an agent",
                "free_text": False,
            },
        ],
    },
]

LISTS = [
    {
        "id": "a",
        "name": "case_identifiers",
        "required_for_relevant": False,
        "input_helper_text": "IDs of matching cases",
        "sorted_at": 0,
    },
]


def test_return_type():
    """build_system_prompt returns a 2-tuple of strings."""
    prompt, prompt_hash = build_system_prompt(TAGS, LISTS, "Goal text")
    assert isinstance(prompt, str)
    assert isinstance(prompt_hash, str)


def test_hash_derived_from_prompt():
    """prompt_hash is the SHA-256 hex digest of the prompt."""
    prompt, prompt_hash = build_system_prompt(TAGS, LISTS, "Goal text")
    expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert prompt_hash == expected
    assert len(prompt_hash) == 64


def test_contract_present():
    """Prompt contains all group exports, option exports, list names, and instructions."""
    prompt, _ = build_system_prompt(TAGS, LISTS, "Goal text")
    assert "exclusion_reasons" in prompt
    assert "inclusion_criteria" in prompt
    assert "no_pdf" in prompt
    assert "other" in prompt
    assert "infectious_agent" in prompt
    assert "case_identifiers" in prompt
    assert "labels" in prompt
    assert "lists" in prompt
    assert "Use export identifiers only." in prompt


def test_criteria_included():
    """Criteria text appears verbatim in the prompt."""
    prompt, _ = build_system_prompt(TAGS, LISTS, "Goal text")
    assert "Goal text" in prompt


def test_criteria_absent():
    """None and whitespace-only criteria both produce 'None provided.'."""
    prompt_none, _ = build_system_prompt(TAGS, LISTS, None)
    prompt_spaces, _ = build_system_prompt(TAGS, LISTS, "   ")
    assert "None provided." in prompt_none
    assert "None provided." in prompt_spaces


def test_determinism():
    """Same inputs yield identical prompt and hash."""
    p1, h1 = build_system_prompt(TAGS, LISTS, "Goal text")
    p2, h2 = build_system_prompt(TAGS, LISTS, "Goal text")
    assert p1 == p2
    assert h1 == h2


def test_hash_sensitivity():
    """Different criteria or different tag options yield different hashes."""
    _, h1 = build_system_prompt(TAGS, LISTS, "Goal A")
    _, h2 = build_system_prompt(TAGS, LISTS, "Goal B")
    assert h1 != h2

    tags_modified = [
        {
            **TAGS[0],
            "values": TAGS[0]["values"]
            + [
                {
                    "id": 2,
                    "export": "extra_option",
                    "label": "extra",
                    "free_text": False,
                }
            ],
        },
        TAGS[1],
    ]
    _, h3 = build_system_prompt(tags_modified, LISTS, "Goal A")
    assert h1 != h3


def test_none_forms():
    """None tags_form and lists_form don't crash; prompt contains empty JSON arrays."""
    prompt, _ = build_system_prompt(None, None, "x")
    assert "[]" in prompt
