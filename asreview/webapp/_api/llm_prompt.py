# Copyright 2019-2025 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the system prompt for LLM-assisted full-text screening."""

import hashlib
import json

PROMPT_TEMPLATE = """You are assisting with full-text screening for a \
systematic review.

Read the attached PDF and PRE-FILL a screening form for a human reviewer. You \
do NOT make the final decision; a human always confirms or corrects your \
answers. Fill the form as accurately as a careful human reviewer would.

Respond with ONLY a single JSON object. No prose, and no markdown code fences.

## Response shape
{{
  "labels": [
    {{"group": "<group_export>", "values": [
      {{"checked": "<option_export>"}},
      {{"checked": "<option_export>", "text": "<free text, only for options that allow it>"}}
    ]}}
  ],
  "lists": [
    {{"list": "<list_name>", "items": ["<string>"]}}
  ]
}}

Rules:
- Use ONLY the exact `group` export names, `checked` option export names, and
  `list` names listed below. Do not invent new ones.
- Use export identifiers only. Never output numeric ids or human labels.
- Include only the options you are checking; omit unchecked options.
- A `single_select` group may have at most one checked value.
- A `require_all` group is a checklist: check its options only when the article
  satisfies all of them; otherwise leave that group unsatisfied.
- Add `text` only for options whose `free_text` is true.
- For each list include only the items that apply; use an empty list when none.

## Label groups (BINDING CONTRACT)
{label_groups_json}

## Lists (BINDING CONTRACT)
{lists_json}

## Additional context (non-binding)
{criteria_block}
"""


def build_system_prompt(tags_form, lists_form, criteria_text=None):
    """Build the system prompt for full-text screening and its SHA-256 hash.

    Parameters
    ----------
    tags_form : list[dict] or None
        Tag group definitions as produced by ``read_tags_data``. Already in
        display order.
    lists_form : list[dict] or None
        List definitions as produced by ``read_lists_data``. Already in
        display order.
    criteria_text : str or None
        An optional human-facing description of the screening goal. Non-binding
        context for the LLM.

    Returns
    -------
    tuple[str, str]
        A 2-tuple of ``(prompt, prompt_hash)`` where ``prompt`` is the
        assembled system prompt and ``prompt_hash`` is its SHA-256 hex digest.
    """
    tags_form = tags_form or []
    lists_form = lists_form or []

    projected_groups = []
    for group in tags_form:
        projected_groups.append(
            {
                "group": group["export"],
                "label": group["label"],
                "single_select": bool(group.get("single_select")),
                "require_all": bool(group.get("require_all")),
                "required_relevant": bool(group.get("required_relevant")),
                "required_irrelevant": bool(group.get("required_irrelevant")),
                "options": [
                    {
                        "export": v["export"],
                        "label": v["label"],
                        "free_text": bool(v.get("free_text")),
                    }
                    for v in group.get("values", [])
                ],
            }
        )

    label_groups_json = json.dumps(projected_groups, indent=2, ensure_ascii=False)

    projected_lists = []
    for lst in lists_form:
        projected_lists.append(
            {
                "list": lst["name"],
                "description": lst.get("input_helper_text", ""),
                "required_for_relevant": bool(lst.get("required_for_relevant")),
            }
        )

    lists_json = json.dumps(projected_lists, indent=2, ensure_ascii=False)

    if criteria_text and criteria_text.strip():
        criteria_block = criteria_text
    else:
        criteria_block = "None provided."

    prompt = PROMPT_TEMPLATE.format(
        label_groups_json=label_groups_json,
        lists_json=lists_json,
        criteria_block=criteria_block,
    )

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    return prompt, prompt_hash
