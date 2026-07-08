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

"""Map LLM screening payloads to ASReview tag/list structures."""

import logging


def map_llm_payload_to_asreview(payload, tags_form, lists_form):
    """Map an LLM screening payload to ASReview tag/list structures.

    Converts export-name-based LLM output into the id-based structures
    ASReview's database layer consumes. The input is untrusted (it comes
    from a language model); malformed entries are dropped with a warning
    and the function never raises.

    Parameters
    ----------
    payload : dict or Any
        Parsed LLM JSON payload. Expected shape has ``labels`` and
        ``lists`` keys, but may be anything.
    tags_form : list[dict] or None
        Tag group definitions as produced by ``read_tags_data``. Provides
        the authoritative export->id mapping and ``free_text`` /
        ``single_select`` flags.
    lists_form : list[dict] or None
        List definitions as produced by ``read_lists_data``. Provides the
        authoritative name->list_id mapping.

    Returns
    -------
    dict
        ``{"tags": [...], "lists": [...]}`` always. On unusable input the
        lists are empty. Never raises.
    """
    tags_form = tags_form or []
    lists_form = lists_form or []

    # Build lookups from the authoritative forms.
    group_lookup = {}
    for group in tags_form:
        if not isinstance(group, dict):
            continue
        options = {}
        for v in group.get("values", []):
            if not isinstance(v, dict):
                continue
            options[v["export"]] = {
                "id": v["id"],
                "free_text": bool(v.get("free_text")),
            }
        group_lookup[group["export"]] = {
            "id": group["id"],
            "single_select": bool(group.get("single_select")),
            "options": options,
        }

    list_lookup = {}
    for lst in lists_form:
        if not isinstance(lst, dict):
            continue
        list_lookup[lst["name"]] = lst["id"]

    tags_out = _map_labels(payload, group_lookup)
    lists_out = _map_lists(payload, list_lookup)

    return {"tags": tags_out, "lists": lists_out}


def _map_labels(payload, group_lookup):
    """Map the ``labels`` portion of the payload."""
    if not isinstance(payload, dict):
        logging.warning("LLM payload is not a dict; no tags produced.")
        return []

    labels = payload.get("labels")
    if not isinstance(labels, list):
        logging.warning("LLM payload 'labels' is not a list; no tags produced.")
        return []

    tags_out = []

    for group_entry in labels:
        if not isinstance(group_entry, dict):
            logging.warning("Skipping non-dict label group entry: %r", group_entry)
            continue

        group_export = group_entry.get("group")
        if not group_export or group_export not in group_lookup:
            logging.warning(
                "Unknown or missing label group export: %r", group_export
            )
            continue

        group_info = group_lookup[group_export]
        values = group_entry.get("values")
        if not isinstance(values, list):
            logging.warning(
                "Label group '%s' values is not a list; skipping group.",
                group_export,
            )
            continue

        seen_option_ids = set()
        group_values_out = []

        for value in values:
            if not isinstance(value, dict):
                logging.warning(
                    "Skipping non-dict value in group '%s': %r",
                    group_export,
                    value,
                )
                continue

            option_export = value.get("checked")
            if not option_export:
                logging.warning(
                    "Value missing 'checked' in group '%s'; skipping.", group_export
                )
                continue

            if option_export not in group_info["options"]:
                logging.warning(
                    "Unknown option export '%s' in group '%s'; skipping.",
                    option_export,
                    group_export,
                )
                continue

            option_info = group_info["options"][option_export]
            option_id = option_info["id"]

            if option_id in seen_option_ids:
                logging.warning(
                    "Duplicate option '%s' in group '%s'; keeping first.",
                    option_export,
                    group_export,
                )
                continue
            seen_option_ids.add(option_id)

            text = None
            if option_info["free_text"]:
                raw_text = value.get("text")
                if isinstance(raw_text, str) and raw_text.strip():
                    text = raw_text

            group_values_out.append(
                {"id": option_id, "checked": True, "text": text}
            )

        # single_select enforcement
        if group_info["single_select"] and len(group_values_out) > 1:
            logging.warning(
                "single_select group '%s' has %d checked values; "
                "keeping only the first.",
                group_export,
                len(group_values_out),
            )
            group_values_out = group_values_out[:1]

        if group_values_out:
            tags_out.append(
                {"id": group_info["id"], "values": group_values_out}
            )

    return tags_out


def _map_lists(payload, list_lookup):
    """Map the ``lists`` portion of the payload."""
    if not isinstance(payload, dict):
        return []

    lists_raw = payload.get("lists")
    if not isinstance(lists_raw, list):
        logging.warning("LLM payload 'lists' is not a list; no lists produced.")
        return []

    lists_out = []

    for list_entry in lists_raw:
        if not isinstance(list_entry, dict):
            logging.warning("Skipping non-dict list entry: %r", list_entry)
            continue

        list_name = list_entry.get("list")
        if not list_name or list_name not in list_lookup:
            logging.warning("Unknown or missing list name: %r", list_name)
            continue

        list_id = list_lookup[list_name]
        items = list_entry.get("items")
        if not isinstance(items, list):
            logging.warning(
                "List '%s' items is not a list; skipping list.", list_name
            )
            continue

        seen_names = set()
        item_index = 0

        for item in items:
            name = str(item).strip()
            if not name:
                logging.warning(
                    "Empty item in list '%s'; dropping.", list_name
                )
                continue

            if "," in name or ";" in name:
                logging.warning(
                    "Item name contains ',' or ';' in list '%s': %r; dropping.",
                    list_name,
                    name,
                )
                continue

            if name in seen_names:
                logging.warning(
                    "Duplicate item '%s' in list '%s'; keeping first.",
                    name,
                    list_name,
                )
                continue
            seen_names.add(name)

            lists_out.append(
                {
                    "list_id": list_id,
                    "name": name,
                    "sorted_at": float(item_index),
                }
            )
            item_index += 1

    return lists_out
