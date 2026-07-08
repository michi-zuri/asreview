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

"""Single-record LLM screening with injected Anthropic client."""

import base64
import json
import logging

from asreview.webapp._api.zotero import ZoteroLookupError


def _pdf_document_block(pdf_bytes):
    """Return a base64 ``document`` content block for the Anthropic API."""
    data = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": data,
        },
        "cache_control": {"type": "ephemeral"},
    }


def _extract_text(response):
    """Concatenate all text blocks from an Anthropic response."""
    return "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()


def _parse_json(text):
    """Parse a JSON object, tolerating ```-fences. Raises ValueError."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        idx = t.find("{")
        if idx != -1:
            t = t[idx:]
    try:
        return json.loads(t)
    except json.JSONDecodeError as err:
        raise ValueError(str(err)) from err


def screen_record(db, resolver, client, record, prompt, prompt_hash,
                  model, max_tokens=2048):
    """Run ONE LLM screening attempt for a record and record the result.

    Returns one of the status strings: "ready", "failed", "missing_pdf".
    Raises on transient API errors (the caller handles retry/backoff).

    Parameters
    ----------
    db : asreview.database.database.Database
        Open database (used for the 2a DAO methods).
    resolver : PdfResolver
        Resolves the record to a local PDF path.
    client : object
        An Anthropic-style client exposing
        client.messages.create(model=, max_tokens=, system=, messages=)
        and returning an object with .content (list of blocks with .type
        and .text) and .usage.input_tokens / .usage.output_tokens.
    record : object
        Record with .record_id and .attachment (passed to resolver).
    prompt : str
        The system prompt (from build_system_prompt).
    prompt_hash : str
        Hash to store the result under.
    model : str
        Model id, e.g. "claude-opus-4-8".
    """
    record_id = record.record_id

    # 1. Resolve PDF. Missing key OR failed download -> missing_pdf, no retry.
    try:
        pdf_path = resolver.resolve(record)
    except ZoteroLookupError as err:
        logging.warning("record %s PDF download failed: %s", record_id, err)
        db.mark_dispatch_missing_pdf(record_id)
        return "missing_pdf"
    if pdf_path is None:
        db.mark_dispatch_missing_pdf(record_id)
        return "missing_pdf"

    pdf_bytes = pdf_path.read_bytes()
    user_content = [
        _pdf_document_block(pdf_bytes),
        {"type": "text",
         "text": "Screen this article. Respond with JSON only."},
    ]

    # 2. First call (transient errors propagate to the caller).
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    text = _extract_text(resp)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)

    # 3. Parse; on failure do ONE repair call (reformat only, no PDF).
    try:
        payload = _parse_json(text)
    except ValueError:
        logging.warning("record %s bad JSON; attempting repair.", record_id)
        repair = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [{
                "type": "text",
                "text": (
                    "The following was supposed to be a single JSON "
                    "object but is malformed. Return ONLY the corrected "
                    "JSON object, with no prose and no code fences:\n\n"
                    + text
                ),
            }]}],
        )
        try:
            payload = _parse_json(_extract_text(repair))
        except ValueError as err:
            logging.warning("record %s JSON unrepairable: %s", record_id, err)
            db.mark_dispatch_failed(record_id, f"invalid JSON: {err}")
            return "failed"

    # 4. Success -> store and mark ready.
    db.store_llm_result(
        record_id, prompt_hash, model, json.dumps(payload),
        input_tokens=in_tok, output_tokens=out_tok,
    )
    return "ready"
