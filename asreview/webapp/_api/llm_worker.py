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
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

import asreview as asr
from asreview.database.database import open_db
from asreview.webapp._api.llm_prompt import build_system_prompt
from asreview.webapp._api.pdf_resolver import PdfResolver
from asreview.webapp._api.utils import read_tags_data, read_lists_data
from asreview.webapp._api.zotero import ZoteroLookupError
from asreview.webapp.utils import asreview_path


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


# Transient HTTP status codes that warrant a retry.
_TRANSIENT_STATUS = {429, 500, 502, 503, 529}


def _default_is_transient(exc):
    """Return True for transient Anthropic API errors."""
    if isinstance(exc, (anthropic.APIConnectionError,
                        anthropic.RateLimitError,
                        anthropic.InternalServerError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return getattr(exc, "status_code", None) in _TRANSIENT_STATUS
    return False


def process_with_retry(db, resolver, client, record, prompt, prompt_hash,
                       model, max_attempts=5, base_delay=1.0,
                       max_delay=60.0, is_transient=None,
                       sleep=time.sleep, jitter=random.random):
    """Run screen_record with exponential backoff + jitter on transient errors.

    Returns the final status string. On a non-transient error, or after
    max_attempts transient errors, marks the dispatch row failed and returns
    "failed".
    """
    is_transient = is_transient or _default_is_transient
    record_id = record.record_id
    attempt = 0
    while True:
        try:
            start = time.time()
            status = screen_record(db, resolver, client, record, prompt,
                                   prompt_hash, model)
            logging.info(
                "llm screen record_id=%s status=%s latency=%.2fs",
                record_id, status, time.time() - start,
            )
            return status
        except Exception as err:
            attempt += 1
            db.increment_dispatch_attempts(record_id)
            if not is_transient(err) or attempt >= max_attempts:
                logging.warning(
                    "record %s failed after %d attempt(s): %s",
                    record_id, attempt, err,
                )
                db.mark_dispatch_failed(record_id, err)
                return "failed"
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) \
                + jitter()
            logging.info(
                "record %s transient error (attempt %d): %s; retrying in "
                "%.1fs", record_id, attempt, err, delay,
            )
            sleep(delay)


def run_worker_once(project, model, max_concurrent=3, max_attempts=5,
                    db_factory=None, resolver_factory=None,
                    executor=None):
    """Claim and process all currently-queued dispatch rows once.

    Returns the number of records processed.  Builds the current system
    prompt from the project's tags/lists + stored criteria text.  Each
    pooled job opens its own Database (db_factory) so sqlite is never
    shared across threads.

    When *executor* is given it is used to schedule jobs (caller owns the
    pool lifetime); otherwise a fresh ThreadPoolExecutor is created.
    """
    from asreview.webapp._api.projects import _llm_settings

    settings = _llm_settings(project)

    # Resolve API key: per-project setting first, env var as fallback.
    api_key = settings.get("api_key", "") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    )
    if not api_key:
        logging.warning(
            "LLM worker: skipping project %s — no API key configured "
            "(set ANTHROPIC_API_KEY env var or configure per project)",
            project.project_path,
        )
        return 0
    client = anthropic.Anthropic(api_key=api_key)

    criteria_text = settings.get("criteria_text", "")
    tags = read_tags_data(project.db)
    lists = read_lists_data(project)
    prompt, prompt_hash = build_system_prompt(tags, lists, criteria_text)

    db_factory = db_factory or (lambda: open_db(project.db_path))
    resolver_factory = resolver_factory or (
        lambda: PdfResolver(project.project_path)
    )

    claimed = []
    while True:
        rid = project.db.claim_next_queued_dispatch()
        if rid is None:
            break
        claimed.append(rid)
    if not claimed:
        return 0

    def _job(record_id):
        job_db = db_factory()
        try:
            resolver = resolver_factory()
            record = job_db.input.get_records(record_id)
            return process_with_retry(
                job_db, resolver, client, record, prompt, prompt_hash,
                model, max_attempts=max_attempts,
            )
        finally:
            job_db.close()

    if executor is None:
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            list(pool.map(_job, claimed))
    else:
        list(executor.map(_job, claimed))
    return len(claimed)


def run_worker(project_path, model=None, max_concurrent=None,
               poll_interval=5.0):
    """Continuously drain a project's LLM queue (single-project mode)."""
    model = model or os.environ.get("ASREVIEW_LLM_MODEL", "claude-opus-4-8")
    if max_concurrent is None:
        max_concurrent = int(
            os.environ.get("ASREVIEW_LLM_MAX_CONCURRENT", "3")
        )
    with asr.Project(project_path) as project:
        while True:
            n = run_worker_once(
                project, model,
                max_concurrent=max_concurrent,
            )
            if n == 0:
                time.sleep(poll_interval)


def main():
    """CLI entrypoint for the LLM screening worker."""
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="ASReview LLM screening worker")
    ap.add_argument("project_path")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-concurrent", type=int, default=None)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()
    run_worker(args.project_path, model=args.model,
               max_concurrent=args.max_concurrent,
               poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()


def discover_project_paths():
    """Return sorted v4 project directories under asreview_path().

    A project directory is any subdirectory containing a project.json whose
    ``project_file_version`` (or detected version) equals the current
    ``Project.VERSION``. Older formats are silently skipped — they need to
    be upgraded first with ``asreview migrate --projects``.
    """
    base = asreview_path()
    paths = []
    for p in sorted(base.glob("*")):
        if p.is_dir() and (p / asr.Project.PATH_CONFIG).exists():
            if asr.is_project(p):
                paths.append(p)
            else:
                logging.debug("LLM worker: skipping %s — not a v%s project",
                              p.name, asr.Project.VERSION)
    return paths


def run_worker_all(model, executor=None, max_concurrent=3, max_attempts=5):
    """Drain the queued dispatch rows of every project once.

    Returns the total number of records processed across all projects. A
    failure opening/processing one project is logged and skipped so one
    bad project cannot kill the service.
    """
    total = 0
    for path in discover_project_paths():
        try:
            with asr.Project(path) as project:
                total += run_worker_once(
                    project, model,
                    max_concurrent=max_concurrent,
                    max_attempts=max_attempts,
                    executor=executor,
                )
        except Exception as err:  # noqa: BLE001
            logging.exception("LLM worker: project %s failed: %s", path, err)
    return total


def run_worker_service(model=None, max_concurrent=None, poll_interval=5.0):
    """Run the all-projects worker forever with a single global pool.

    The one ThreadPoolExecutor(max_concurrent) shared across all projects
    is what makes ``max_concurrent`` a GLOBAL cap. Only drains; never tops
    up. Each project's Anthropic API key and screening criteria are read
    from its stored LLM settings (per-project config), falling back to the
    ``ANTHROPIC_API_KEY`` environment variable when no key is configured.
    """
    model = model or os.environ.get("ASREVIEW_LLM_MODEL", "claude-opus-4-8")
    if max_concurrent is None:
        max_concurrent = int(
            os.environ.get("ASREVIEW_LLM_MAX_CONCURRENT", "3")
        )
    logging.info(
        "ASReview LLM worker starting (model=%s, max_concurrent=%s)",
        model, max_concurrent,
    )
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        while True:
            n = run_worker_all(
                model, executor=executor,
                max_concurrent=max_concurrent,
            )
            if n == 0:
                time.sleep(poll_interval)
