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

"""Background Zotero full-text sync for all projects.

Walks every project under the ASReview path and refreshes the cached Zotero
attachment lookup of records that do not have an attachment key yet. Records
whose last failed lookup is younger than 24 hours are skipped, so Zotero is
re-queried at most once per day per record. A PDF added to Zotero directly is
therefore picked up within a day even if the record was never shown on screen.

The sync runs as a daemon thread inside the standalone LLM worker service
(see ``run_worker_service`` in llm_worker.py) — never inside Flask.
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd

import asreview as asr
from asreview.database.database import open_db
from asreview.webapp._api.zotero import BACKGROUND_RECHECK_INTERVAL
from asreview.webapp._api.zotero import ZoteroLookupError
from asreview.webapp._api.zotero import attachment_check_due
from asreview.webapp._api.zotero import get_zotero_config
from asreview.webapp._api.zotero import is_attachment_key
from asreview.webapp._api.zotero import refresh_attachment


def records_due_for_check(rows, max_age, now=None):
    """Return record_ids whose attachment lookup is due, in check order.

    Parameters
    ----------
    rows : iterable of (record_id, original_id, attachment, duplicate_of)
        One tuple per record. Missing values may be None, NaN or pd.NA.
    max_age : float
        Seconds after a failed lookup before it may be repeated.
    now : datetime, optional
        Current time (UTC), passed to ``attachment_check_due``.

    Returns
    -------
    list
        Record ids that need a lookup: never-checked records first (by
        record_id), then records with a stale failed lookup, oldest first.
    """
    never_checked = []
    checked = []  # (last_checked, record_id)

    for record_id, original_id, attachment, duplicate_of in rows:
        if pd.isna(original_id) or original_id == "":
            continue
        if not pd.isna(duplicate_of):
            continue

        if pd.isna(attachment) or attachment == "":
            never_checked.append(record_id)
            continue
        if not attachment_check_due(attachment, max_age, now=now):
            continue

        try:
            last_checked = datetime.fromisoformat(attachment)
        except (TypeError, ValueError):
            # Unparseable timestamps count as never-checked.
            never_checked.append(record_id)
            continue
        if last_checked.tzinfo is None:
            last_checked = last_checked.replace(tzinfo=timezone.utc)
        checked.append((last_checked, record_id))

    never_checked.sort()
    checked.sort(key=lambda row: row[0])
    return never_checked + [record_id for _, record_id in checked]


def sync_db_attachments(db, config, max_age=BACKGROUND_RECHECK_INTERVAL,
                        delay=1.0, sleep=time.sleep):
    """Refresh the Zotero attachment cache of every due record in `db`.

    Records whose cached lookup changed between the initial scan and their
    turn (e.g. a concurrent PDF upload stored an attachment key) are skipped
    without sleeping. A ZoteroLookupError stops this project's pass; the
    next pass retries. After the lookup loop, records whose llm_dispatch row
    is 'missing_pdf' but which now have an attachment key lose that row, so
    the regular top-up can pick them up again.

    Parameters
    ----------
    db : asreview.database.Database
        Open project database.
    config : ZoteroConfig
        Resolved Zotero configuration.
    max_age : float
        Seconds after a failed lookup before it may be repeated.
    delay : float
        Seconds to wait after each Zotero request.
    sleep : callable
        Sleep function (injectable for tests).

    Returns
    -------
    int
        The number of Zotero lookups performed.
    """
    df = db.input[["record_id", "original_id", "attachment", "duplicate_of"]]
    ids = records_due_for_check(
        df.itertuples(index=False, name=None), max_age
    )

    lookups = 0
    keys = 0
    for rid in ids:
        record = db.input.get_records(rid)
        if not attachment_check_due(record.attachment, max_age):
            continue
        try:
            value = refresh_attachment(db, config, record, max_age)
        except ZoteroLookupError as err:
            logging.warning(
                "Zotero sync: lookup failed (%s), stopping this pass", err
            )
            break
        lookups += 1
        if is_attachment_key(value):
            keys += 1
            logging.info(
                "Zotero sync: record %s: attachment key resolved via Zotero",
                rid,
            )
        else:
            logging.info(
                "Zotero sync: record %s: no PDF attachment found in Zotero",
                rid,
            )
        sleep(delay)

    # Sweep: a record that has a missing_pdf dispatch row but an attachment
    # key now no longer needs the row — delete it so the regular top-up
    # re-queues the record in ranking order. Rows with any other status are
    # never touched. This also heals rows left over from before this sweep
    # existed (e.g. a key found by the on-screen check).
    cleared = 0
    for rid in db.get_missing_pdf_dispatch_ids():
        if is_attachment_key(db.input.get_records(rid).attachment):
            if db.delete_missing_pdf_dispatch(rid):
                cleared += 1

    logging.info(
        "Zotero sync: %d lookup(s) performed, %d resolved to keys, "
        "%d cleared from missing_pdf",
        lookups, keys, cleared,
    )
    return lookups


def sync_project_attachments(project_path,
                             max_age=BACKGROUND_RECHECK_INTERVAL,
                             delay=1.0, sleep=time.sleep):
    """Refresh the Zotero attachment cache of one project directory.

    Returns 0 without opening the database when Zotero is not configured.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory.
    max_age : float
        Seconds after a failed lookup before it may be repeated.
    delay : float
        Seconds to wait after each Zotero request.
    sleep : callable
        Sleep function (injectable for tests).

    Returns
    -------
    int
        The number of Zotero lookups performed.
    """
    config = get_zotero_config(project_path)
    if not config.enabled:
        logging.debug(
            "Zotero sync: project %s has no Zotero configuration, skipping",
            project_path,
        )
        return 0

    with asr.Project(project_path) as project:
        db = open_db(project.db_path)
        try:
            return sync_db_attachments(
                db, config, max_age, delay=delay, sleep=sleep
            )
        finally:
            db.close()


def sync_all_projects(max_age=BACKGROUND_RECHECK_INTERVAL, delay=1.0):
    """One pass over every project under the ASReview path.

    Exceptions per project are logged and skipped so one bad project cannot
    stop the pass.

    Parameters
    ----------
    max_age : float
        Seconds after a failed lookup before it may be repeated.
    delay : float
        Seconds to wait after each Zotero request.

    Returns
    -------
    int
        Total number of Zotero lookups performed across all projects.
    """
    # Imported here on purpose: llm_worker imports this module (Part B),
    # so a top-level import would be circular.
    from asreview.webapp._api.llm_worker import discover_project_paths

    total = 0
    for path in discover_project_paths():
        try:
            total += sync_project_attachments(path, max_age, delay=delay)
        except Exception as err:  # noqa: BLE001
            logging.exception("Zotero sync: project %s failed: %s", path, err)
    return total


def run_sync_forever(poll_interval=300.0, delay=1.0):
    """Loop sync_all_projects forever, sleeping between passes.

    Catches and logs any exception from a pass so the thread never dies.

    Parameters
    ----------
    poll_interval : float
        Seconds to sleep between passes.
    delay : float
        Seconds to wait after each Zotero request.
    """
    while True:
        try:
            sync_all_projects(delay=delay)
        except Exception as err:  # noqa: BLE001
            logging.exception("Zotero sync: pass failed: %s", err)
        time.sleep(poll_interval)
