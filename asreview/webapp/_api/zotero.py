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

"""Helpers for linking ASReview records to their Zotero full text.

Each record can have an ``original_id`` that corresponds to a Zotero item key. This
module queries the Zotero web API for the PDF attachment of such an item and builds a
link that opens the Zotero PDF reader.

Zotero credentials are stored per project in a ``zotero.json`` file inside the
project directory.
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import requests

__all__ = [
    "ZoteroConfig",
    "ZoteroLookupError",
    "ZoteroUploadError",
    "ZoteroDeleteError",
    "get_zotero_config",
    "is_attachment_key",
    "fetch_pdf_attachment_key",
    "download_attachment_file",
    "build_reader_url",
    "upload_pdf_to_zotero",
    "delete_pdf",
    "validate_zotero_credentials",
]


class ZoteroLookupError(Exception):
    """Raised when a Zotero attachment lookup fails for a transient reason.

    This distinguishes "we could not reach Zotero / the request failed" from "Zotero
    told us this item has no PDF attachment". Only the latter should be cached as a
    negative result; transient failures should be retried on the next request.
    """


class ZoteroUploadError(Exception):
    """Raised when uploading a PDF attachment to Zotero fails.

    This covers permanent failures (permissions, quota) and transient failures
    (network, library locked). The caller should distinguish using the HTTP status
    code attached to the exception.
    """

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class ZoteroDeleteError(Exception):
    """Raised when deleting a Zotero attachment fails.

    Covers permanent failures (permissions), conflicts (concurrent modification),
    and transient failures (network, library locked). The caller should distinguish
    using the HTTP status code attached to the exception.
    """

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

# A Zotero object key is exactly 8 characters from the set [A-Z0-9].
# See https://www.zotero.org/support/dev/web_api/v3/basics#zotero_web_api_item_typefield_requests
ZOTERO_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

ZOTERO_API_BASE = "https://api.zotero.org"
ZOTERO_WEB_BASE = "https://www.zotero.org"
ZOTERO_API_VERSION = "3"

# By default re-check items whose full text was previously unavailable every 5 minutes.
DEFAULT_RECHECK_INTERVAL = 300


class ZoteroConfig:
    """Resolved Zotero configuration read from a project's ``zotero.json``."""

    def __init__(self, group_id, group_slug, api_key, recheck_interval):
        self.group_id = group_id
        self.group_slug = group_slug
        self.api_key = api_key
        self.recheck_interval = recheck_interval

    @property
    def enabled(self):
        """Whether enough is configured to query Zotero."""
        return bool(self.group_id and self.api_key)


def get_zotero_config(project_path):
    """Read the Zotero configuration from a project's ``zotero.json`` file.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory.

    Returns
    -------
    ZoteroConfig
    """
    config_path = Path(project_path, "zotero.json")

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    group_id = data.get("group_id") or None
    group_slug = data.get("group_slug") or None
    api_key = data.get("api_key") or None

    return ZoteroConfig(
        group_id=str(group_id) if group_id is not None else None,
        group_slug=str(group_slug) if group_slug is not None else None,
        api_key=str(api_key) if api_key is not None else None,
        recheck_interval=DEFAULT_RECHECK_INTERVAL,
    )


def is_attachment_key(value):
    """Return whether `value` looks like a Zotero attachment key (and not a timestamp).

    The ``attachment`` field of a record holds either a Zotero key or an ISO 8601
    timestamp of when the lookup last failed. This distinguishes the two.
    """
    return bool(value) and bool(ZOTERO_KEY_RE.match(value))


def fetch_pdf_attachment_key(config, item_key, timeout=10):
    """Query the Zotero API for the PDF attachment key of an item.

    Parameters
    ----------
    config : ZoteroConfig
        Resolved Zotero configuration.
    item_key : str
        The Zotero item key (stored as the record's ``original_id``).
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    str | None
        The key of the first PDF attachment of the item, or `None` if Zotero reports
        that the item has no PDF attachment.

    Raises
    ------
    ZoteroLookupError
        If the request to Zotero fails (network error, rate limit, invalid response).
        This is distinct from `None`, which means the lookup succeeded but found no PDF.
    """
    url = (
        f"{ZOTERO_API_BASE}/groups/{config.group_id}"
        f"/items/{item_key}/children"
    )
    headers = {
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Zotero-API-Key": config.api_key,
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"itemType": "attachment"},
            timeout=timeout,
        )
        response.raise_for_status()
        children = response.json()
    except requests.RequestException as err:
        logging.warning(f"Failed to fetch Zotero attachments for {item_key}: {err}")
        raise ZoteroLookupError(str(err)) from err
    except ValueError as err:
        logging.warning(f"Invalid Zotero response for {item_key}: {err}")
        raise ZoteroLookupError(str(err)) from err

    for child in children:
        data = child.get("data", {})
        if (
            data.get("itemType") == "attachment"
            and data.get("contentType") == "application/pdf"
            and data.get("key")
        ):
            return data["key"]

    return None


def download_attachment_file(config, attachment_key, timeout=30):
    """Download the raw file bytes of a Zotero attachment.

    Parameters
    ----------
    config : ZoteroConfig
        Resolved Zotero configuration.
    attachment_key : str
        The Zotero attachment key (as returned by `fetch_pdf_attachment_key`).
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    bytes
        The raw bytes of the attachment file.

    Raises
    ------
    ZoteroLookupError
        If the request to Zotero fails (network error, HTTP error, etc.).
    """
    url = (
        f"{ZOTERO_API_BASE}/groups/{config.group_id}"
        f"/items/{attachment_key}/file"
    )
    headers = {
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Zotero-API-Key": config.api_key,
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as err:
        logging.warning(
            f"Failed to download attachment {attachment_key}: {err}"
        )
        raise ZoteroLookupError(str(err)) from err


def build_reader_url(config, item_key, attachment_key):
    """Build a Zotero PDF reader URL.

    The resulting URL opens the Zotero web reader for the given attachment, for example
    ``https://www.zotero.org/groups/6350524/fiierce/items/ANQ4888U/attachment/4KVUV33G/reader``.
    """
    group_segment = f"{config.group_id}"
    if config.group_slug:
        group_segment = f"{group_segment}/{config.group_slug}"

    return (
        f"{ZOTERO_WEB_BASE}/groups/{group_segment}"
        f"/items/{item_key}/attachment/{attachment_key}/reader"
    )


def _zotero_upload_handshake(config, attachment_key, name, blob, md5_hex, mtime_ms):
    """Run the three-step Zotero file upload handshake (authorize → upload → register).

    Returns the attachment key on success (it may be immediately usable if Zotero
    deduplicates via ``exists``). Raises ``ZoteroUploadError`` on any failure.
    """
    s = requests.Session()
    s.headers.update({
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Authorization": f"Bearer {config.api_key}",
    })
    prefix = f"{ZOTERO_API_BASE}/groups/{config.group_id}"

    # Step 2a — request upload authorization
    try:
        r = s.post(
            f"{prefix}/items/{attachment_key}/file",
            data={
                "md5": md5_hex,
                "filename": name,
                "filesize": len(blob),
                "mtime": mtime_ms,
            },
            headers={"If-None-Match": "*"},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as err:
        raise ZoteroUploadError(
            f"Upload authorization failed: {err}",
            status_code=getattr(err.response, "status_code", None),
        ) from err

    auth = r.json()

    # Zotero already holds a file with this hash — done.
    if auth.get("exists"):
        return attachment_key

    # Step 2b — upload bytes to the S3 endpoint (no Zotero auth headers)
    payload = auth["prefix"].encode() + blob + auth["suffix"].encode()
    try:
        r2 = requests.post(
            auth["url"],
            data=payload,
            headers={"Content-Type": auth["contentType"]},
            timeout=120,
        )
        r2.raise_for_status()
    except requests.RequestException as err:
        raise ZoteroUploadError(
            f"File upload to storage failed: {err}",
            status_code=getattr(err.response, "status_code", None),
        ) from err

    # Step 2c — register the upload
    try:
        r = s.post(
            f"{prefix}/items/{attachment_key}/file",
            data={"upload": auth["uploadKey"]},
            headers={"If-None-Match": "*"},
            timeout=30,
        )
        if r.status_code == 412:
            raise ZoteroUploadError(
                "File already exists on attachment (concurrent upload)",
                status_code=412,
            )
        if r.status_code != 204:
            raise ZoteroUploadError(
                f"Upload registration failed: {r.status_code} {r.text}",
                status_code=r.status_code,
            )
    except requests.RequestException as err:
        raise ZoteroUploadError(
            f"Upload registration failed: {err}",
            status_code=getattr(err.response, "status_code", None),
        ) from err

    # Cheap verification
    try:
        r = s.get(f"{prefix}/items/{attachment_key}", timeout=10)
        r.raise_for_status()
        if r.json()["data"].get("md5") != md5_hex:
            raise ZoteroUploadError("MD5 mismatch after registration")
    except (requests.RequestException, ValueError, KeyError) as err:
        raise ZoteroUploadError(
            f"Upload verification failed: {err}",
            status_code=getattr(err.response, "status_code", None) if hasattr(err, "response") else None,
        ) from err

    return attachment_key


def upload_pdf_to_zotero(config, parent_item_key, pdf_path):
    """Upload a PDF to Zotero as a child attachment of a parent item.

    Creates a new child attachment item (``linkMode: imported_file``) and uploads
    the file through Zotero's three-step authorization handshake. On success,
    returns the new attachment key.

    Parameters
    ----------
    config : ZoteroConfig
        Resolved Zotero configuration. Must have ``enabled == True``.
    parent_item_key : str
        The Zotero item key of the parent article.
    pdf_path : str or Path
        Local path to the PDF file to upload.

    Returns
    -------
    str
        The Zotero attachment key of the newly created attachment.

    Raises
    ------
    ZoteroUploadError
        If any part of the process fails. The exception carries a ``status_code``
        attribute for distinguishing permanent (403, 413) from transient (409, 5xx)
        failures.
    """
    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        blob = f.read()

    md5_hex = hashlib.md5(blob).hexdigest()
    mtime_ms = int(os.path.getmtime(pdf_path) * 1000)
    name = pdf_path.name

    s = requests.Session()
    s.headers.update({
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Authorization": f"Bearer {config.api_key}",
    })
    api_prefix = f"{ZOTERO_API_BASE}/groups/{config.group_id}"

    # Phase 1 — create child attachment item
    try:
        r = s.post(
            f"{api_prefix}/items",
            json=[{
                "itemType": "attachment",
                "linkMode": "imported_file",
                "parentItem": parent_item_key,
                "title": "Full Text PDF",
                "contentType": "application/pdf",
                "filename": name,
                "tags": [],
                "relations": {},
                "md5": None,
                "mtime": None,
            }],
            headers={"Zotero-Write-Token": uuid.uuid4().hex},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as err:
        raise ZoteroUploadError(
            f"Attachment item creation failed: {err}",
            status_code=getattr(err.response, "status_code", None),
        ) from err

    body = r.json()
    if body.get("failed"):
        failed_entry = body["failed"].get("0", {})
        raise ZoteroUploadError(
            f"Attachment item creation failed: {failed_entry}",
            status_code=r.status_code,
        )

    try:
        att_key = body["successful"]["0"]["key"]
    except (KeyError, IndexError):
        raise ZoteroUploadError(
            "Unexpected response creating attachment item",
            status_code=r.status_code,
        )

    # Phase 2 — three-step handshake
    _zotero_upload_handshake(config, att_key, name, blob, md5_hex, mtime_ms)

    logging.info(
        "Uploaded PDF '%s' to Zotero parent %s → attachment %s",
        name, parent_item_key, att_key,
    )
    return att_key


def delete_pdf(config, attachment_key):
    """Permanently delete a Zotero attachment item and its stored file.

    Idempotent: a missing or already-deleted attachment is treated as success.
    Includes one automatic retry on version conflict (HTTP 412).

    Parameters
    ----------
    config : ZoteroConfig
        Resolved Zotero configuration. Must have ``enabled == True``.
    attachment_key : str
        The Zotero attachment key to delete.

    Raises
    ------
    ZoteroDeleteError
        If the delete fails after retries. The exception carries a
        ``status_code`` attribute for programmatic handling (403, 409, 412).
    """
    s = requests.Session()
    s.headers.update({
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Authorization": f"Bearer {config.api_key}",
    })
    base = f"{ZOTERO_API_BASE}/groups/{config.group_id}/items/{attachment_key}"

    for attempt in range(2):
        # Step 0 — fetch the current version for optimistic concurrency
        try:
            r = s.get(base, timeout=10)
        except requests.RequestException as err:
            raise ZoteroDeleteError(
                f"Failed to fetch attachment version: {err}",
                status_code=getattr(err.response, "status_code", None),
            ) from err

        if r.status_code == 404:
            return  # already gone — idempotent success

        try:
            r.raise_for_status()
        except requests.RequestException as err:
            raise ZoteroDeleteError(
                f"Failed to fetch attachment version: {err}",
                status_code=r.status_code,
            ) from err

        version = r.headers["Last-Modified-Version"]

        # Delete
        try:
            resp = s.delete(
                base,
                headers={"If-Unmodified-Since-Version": version},
                timeout=30,
            )
        except requests.RequestException as err:
            raise ZoteroDeleteError(
                f"Delete request failed: {err}",
                status_code=getattr(err.response, "status_code", None),
            ) from err

        if resp.status_code in (204, 404):
            logging.info(
                "Deleted Zotero attachment %s (group %s)", attachment_key, config.group_id,
            )
            return

        if resp.status_code == 412 and attempt == 0:
            continue  # version raced — retry once

        raise ZoteroDeleteError(
            f"Delete failed with status {resp.status_code}: {resp.text}",
            status_code=resp.status_code,
        )

    raise ZoteroDeleteError(
        "Version conflict persisted after retry", status_code=412,
    )


def validate_zotero_credentials(group_id, api_key, timeout=10):
    """Validate Zotero credentials and return the group name.

    Queries the Zotero API for the group metadata. A successful response
    proves the API key has read access to the group.

    Parameters
    ----------
    group_id : str
        Numeric Zotero group ID.
    api_key : str
        Zotero API key with read access to the group.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    str
        The group name from ``.data.name`` in the API response.

    Raises
    ------
    ZoteroLookupError
        If the request fails (network error, HTTP error, invalid key, etc.).
    """
    url = f"{ZOTERO_API_BASE}/groups/{group_id}"
    headers = {
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Zotero-API-Key": api_key,
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as err:
        raise ZoteroLookupError(
            f"Could not reach Zotero: {err}"
        ) from err
    except ValueError as err:
        raise ZoteroLookupError(
            f"Invalid Zotero response: {err}"
        ) from err

    name = data.get("data", {}).get("name", "")
    if not name:
        raise ZoteroLookupError(
            "Zotero returned a response but no group name was found. "
            "Check the group ID and API key."
        )
    return name
