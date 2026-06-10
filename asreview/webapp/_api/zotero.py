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
"""

import logging
import re

import requests
from flask import current_app

__all__ = [
    "ZoteroConfig",
    "ZoteroLookupError",
    "get_zotero_config",
    "is_attachment_key",
    "fetch_pdf_attachment_key",
    "build_reader_url",
]


class ZoteroLookupError(Exception):
    """Raised when a Zotero attachment lookup fails for a transient reason.

    This distinguishes "we could not reach Zotero / the request failed" from "Zotero
    told us this item has no PDF attachment". Only the latter should be cached as a
    negative result; transient failures should be retried on the next request.
    """

# A Zotero object key is exactly 8 characters from the set [A-Z0-9].
# See https://www.zotero.org/support/dev/web_api/v3/basics#zotero_web_api_item_typefield_requests
ZOTERO_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

ZOTERO_API_BASE = "https://api.zotero.org"
ZOTERO_WEB_BASE = "https://www.zotero.org"
ZOTERO_API_VERSION = "3"

# By default re-check items whose full text was previously unavailable every 5 minutes.
DEFAULT_RECHECK_INTERVAL = 300


class ZoteroConfig:
    """Resolved Zotero configuration read from the Flask app config."""

    def __init__(self, group_id, group_slug, api_key, recheck_interval):
        self.group_id = group_id
        self.group_slug = group_slug
        self.api_key = api_key
        self.recheck_interval = recheck_interval

    @property
    def enabled(self):
        """Whether enough is configured to query Zotero."""
        return bool(self.group_id and self.api_key)


def get_zotero_config():
    """Read the Zotero configuration from the current Flask app config.

    The following config keys are used (settable via the ``ASREVIEW_LAB_`` prefixed
    environment variables, e.g. ``ASREVIEW_LAB_ZOTERO_GROUP_ID``):

    - ``ZOTERO_GROUP_ID``: numeric id of the Zotero group library.
    - ``ZOTERO_GROUP_SLUG``: url slug of the group (used to build reader links).
    - ``ZOTERO_API_KEY``: Zotero API key with read access to the group.
    - ``ZOTERO_RECHECK_INTERVAL``: seconds before re-checking an item whose full text
      was previously unavailable. Defaults to five minutes.

    Returns
    -------
    ZoteroConfig
    """
    group_id = current_app.config.get("ZOTERO_GROUP_ID")
    group_slug = current_app.config.get("ZOTERO_GROUP_SLUG")
    api_key = current_app.config.get("ZOTERO_API_KEY")
    recheck_interval = current_app.config.get(
        "ZOTERO_RECHECK_INTERVAL", DEFAULT_RECHECK_INTERVAL
    )

    return ZoteroConfig(
        group_id=str(group_id) if group_id is not None else None,
        group_slug=str(group_slug) if group_slug is not None else None,
        api_key=str(api_key) if api_key is not None else None,
        recheck_interval=int(recheck_interval),
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
