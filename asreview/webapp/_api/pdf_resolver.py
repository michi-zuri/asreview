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

"""Resolve records to locally cached Zotero PDF files."""

from pathlib import Path

from asreview.webapp._api.zotero import (
    get_zotero_config,
    is_attachment_key,
    download_attachment_file,
)


class PdfResolver:
    """Resolve records to locally cached Zotero PDF files.

    The record's ``attachment`` field already holds a Zotero attachment
    key (when a PDF is available) or a failure timestamp otherwise; see
    the Zotero integration. This class only downloads and caches the
    file for keys that are already resolved — it does NOT perform the
    item->attachment lookup itself.
    """

    CACHE_DIRNAME = "llm_pdf_cache"

    def __init__(self, project_path, config=None):
        """Initialize the resolver.

        Parameters
        ----------
        project_path : str or Path
            Path to the project directory.
        config : ZoteroConfig, optional
            Resolved Zotero configuration. If None, read it from the
            project's ``zotero.json`` via ``get_zotero_config``.
        """
        self.project_path = Path(project_path)
        self.config = config if config is not None else get_zotero_config(project_path)
        self.cache_dir = self.project_path / self.CACHE_DIRNAME

    def resolve(self, record):
        """Return a Path to the cached PDF for ``record``, or None.

        Returns None (no PDF available; caller should treat as
        ``missing_pdf``) when:

        - Zotero is not configured (``self.config.enabled`` is False), or
        - ``record.attachment`` is not a valid attachment key
          (i.e. is None or a failure timestamp).

        On a cache hit returns the existing path without any network call.
        On a cache miss downloads the file via ``download_attachment_file``
        and writes it to the cache, then returns the path.

        May raise ``ZoteroLookupError`` if the download fails — the caller
        (worker) is responsible for mapping that to ``missing_pdf`` and
        not retrying. Do NOT catch it here.

        Parameters
        ----------
        record : object
            A record object with an ``attachment`` attribute.

        Returns
        -------
        Path or None
            The path to the cached PDF file, or None if no PDF is available.
        """
        if not self.config.enabled:
            return None
        key = getattr(record, "attachment", None)
        if not is_attachment_key(key):
            return None
        return self._cached_path(key)

    def _cached_path(self, attachment_key):
        """Return the path to the cached PDF for the given attachment key.

        Downloads the file on a cache miss and writes it atomically.

        Parameters
        ----------
        attachment_key : str
            The Zotero attachment key.

        Returns
        -------
        Path
            The path to the cached PDF file.
        """
        dest = self.cache_dir / f"{attachment_key}.pdf"
        if dest.exists():
            return dest
        data = download_attachment_file(self.config, attachment_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return dest
