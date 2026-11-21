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

"""Fixtures shared by tests/ and asreview/webapp/tests/."""

import json
import logging
import os
import shutil
import socket
import urllib.request
from pathlib import Path

import pytest
import synergy_dataset as sd

from asreview.datasets import SynergyDataSet

SYNERGY_CACHE_DIR = Path.home() / ".cache" / "asreview_tests" / "synergy"
SYNERGY_DOWNLOAD_TIMEOUT = 60  # seconds without data before a download fails

# The v1.0 tag on GitHub has the same per-dataset files as dataverse.nl.
SYNERGY_GITHUB_CONTENTS = "https://api.github.com/repos/asreview/synergy-dataset/contents/{name}?ref=v{version}"


def _download_synergy_from_github(name):
    """Download one Synergy dataset from GitHub into the synergy_dataset source.

    Afterwards ``synergy_dataset.Dataset(name)`` reads it locally, without
    dataverse.nl.
    """
    target = Path(sd.base._get_path_raw_dataset(), name)
    if target.exists():
        return
    url = SYNERGY_GITHUB_CONTENTS.format(name=name, version=sd.base.SYNERGY_VERSION)
    with urllib.request.urlopen(url) as response:
        files = json.load(response)

    tmp = target.with_name(f"{name}.{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in files:
        if f["type"] == "file":
            urllib.request.urlretrieve(f["download_url"], tmp / f["name"])
    os.replace(tmp, target)  # atomic: a half download never looks complete


@pytest.fixture(autouse=True, scope="session")
def cached_synergy_datasets():
    """Download each Synergy dataset once and reuse it across test runs.

    Synergy benchmarks otherwise download from dataverse.nl each time (no
    timeout, so a stalled connection hangs the run). The CSV is cached in
    ``~/.cache/asreview_tests/synergy``; delete that folder to force a fresh
    download. When dataverse.nl fails, the dataset is fetched from GitHub.
    """
    original_to_file = SynergyDataSet.to_file

    def to_file(self, path=None):
        cached = SYNERGY_CACHE_DIR / self.filename
        if not cached.exists():
            SYNERGY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(f".{os.getpid()}.tmp")
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(SYNERGY_DOWNLOAD_TIMEOUT)
            try:
                try:
                    original_to_file(self, tmp)
                except Exception as err:
                    logging.warning(
                        "dataverse.nl download of %s failed (%r); using GitHub",
                        self.dataset_id,
                        err,
                    )
                    _download_synergy_from_github(self.dataset_id)
                    original_to_file(self, tmp)  # now reads the local source
            finally:
                socket.setdefaulttimeout(old_timeout)
            os.replace(tmp, cached)  # atomic: parallel runs never see half a file
        if path is None:
            return cached.read_text()
        shutil.copyfile(cached, path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SynergyDataSet, "to_file", to_file)
        yield
