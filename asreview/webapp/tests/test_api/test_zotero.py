from unittest import mock

import pytest
import requests

from asreview.webapp._api.zotero import (
    ZoteroConfig,
    ZoteroLookupError,
    download_attachment_file,
)


def make_config():
    return ZoteroConfig(
        group_id="123", group_slug="slug", api_key="KEY", recheck_interval=300
    )


def test_download_attachment_file_success():
    """Success: returns response.content bytes."""
    config = make_config()
    fake_bytes = b"%PDF-1.7 fake bytes"

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_response = mock.Mock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = mock.Mock()
        mock_get.return_value = mock_response

        result = download_attachment_file(config, "ABCD1234")

        assert result == fake_bytes
        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        assert call_args[0] == (
            "https://api.zotero.org/groups/123/items/ABCD1234/file"
        )
        assert call_kwargs["headers"]["Zotero-API-Key"] == "KEY"
        assert call_kwargs["headers"]["Zotero-API-Version"] == "3"
        assert call_kwargs["timeout"] == 30


def test_download_attachment_file_http_error():
    """HTTP error raises ZoteroLookupError."""
    config = make_config()

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        with pytest.raises(ZoteroLookupError) as exc_info:
            download_attachment_file(config, "ABCD1234")

        assert "404" in str(exc_info.value)


def test_download_attachment_file_network_error():
    """Network error raises ZoteroLookupError."""
    config = make_config()

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("boom")

        with pytest.raises(ZoteroLookupError) as exc_info:
            download_attachment_file(config, "ABCD1234")

        assert "boom" in str(exc_info.value)
