import types
from unittest import mock

import pytest

from asreview.webapp._api.zotero import ZoteroConfig, ZoteroLookupError
from asreview.webapp._api.pdf_resolver import PdfResolver


def fake_record(attachment):
    return types.SimpleNamespace(
        attachment=attachment, original_id="ITEMKEY0"
    )


def make_enabled_config():
    return ZoteroConfig(
        group_id="123", group_slug="slug", api_key="KEY", recheck_interval=300
    )


def make_disabled_config():
    return ZoteroConfig(
        group_id=None, group_slug=None, api_key=None, recheck_interval=300
    )


def test_disabled_config_returns_none(tmp_path):
    """config.enabled is False -> resolve returns None, download never called."""
    config = make_disabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record("ABCD1234")

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        result = resolver.resolve(record)

    assert result is None
    mock_dl.assert_not_called()


def test_attachment_none_returns_none(tmp_path):
    """attachment is None -> resolve returns None, download never called."""
    config = make_enabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record(None)

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        result = resolver.resolve(record)

    assert result is None
    mock_dl.assert_not_called()


def test_attachment_timestamp_returns_none(tmp_path):
    """attachment is a failure timestamp -> resolve returns None, download never called."""
    config = make_enabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record("2026-01-01T00:00:00+00:00")

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        result = resolver.resolve(record)

    assert result is None
    mock_dl.assert_not_called()


def test_cache_miss_downloads(tmp_path):
    """Cache MISS with a valid key downloads and caches the file."""
    config = make_enabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record("ABCD1234")
    fake_bytes = b"%PDF-1.7 data"

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        mock_dl.return_value = fake_bytes
        result = resolver.resolve(record)

    expected_path = tmp_path / "llm_pdf_cache" / "ABCD1234.pdf"
    assert result == expected_path
    assert expected_path.exists()
    assert expected_path.read_bytes() == fake_bytes
    mock_dl.assert_called_once_with(config, "ABCD1234")


def test_cache_hit_skips_download(tmp_path):
    """Cache HIT returns the existing path without calling download."""
    config = make_enabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record("ABCD1234")

    cache_file = tmp_path / "llm_pdf_cache" / "ABCD1234.pdf"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached_bytes = b"%PDF-1.4 already cached"
    cache_file.write_bytes(cached_bytes)

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        result = resolver.resolve(record)

    assert result == cache_file
    assert result.read_bytes() == cached_bytes
    mock_dl.assert_not_called()


def test_download_failure_propagates(tmp_path):
    """Download failure raises ZoteroLookupError — not swallowed."""
    config = make_enabled_config()
    resolver = PdfResolver(tmp_path, config=config)
    record = fake_record("ABCD1234")

    with mock.patch(
        "asreview.webapp._api.pdf_resolver.download_attachment_file"
    ) as mock_dl:
        mock_dl.side_effect = ZoteroLookupError("boom")
        with pytest.raises(ZoteroLookupError) as exc_info:
            resolver.resolve(record)

    assert "boom" in str(exc_info.value)
