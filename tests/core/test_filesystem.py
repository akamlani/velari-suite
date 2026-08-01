"""Tests for velari_core.core.io.filesystem."""

import contextlib
import io
import json
import zipfile

import httpx
import pytest


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, content=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")

    def iter_bytes(self):
        yield self._content


@contextlib.contextmanager
def _fake_stream_response(content=b"", status_code=200):
    yield _FakeResponse(status_code=status_code, content=content)


def _raise_http_error(*args, **kwargs):
    raise httpx.HTTPError("simulated connection failure")


def _capturing_put(calls, status_code=200):
    def _put(uri, content=None, **kwargs):
        data = b"".join(content) if content is not None else b""
        calls.append((uri, data))
        return _FakeResponse(status_code=status_code)

    return _put


def _make_zip_bytes(filename: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


class TestReadWrite:
    def test_write_read_json_roundtrip(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        path = tmp_path / "data.json"
        Filesystem.write(path, {"key": "value"})
        assert Filesystem.read(path) == {"key": "value"}

    def test_write_read_yaml_roundtrip(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        path = tmp_path / "data.yaml"
        Filesystem.write(path, {"key": "value"})
        assert Filesystem.read(path) == {"key": "value"}

    def test_write_read_text_roundtrip(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        path = tmp_path / "notes.txt"
        Filesystem.write(path, "hello world")
        assert Filesystem.read(path) == "hello world"

    def test_write_read_bytes_roundtrip(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        path = tmp_path / "blob.bin"
        Filesystem.write(path, b"\x00\x01\x02")
        assert Filesystem.read(path) == b"\x00\x01\x02"

    def test_read_missing_file_raises_filenotfounderror(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        with pytest.raises(FileNotFoundError):
            Filesystem.read(tmp_path / "missing.txt")

    def test_read_invalid_json_raises_jsondecodeerror(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            Filesystem.read(path)


class TestDirectoryOps:
    def test_make_dir_creates_nested_and_is_idempotent(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        target = tmp_path / "a" / "b" / "c"
        result = Filesystem.make_dir(target)
        assert result == target
        assert target.is_dir()
        Filesystem.make_dir(target)

    def test_copy_file(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        src = tmp_path / "notes.txt"
        src.write_text("hello")
        dst = tmp_path / "copy.txt"
        result = Filesystem.copy(src, dst)
        assert result == dst
        assert dst.read_text() == "hello"

    def test_copy_directory(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "notes.txt").write_text("hello")
        dst_dir = tmp_path / "dstdir"
        result = Filesystem.copy(src_dir, dst_dir)
        assert result == dst_dir
        assert (dst_dir / "notes.txt").read_text() == "hello"

    def test_move_file(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        src = tmp_path / "notes.txt"
        src.write_text("hello")
        dst = tmp_path / "moved.txt"
        result = Filesystem.move(src, dst)
        assert result == dst
        assert dst.read_text() == "hello"
        assert not src.exists()

    def test_delete_file(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        target = tmp_path / "notes.txt"
        target.write_text("hello")
        Filesystem.delete(target)
        assert not target.exists()

    def test_delete_directory(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        target = tmp_path / "dir"
        target.mkdir()
        (target / "notes.txt").write_text("hello")
        Filesystem.delete(target)
        assert not target.exists()

    def test_delete_missing_raises_filenotfounderror(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        with pytest.raises(FileNotFoundError):
            Filesystem.delete(tmp_path / "missing.txt")

    def test_list_glob_pattern(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        (tmp_path / "a.txt").write_text("1")
        (tmp_path / "b.txt").write_text("2")
        (tmp_path / "c.json").write_text("{}")
        result = Filesystem.list(tmp_path, "*.txt")
        assert set(result) == {tmp_path / "a.txt", tmp_path / "b.txt"}


class TestArchive:
    @pytest.mark.parametrize("archive_name", ["bundle.zip", "bundle.tar.gz"])
    def test_compress_extract_roundtrip(self, tmp_path, archive_name):
        from velari_core.core.io.filesystem import Filesystem

        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "notes.txt").write_text("hello")

        archive_path = tmp_path / archive_name
        result = Filesystem.compress(src_dir, archive_path)
        assert result == archive_path
        assert archive_path.exists()

        extract_dir = tmp_path / "extracted"
        extracted = Filesystem.extract(archive_path, extract_dir)
        assert extracted == extract_dir
        assert (extract_dir / "payload" / "notes.txt").read_text() == "hello"

    def test_compress_unsupported_format_raises_valueerror(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        src = tmp_path / "file.txt"
        src.write_text("x")
        with pytest.raises(ValueError):
            Filesystem.compress(src, tmp_path / "out.unknownext")

    def test_extract_unsupported_format_raises_valueerror(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem

        bogus = tmp_path / "file.unknownext"
        bogus.write_text("x")
        with pytest.raises(ValueError):
            Filesystem.extract(bogus)


class TestProperties:
    def test_get_mime_type(self):
        from velari_core.core.io.filesystem import Filesystem

        assert Filesystem.get_mime_type("notes.txt") == "text/plain"
        assert Filesystem.get_mime_type("file.unknownext123") == "application/octet-stream"

    def test_get_properties_local_file(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem
        from velari_core.core.io.types import ArtifactKind

        path = tmp_path / "notes.txt"
        path.write_text("hello")
        props = Filesystem.get_properties(path)
        assert props.location.is_local is True
        assert props.kind.exists is True
        assert props.kind.type == ArtifactKind.FILE
        assert props.stats.size == len(b"hello")

    def test_get_properties_local_directory(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem
        from velari_core.core.io.types import ArtifactKind

        (tmp_path / "notes.txt").write_text("hello")
        props = Filesystem.get_properties(tmp_path)
        assert props.kind.type == ArtifactKind.DIRECTORY
        assert props.kind.exists is True

    def test_get_properties_local_missing_path(self, tmp_path):
        from velari_core.core.io.filesystem import Filesystem
        from velari_core.core.io.types import ArtifactKind

        props = Filesystem.get_properties(tmp_path / "missing.txt")
        assert props.kind.exists is False
        assert props.kind.type == ArtifactKind.UNKNOWN

    def test_get_properties_remote_success(self, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem
        from velari_core.core.io.types import ArtifactKind

        monkeypatch.setattr(
            httpx,
            "head",
            lambda *a, **kw: _FakeResponse(
                status_code=200,
                headers={"content-length": "1024", "content-type": "application/pdf; charset=binary"},
            ),
        )
        props = Filesystem.get_properties("https://example.com/report.pdf")
        assert props.location.is_remote is True
        assert props.kind.exists is True
        assert props.kind.type == ArtifactKind.FILE
        assert props.stats.size == 1024
        assert props.name.mime_type == "application/pdf"

    def test_get_properties_remote_head_failure_falls_back(self, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem
        from velari_core.core.io.types import ArtifactKind

        monkeypatch.setattr(httpx, "head", _raise_http_error)
        props = Filesystem.get_properties("https://example.com/data.bin")
        assert props.kind.exists is False
        assert props.kind.type == ArtifactKind.UNKNOWN
        assert props.stats.size is None


class TestDownloadUpload:
    def test_download_single_uri(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _fake_stream_response(content=b"hello world"))
        dest = tmp_path / "notes.txt"
        result = Filesystem.download("https://example.com/notes.txt", dest=str(dest))
        assert result == dest
        assert dest.read_bytes() == b"hello world"

    def test_download_list_of_uris(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _fake_stream_response(content=b"hello world"))
        dest_a, dest_b = tmp_path / "a.txt", tmp_path / "b.txt"
        result = Filesystem.download(
            [("https://example.com/a.txt", str(dest_a)), ("https://example.com/b.txt", str(dest_b))]
        )
        assert result == [dest_a, dest_b]
        assert dest_a.read_bytes() == b"hello world"
        assert dest_b.read_bytes() == b"hello world"

    def test_download_auto_extracts_archive(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        zip_bytes = _make_zip_bytes("inner.txt", b"payload")
        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _fake_stream_response(content=zip_bytes))
        dest = tmp_path / "bundle.zip"
        result = Filesystem.download("https://example.com/bundle.zip", dest=str(dest))
        assert result == dest.parent
        assert (dest.parent / "inner.txt").read_bytes() == b"payload"

    def test_download_http_error_is_raised(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _fake_stream_response(status_code=500))
        with pytest.raises(httpx.HTTPError):
            Filesystem.download("https://example.com/notes.txt", dest=str(tmp_path / "notes.txt"))

    def test_upload_single_path_requires_uri_raises_valueerror(self):
        from velari_core.core.io.filesystem import Filesystem

        with pytest.raises(ValueError):
            Filesystem.upload("some/local/path")

    def test_upload_single_path(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        calls = []
        monkeypatch.setattr(httpx, "put", _capturing_put(calls))
        src = tmp_path / "notes.txt"
        src.write_text("hello")
        uri = "https://example.com/upload/notes.txt"
        result = Filesystem.upload(src, uri=uri)
        assert result == uri
        assert calls == [(uri, b"hello")]

    def test_upload_list_of_paths(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        calls = []
        monkeypatch.setattr(httpx, "put", _capturing_put(calls))
        src_a, src_b = tmp_path / "a.txt", tmp_path / "b.txt"
        src_a.write_text("a")
        src_b.write_text("b")
        uri_a, uri_b = "https://example.com/a.txt", "https://example.com/b.txt"
        result = Filesystem.upload([(src_a, uri_a), (src_b, uri_b)])
        assert result == [uri_a, uri_b]

    def test_upload_directory_auto_compresses(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        calls = []
        monkeypatch.setattr(httpx, "put", _capturing_put(calls))
        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "notes.txt").write_text("hello")
        uri = "https://example.com/payload.zip"
        result = Filesystem.upload(src_dir, uri=uri)
        assert result == uri
        assert calls[0][0] == uri
        assert calls[0][1].startswith(b"PK")  # zip magic number - confirms compression ran

    def test_upload_http_error_is_raised(self, tmp_path, monkeypatch):
        from velari_core.core.io.filesystem import Filesystem

        monkeypatch.setattr(httpx, "put", _capturing_put([], status_code=500))
        src = tmp_path / "notes.txt"
        src.write_text("hello")
        with pytest.raises(httpx.HTTPError):
            Filesystem.upload(src, uri="https://example.com/notes.txt")
