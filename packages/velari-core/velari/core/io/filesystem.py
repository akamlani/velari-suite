import  httpx
import  json
import  logging
import  mimetypes
import  shutil
import  tempfile
import  yaml
from    pathlib import Path
from    typing import Any, Iterator, List, Optional, Tuple, Union

from    .types import ArtifactFormat, ArtifactKind, ArtifactProperties

logger = logging.getLogger(__name__)

get_username  = lambda: Path.home().name
get_user_home = lambda: str(Path.home())

_ARCHIVE_FORMATS = {
    suffix: name
    for name, suffixes, _ in shutil.get_unpack_formats()
    for suffix in suffixes
}


class Filesystem(object):
    """Filesystem operations: read, write, move, delete, list, make_dir, copy, extract, compress, download, upload, get_properties.

    All methods are static — call them directly on the class, no instance needed.
    download streams a remote URI (or a list of (uri, dest) pairs, for bulk downloads)
    to a local file via httpx, auto-extracting the result if it's a recognized archive
    format (via extract()). dest/dest_dir default to a new OS temp directory when omitted.
    upload streams a local file (or a list of (path, uri) pairs, for bulk uploads) to a
    remote URI via PUT, auto-compressing directories to a temp archive first.
    get_properties inspects a local path or remote URI and returns an ArtifactProperties
    dataclass with location/kind/name/stats subsections (local paths use pathlib/shutil;
    remote URIs use a real httpx.head() request).

    Examples:
        >>> Filesystem.get_mime_type("notes.txt")
        'text/plain'
        >>> Filesystem.write("notes.txt", "hello")
        >>> Filesystem.read("notes.txt")
        'hello'
        >>> Filesystem.write("config.json", {"key": "value"})
        >>> Filesystem.read("config.json")
        {'key': 'value'}
        >>> Filesystem.make_dir("archive")
        >>> Filesystem.move("notes.txt", "archive/notes.txt")
        >>> Filesystem.make_dir("backup")
        >>> Filesystem.copy("archive/notes.txt", "backup/notes.txt")                   # deep copy (default)
        >>> Filesystem.copy("archive/notes.txt", "backup/notes_link.txt", deep=False)  # shallow copy, preserves symlinks
        >>> Filesystem.list("archive")
        [PosixPath('archive/notes.txt')]
        >>> Filesystem.compress("archive", "bundle.zip")
        PosixPath('bundle.zip')
        >>> Filesystem.extract("bundle.zip", "extracted")
        PosixPath('extracted')
        >>> Filesystem.delete("archive/notes.txt")
    """

    @staticmethod
    def get_mime_type(file_path: Union[str, Path]) -> str:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    @staticmethod
    def get_properties(path: Union[str, Path]) -> ArtifactProperties:
        if str(path).startswith(("http://", "https://")):
            return Filesystem._remote_properties(str(path))
        return Filesystem._local_properties(Path(path))

    @staticmethod
    def _local_properties(local_path: Path) -> ArtifactProperties:
        exists = local_path.exists()
        is_dir = exists and local_path.is_dir()
        suffix = Filesystem._archive_suffix(local_path)
        kind = (
            ArtifactKind.DIRECTORY if is_dir
            else ArtifactKind.ARCHIVE if suffix is not None
            else ArtifactKind.FILE if exists
            else ArtifactKind.UNKNOWN
        )

        size = None
        if exists:
            size = (
                sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
                if is_dir else local_path.stat().st_size
            )
        disk_total, disk_used, disk_free = shutil.disk_usage(local_path) if exists else (None, None, None)

        return ArtifactProperties(
            location=ArtifactProperties.Location(is_local=True, is_remote=False, path=str(local_path), uri=None),
            kind=ArtifactProperties.Kind(exists=exists, type=kind),
            name=ArtifactProperties.Name(
                parent=str(local_path.parent),
                base_name=local_path.name,
                extension=suffix or local_path.suffix,
                mime_type=Filesystem.get_mime_type(local_path),
            ),
            stats=ArtifactProperties.Stats(size=size, disk_total=disk_total, disk_used=disk_used, disk_free=disk_free),
        )

    @staticmethod
    def _remote_properties(uri: str) -> ArtifactProperties:
        url_path = Path(httpx.URL(uri).path)
        suffix = Filesystem._archive_suffix(url_path)
        exists, size, mime_type = False, None, Filesystem.get_mime_type(url_path.name)
        try:
            response = httpx.head(uri, follow_redirects=True)
            exists = response.status_code < 400
            content_length = response.headers.get("content-length")
            size = int(content_length) if content_length is not None else None
            mime_type = response.headers.get("content-type", mime_type).split(";")[0]
        except httpx.HTTPError as e:
            logger.warning(f"failed to fetch remote properties for {uri}: {e}")
        kind = ArtifactKind.ARCHIVE if suffix is not None else ArtifactKind.FILE if exists else ArtifactKind.UNKNOWN

        return ArtifactProperties(
            location=ArtifactProperties.Location(is_local=False, is_remote=True, path=None, uri=uri),
            kind=ArtifactProperties.Kind(exists=exists, type=kind),
            name=ArtifactProperties.Name(
                parent=str(url_path.parent),
                base_name=url_path.name,
                extension=suffix or url_path.suffix,
                mime_type=mime_type,
            ),
            stats=ArtifactProperties.Stats(size=size, disk_total=None, disk_used=None, disk_free=None),
        )

    @staticmethod
    def read(path: Union[str, Path]) -> Any:
        file_path = Path(path)
        try:
            fmt = ArtifactFormat.from_ext(file_path.suffix.lower())
            match fmt:
                case ArtifactFormat.JSON:
                    return json.loads(file_path.read_text())
                case ArtifactFormat.YAML:
                    return yaml.safe_load(file_path.read_text())
                case _ if Filesystem.get_mime_type(file_path).startswith("text/"):
                    return file_path.read_text()
                case _:
                    return file_path.read_bytes()
        except FileNotFoundError as e:
            logger.error(f"file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"invalid JSON in {path}: {e}")
            raise

    @staticmethod
    def write(path: Union[str, Path], content: Any) -> None:
        dest = Path(path)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fmt = ArtifactFormat.from_ext(dest.suffix.lower())
            match fmt:
                case ArtifactFormat.JSON:
                    dest.write_text(json.dumps(content))
                case ArtifactFormat.YAML:
                    dest.write_text(yaml.safe_dump(content))
                case _ if isinstance(content, bytes):
                    dest.write_bytes(content)
                case _:
                    dest.write_text(content)
        except OSError as e:
            logger.error(f"failed to write file {dest}: {e}")
            raise

    @staticmethod
    def make_dir(path: Union[str, Path]) -> Path:
        target = Path(path)
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError as e:
            logger.error(f"failed to create directory {target}: {e}")
            raise

    @staticmethod
    def copy(src: Union[str, Path], dst: Union[str, Path], deep: bool = True) -> Path:
        source = Path(src)
        try:
            if source.is_dir():
                result = shutil.copytree(source, dst, symlinks=not deep, dirs_exist_ok=True)
            else:
                result = shutil.copy2(source, dst, follow_symlinks=deep)
            return Path(result)
        except (FileNotFoundError, shutil.Error, OSError) as e:
            logger.error(f"failed to copy {src} to {dst}: {e}")
            raise

    @staticmethod
    def extract(path: Union[str, Path], dest_dir: Optional[Union[str, Path]] = None) -> Path:
        archive_path = Path(path)
        dest = Path(dest_dir) if dest_dir is not None else Path(tempfile.mkdtemp())
        suffix = Filesystem._archive_suffix(archive_path)
        if suffix is None:
            raise ValueError(f"unsupported archive format: {archive_path}")
        fmt = _ARCHIVE_FORMATS[suffix]
        try:
            dest.mkdir(parents=True, exist_ok=True)
            extra = {} if fmt == "zip" else {"filter": "data"}
            shutil.unpack_archive(str(archive_path), str(dest), format=fmt, **extra)
            return dest
        except OSError as e:
            logger.error(f"failed to extract {archive_path}: {e}")
            raise

    @staticmethod
    def compress(path: Union[str, Path], archive_path: Union[str, Path]) -> Path:
        source = Path(path)
        dest = Path(archive_path)
        suffix = Filesystem._archive_suffix(dest)
        if suffix is None:
            raise ValueError(f"unsupported archive format: {dest}")
        fmt = _ARCHIVE_FORMATS[suffix]
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            base_name = str(dest)[: -len(suffix)]
            result = Path(shutil.make_archive(base_name, fmt, root_dir=str(source.parent), base_dir=source.name))
            if result != dest:
                result = Path(shutil.move(str(result), str(dest)))
            return result
        except OSError as e:
            logger.error(f"failed to compress {source}: {e}")
            raise

    @staticmethod
    def _archive_suffix(path: Path) -> Optional[str]:
        name = path.name.lower()
        return next((suffix for suffix in _ARCHIVE_FORMATS if name.endswith(suffix)), None)

    @staticmethod
    def _iter_file(file_obj, chunk_size: int = 65536) -> Iterator[bytes]:
        while chunk := file_obj.read(chunk_size):
            yield chunk

    @staticmethod
    def download(
        uri: Union[str, List[Tuple[str, Optional[Union[str, Path]]]]],
        dest: Optional[Union[str, Path]] = None,
    ) -> Union[Path, List[Path]]:
        if isinstance(uri, list):
            return [Filesystem.download(item_uri, item_dest) for item_uri, item_dest in uri]
        if dest is None:
            filename = Path(httpx.URL(uri).path).name or "download"
            dest_path = Path(tempfile.mkdtemp()) / filename
        else:
            dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream("GET", uri, follow_redirects=True) as response:
                response.raise_for_status()
                with dest_path.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
        except httpx.HTTPError as e:
            logger.error(f"failed to download {uri}: {e}")
            raise
        if Filesystem._archive_suffix(dest_path) is not None:
            return Filesystem.extract(dest_path, dest_path.parent)
        return dest_path

    @staticmethod
    def upload(
        path: Union[str, Path, List[Tuple[Union[str, Path], str]]],
        uri: Optional[str] = None,
    ) -> Union[str, List[str]]:
        if isinstance(path, list):
            return [Filesystem.upload(item_path, item_uri) for item_path, item_uri in path]
        source = Path(path)
        if source.is_dir():
            suffix = Filesystem._archive_suffix(Path(httpx.URL(uri).path)) or ".zip"
            archive_path = Path(tempfile.mkdtemp()) / f"{source.name}{suffix}"
            source = Filesystem.compress(source, archive_path)
        try:
            with source.open("rb") as f:
                response = httpx.put(uri, content=Filesystem._iter_file(f))
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"failed to upload {source} to {uri}: {e}")
            raise
        return uri

    @staticmethod
    def move(src: Union[str, Path], dst: Union[str, Path]) -> Path:
        try:
            return Path(shutil.move(str(src), str(dst)))
        except (FileNotFoundError, shutil.Error) as e:
            logger.error(f"failed to move {src} to {dst}: {e}")
            raise

    @staticmethod
    def delete(path: Union[str, Path]) -> None:
        target = Path(path)
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except FileNotFoundError:
            logger.error(f"path not found: {path}")
            raise

    @staticmethod
    def list(path: Union[str, Path], pattern: str = "*") -> List[Path]:
        return [entry for entry in Path(path).glob(pattern)]
