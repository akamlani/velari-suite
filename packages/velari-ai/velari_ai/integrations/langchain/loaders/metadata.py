from __future__ import annotations

from    dataclasses import dataclass, field
from    typing import Any, Dict, List, Optional


@dataclass
class Metadata(object):
    """Metadata every loaded document has: where it came from, when, and its content size.

    `PageMetadata` and `FileMetadata` extend this with attributes specific to their content.

    Examples:
        >>> metadata = Metadata()
        >>> metadata.source.info.relative = "data/cs/datasets/faq.csv"
        >>> metadata.source.properties.extension = "csv"
        >>> pd.json_normalize(asdict(metadata), sep=".").dropna(axis=1).to_dict("records")[0]
        {'source.info.relative': 'data/cs/datasets/faq.csv', 'source.properties.extension': 'csv'}
    """
    @dataclass
    class Source:
        @dataclass
        class Info:
            uri:      Optional[str]   = field(default=None)
            relative: Optional[str]   = field(default=None)
            name:     Optional[str]   = field(default=None)

        @dataclass
        class Properties:
            extension: Optional[str]  = field(default=None)
            is_remote: Optional[bool] = field(default=None)

        info:       Info        = field(default_factory=Info)
        properties: Properties  = field(default_factory=Properties)

    @dataclass
    class Timestamp:
        acquired_at: Optional[str]  = field(default=None)

    @dataclass
    class Content:
        @dataclass
        class Metrics:
            length:   Optional[int]    = field(default=None)
            char_sz:  Optional[int]    = field(default=None)
            word_sz:  Optional[int]    = field(default=None)
            token_sz: Optional[int]    = field(default=None)
            page_sz:  Optional[float]  = field(default=None)

        # page content is not part of metadata
        preview: Optional[str]  = field(default=None)
        metrics: Metrics        = field(default_factory=Metrics)

    source:    Source     = field(default_factory=Source)
    timestamp: Timestamp  = field(default_factory=Timestamp)
    content:   Content    = field(default_factory=Content)


@dataclass
class PageMetadata(Metadata):
    """Metadata for a loaded web page: the base metadata plus the page's own title, description, topic, and links.

    Examples:
        >>> metadata = PageMetadata()
        >>> metadata.source.info.uri = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        >>> metadata.page.title = "LLM Powered Autonomous Agents"
        >>> df_metadata = pd.json_normalize(asdict(metadata), sep=".").dropna(axis=1)
        >>> df_metadata.columns.tolist()
        ['source.info.uri', 'page.title']
    """
    @dataclass
    class Page:
        title:              Optional[str]        = field(default=None)
        description:        Optional[str]        = field(default=None)
        description_length: Optional[int]        = field(default=None)
        topic:              Optional[str]        = field(default=None)
        language:           Optional[str]        = field(default=None)
        links:              Optional[List[str]]  = field(default=None)

    page: Page  = field(default_factory=Page)


@dataclass
class FileMetadata(Metadata):
    """Metadata for a loaded local file, kept at two levels: the file (`source`) and one record of it (`record`).

    `source` extends the base source with the file's shape and is identical on every document from the file;
    `record` only applies when a document is one record (row) of it: `index` is its row position (set when loaded),
    `fields` its column values (set by `LocalFileLoader.enrich()`).

    Examples:
        >>> metadata = FileMetadata()
        >>> metadata.source.table.num_rows = 150
        >>> metadata.record.fields["category"] = "Billing"
        >>> df_metadata = pd.json_normalize(asdict(metadata), sep=".").dropna(axis=1)
        >>> df_metadata.columns.tolist()
        ['source.table.num_rows', 'record.fields.category']
    """
    @dataclass
    class Source(Metadata.Source):
        @dataclass
        class Table:
            num_rows:    Optional[int]  = field(default=None)
            num_columns: Optional[int]  = field(default=None)

        table: Table  = field(default_factory=Table)

    @dataclass
    class Record:
        index:  Optional[int]   = field(default=None)
        fields: Dict[str, Any]  = field(default_factory=dict)

    # deliberately narrows Metadata.source to the extended Source (info + properties + table)
    source: Source  = field(default_factory=Source)  # pyright: ignore[reportIncompatibleVariableOverride]
    record: Record  = field(default_factory=Record)
