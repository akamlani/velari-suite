from typing import Any
from pypdf import PdfReader, PageObject
from datetime import datetime

from ..utils import trsfrm_camelcase_to_snakecase


class PartitionPdf(object):
    def __init__(self, uri: str | Any):
        self.uri = uri
        self.reader = PdfReader(uri, strict=False)
        # individual document level info
        self.num_pages = self.reader.get_num_pages()

        # read metadata
        metadata = {trsfrm_camelcase_to_snakecase(key.lstrip("/")): value for key, value in self.reader.metadata.items()}
        # transform date
        if "creation_date" in metadata and isinstance(metadata["creation_date"], str):
            metadata["creation_date"] = self.parse_pdf_date(metadata["creation_date"])
        if "mod_date" in metadata and isinstance(metadata["mod_date"], str):
            metadata["mod_date"] = self.parse_pdf_date(metadata["mod_date"])

        self.doc_properties = dict(
            num_pages=self.num_pages,
            is_encrypted=self.reader.is_encrypted,
            metadata=metadata,
            page_info=dict(
                page_header=self.reader.pdf_header, page_mode=self.reader.page_mode, page_labels=self.reader.page_labels
            ),
        )
        # list of page objects
        self.pages = list(self.reader.pages)

    def parse_pdf_date(self, date_str: str) -> str:
        if not date_str.startswith("D:"):
            return date_str
        try:
            return datetime.strptime(date_str[2:16], "%Y%m%d%H%M%S").strftime("%B %d, %Y %H:%M:%S")
        except ValueError:
            return date_str

    def __len__(self):
        return len(self.reader.pages)

    def read(self) -> str:
        return "\n".join([page.extract_text().strip() for page in self.reader.pages])
