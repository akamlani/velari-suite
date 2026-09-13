from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Any, ContextManager, Dict, Optional, Protocol
from omegaconf import DictConfig


@dataclass
class LLMCallResult:
    """Provider/framework-agnostic summary of one LLM call, for tracing."""
    output:            str
    prompt_tokens:     Optional[int]            = field(default=None)
    completion_tokens: Optional[int]            = field(default=None)
    user_id:           Optional[str]            = field(default=None)
    metadata:          Optional[Dict[str, Any]] = field(default=None)


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...


class TracerLike(Protocol):
    def start_as_current_span(self, name: str) -> ContextManager[SpanLike]: ...


class TracingConnector(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: DictConfig) -> TracingConnector: ...

    @abstractmethod
    def get_tracer(self, name: str, version: Optional[str] = None) -> TracerLike: ...

    @property
    @abstractmethod
    def config(self) -> Any: ...

    @property
    @abstractmethod
    def url(self) -> str: ...

    @property
    @abstractmethod
    def client(self) -> Any: ...
