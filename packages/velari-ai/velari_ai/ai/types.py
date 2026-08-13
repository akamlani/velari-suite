
from enum import StrEnum, auto
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Self, TypedDict, Literal, Union
from omegaconf import DictConfig

class ProviderMode(StrEnum):
    CHAT                    = auto()
    RESPONSES               = auto()


class ProviderName(StrEnum):
    OPENAI                  = auto()
    HUGGINGFACE             = auto()
    ANTHROPIC               = auto()
    SENTENCE_TRANSFORMERS   = auto()

class Message(TypedDict):
    role:    Literal["user", "assistant"]
    content: str


@dataclass
class ModelConfig:
    """Model-level settings for an agent — the provider:model string plus provider-specific kwargs."""
    model: str            = field(default="openai:gpt-4o-mini")
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, entry: Union[DictConfig, Dict[str, Any]]) -> Self:
        """Build a ModelConfig from one agent's `model_config:` YAML entry.

        Args:
            entry (Union[DictConfig, Dict[str, Any]]): Raw `model_config:` mapping —
                `model` is a known field; anything else (temperature, api_key,
                max_tokens, ...) lands in `extra`.

        Returns:
            Self: Ready for `Agent(model_config=...)`.
        """
        known  = {f.name for f in fields(cls) if f.name != "extra"}
        kwargs = {str(k): v for k, v in entry.items() if k in known}
        extra  = {str(k): v for k, v in entry.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class AgentConfig:
    """Agent-level settings — identity and tool-loop behavior."""
    name:           Optional[str]  = field(default=None)
    stream:         bool           = field(default=False)
    max_tool_calls: int            = field(default=5)
    extra:          Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, entry: Union[DictConfig, Dict[str, Any]]) -> Self:
        """Build an AgentConfig from one agent's `agent_config:` YAML entry.

        Args:
            entry (Union[DictConfig, Dict[str, Any]]): Raw `agent_config:` mapping —
                `name`/`stream`/`max_tool_calls` are known fields; anything else (e.g.
                `default_thread_id`) lands in `extra`.

        Returns:
            Self: Ready for `Agent(agent_config=...)`.
        """
        known  = {f.name for f in fields(cls) if f.name != "extra"}
        kwargs = {str(k): v for k, v in entry.items() if k in known}
        extra  = {str(k): v for k, v in entry.items() if k not in known}
        return cls(**kwargs, extra=extra)
