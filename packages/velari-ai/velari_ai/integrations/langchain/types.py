from __future__ import annotations

from    dataclasses import dataclass, field
from    typing import Any, Dict, Optional


@dataclass
class ContextSchema:
    """Runtime context for Agent.run() — passed through to LangGraph's `context=` at invoke time."""
    experiment_name: Optional[str]  = field(default=None)
    seed:            Optional[int]  = field(default=None)
    install_dir:     Optional[str]  = field(default=None)
    username:        Optional[str]  = field(default=None)
    agent_id:        Optional[str]  = field(default=None)
    extra:           Dict[str, Any] = field(default_factory=dict)
