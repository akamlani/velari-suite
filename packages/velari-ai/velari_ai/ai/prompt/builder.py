from    pathlib     import Path
from    typing      import Any, Dict, Optional
from    dataclasses import dataclass
# package modules
from    velari_core.core import read_root_dir
from    velari_core.core.io.filesystem import Filesystem


@dataclass(frozen=True)
class PromptSource:
    """Parsed content of one prompt source: system/user templates plus free-form metadata."""
    meta:   Dict[str, Any]
    system: str
    user:   str


class PromptBuilder(object):
    """Locates and parses prompt YAML files into `PromptSource` records — framework-agnostic.

    Args:
        prompt_dir (Optional[Path]): Directory containing prompt YAML files; defaults to
            `<repo_root>/config/prompts/templates` when omitted.
    """
    def __init__(self, prompt_dir: Optional[Path] = None) -> None:
        self._prompt_dir = prompt_dir or Path(read_root_dir()) / "config" / "prompts" / "templates"

    def load_file(self, prompt_name: str, task_name: Optional[str] = None, prefix: str = "tasks") -> PromptSource:
        """Read one prompt YAML file into a PromptSource.

        Args:
            prompt_name (str): File name within this builder's `prompt_dir`, e.g. `"tasks.yaml"`.
            task_name (Optional[str]): When given, look up `<prefix>.<task_name>` inside the file
                instead of reading `meta`/`system`/`user` from the file root — for files holding one
                or more named tasks under a top-level qualifying key.
            prefix (str): Top-level key `task_name` is looked up under when given; defaults to
                `"tasks"` (matches `config/prompts/tasks.yaml`/`templates/tasks.yaml`'s shape).

        Returns:
            PromptSource: The resolved entry's `meta`/`system`/`user` fields.

        Raises:
            ValueError: If `task_name` is given but absent from the file's `<prefix>:` mapping.

        Examples:
            >>> builder = PromptBuilder()
            >>> source = builder.load_file("tasks.yaml", task_name="classification_task")
            >>> source.meta
            {'task': 'content and query classification'}
        """
        prompt_cfg = Filesystem.read(self._prompt_dir / prompt_name)
        if task_name is not None:
            task_group = getattr(prompt_cfg, prefix)
            task_entry = task_group.get(task_name)
            if task_entry is None:
                raise ValueError(
                    f"Unknown task {task_name!r} in {prompt_name!r} — choices: {list(task_group.keys())}"
                )
            prompt_cfg = task_entry
        return PromptSource(meta=dict(prompt_cfg.meta), system=prompt_cfg.system, user=prompt_cfg.user)
